//! Original device-produced model scratch, policy derivative and critic work.
//! Allocation and metadata are cold; only actual retained producers charge.

use super::*;

const TALLY_WORDS: usize = 11;

#[repr(C)]
#[derive(Clone, Copy, Default)]
struct PublicationPolicyWorkEntry {
    tally: u64,
    ceiling: [u64; 9],
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
struct PublicationPolicyWorkInput {
    entries: u64,
    count: u64,
}

// SAFETY: both C-layout mirrors contain only initialized fixed-width integers;
// their sizes, alignment and offsets are checked against the device contract.
unsafe impl DeviceRepr for PublicationPolicyWorkEntry {}
unsafe impl DeviceRepr for PublicationPolicyWorkInput {}
const _: () = assert!(size_of::<PublicationPolicyWorkEntry>() == 80);
const _: () = assert!(size_of::<PublicationPolicyWorkInput>() == 16);
const _: () = assert!(std::mem::align_of::<PublicationPolicyWorkEntry>() == 8);
const _: () = assert!(std::mem::align_of::<PublicationPolicyWorkInput>() == 8);
const _: () = assert!(std::mem::offset_of!(PublicationPolicyWorkEntry, ceiling) == 8);
const _: () = assert!(std::mem::offset_of!(PublicationPolicyWorkInput, count) == 8);

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub(super) enum PolicyWorkKey {
    Local(u64, usize),
    Imported(ImportedPolicyContributionKey),
    Critic,
    ModelScratch,
    UpdateCanary,
}

#[derive(Clone)]
pub(super) struct PolicyWorkComponent {
    pub(super) key: PolicyWorkKey,
    pub(super) issuer: Arc<()>,
    pub(super) token: u64,
    pub(super) slot: usize,
    pub(super) tally: DeviceMemoryView<u64>,
    pub(super) ceiling: [u64; 9],
}

struct FrozenPolicyWorkMetadata {
    entries: Vec<PublicationPolicyWorkEntry>,
    input: PublicationPolicyWorkInput,
    ceiling: [u64; 9],
}

struct PolicyWorkBank {
    tallies: TrackedCudaSlice<u64>,
    entries: TrackedCudaSlice<PublicationPolicyWorkEntry>,
    input: TrackedCudaSlice<PublicationPolicyWorkInput>,
    components: Vec<PolicyWorkComponent>,
    keys: BTreeMap<PolicyWorkKey, usize>,
    frozen: Option<FrozenPolicyWorkMetadata>,
}

/// A complete cold reservation retains both banks and every original source.
/// Unexecuted registered components remain zero; capacity is never a charge.
pub(super) struct PolicyWorkBuffers {
    banks: [PolicyWorkBank; 2],
    capacity: usize,
}

impl PolicyWorkBuffers {
    pub(super) fn bytes(capacity: usize) -> Result<usize, SemanticTransitionError> {
        capacity
            .checked_mul(TALLY_WORDS * size_of::<u64>() + size_of::<PublicationPolicyWorkEntry>())
            .and_then(|bytes| bytes.checked_mul(2))
            .and_then(|bytes| bytes.checked_add(2 * size_of::<PublicationPolicyWorkInput>()))
            .ok_or(SemanticTransitionError::GenerationExhausted)
    }

    pub(super) fn allocate(
        reservation: &mut GpuMemoryReservation,
        capacity: usize,
    ) -> Result<Self, SemanticTransitionError> {
        Self::bytes(capacity)?;
        let words = capacity
            .checked_mul(TALLY_WORDS)
            .ok_or(SemanticTransitionError::GenerationExhausted)?;
        let mut allocate_bank = || -> Result<PolicyWorkBank, SemanticTransitionError> {
            Ok(PolicyWorkBank {
                tallies: reservation
                    .alloc(words)
                    .map_err(|error| runtime_error("original policy work allocation", error))?,
                entries: reservation.alloc(capacity).map_err(|error| {
                    runtime_error("original policy work entry allocation", error)
                })?,
                input: reservation.alloc(1).map_err(|error| {
                    runtime_error("original policy work input allocation", error)
                })?,
                components: Vec::new(),
                keys: BTreeMap::new(),
                frozen: None,
            })
        };
        Ok(Self {
            banks: [allocate_bank()?, allocate_bank()?],
            capacity,
        })
    }

    pub(super) fn register(
        &mut self,
        bank: usize,
        key: PolicyWorkKey,
        issuer: Arc<()>,
        token: u64,
        ceiling: [u64; 9],
    ) -> Result<(), SemanticTransitionError> {
        let bank = self
            .banks
            .get_mut(bank)
            .ok_or(SemanticTransitionError::ObservationMismatch)?;
        if let Some(slot) = bank.keys.get(&key) {
            let original = bank
                .components
                .get(*slot)
                .ok_or(SemanticTransitionError::ObservationMismatch)?;
            return if Arc::ptr_eq(&issuer, &original.issuer)
                && original.token == token
                && original.ceiling == ceiling
            {
                Ok(())
            } else {
                Err(SemanticTransitionError::ObservationMismatch)
            };
        }
        if bank.frozen.is_some() {
            return Err(publication_input_error(
                "original policy work registration is already frozen",
            ));
        }
        if matches!(key, PolicyWorkKey::Local(original, source_bank) if original != token || source_bank > 1)
        {
            return Err(SemanticTransitionError::ObservationMismatch);
        }
        let slot = bank.components.len();
        if slot >= self.capacity {
            return Err(publication_input_error(
                "original policy work exceeds its complete cold reservation",
            ));
        }
        let begin = slot
            .checked_mul(TALLY_WORDS)
            .ok_or(SemanticTransitionError::GenerationExhausted)?;
        let end = begin
            .checked_add(TALLY_WORDS)
            .ok_or(SemanticTransitionError::GenerationExhausted)?;
        let tally = bank
            .tallies
            .try_slice(begin..end)
            .ok_or(SemanticTransitionError::ObservationMismatch)?;
        bank.components.push(PolicyWorkComponent {
            key,
            issuer,
            token,
            slot,
            tally,
            ceiling,
        });
        bank.keys.insert(key, slot);
        Ok(())
    }

    pub(super) fn counter(
        &self,
        bank: usize,
        key: PolicyWorkKey,
        issuer: &Arc<()>,
        token: u64,
    ) -> Result<DeviceMemoryView<u64>, SemanticTransitionError> {
        let bank = self.bank(bank)?;
        Self::frozen(bank)?;
        let component = bank
            .keys
            .get(&key)
            .and_then(|slot| bank.components.get(*slot))
            .ok_or(SemanticTransitionError::ObservationMismatch)?;
        if !Arc::ptr_eq(issuer, &component.issuer) || token != component.token {
            return Err(SemanticTransitionError::ObservationMismatch);
        }
        Ok(component.tally.clone())
    }

    pub(super) fn freeze(&mut self, bank: usize) -> Result<(), SemanticTransitionError> {
        let bank = self
            .banks
            .get_mut(bank)
            .ok_or(SemanticTransitionError::ObservationMismatch)?;
        if bank.frozen.is_some() {
            return Ok(());
        }
        let mut ceiling = [0u64; 9];
        let mut entries = Vec::with_capacity(bank.components.len());
        for (slot, component) in bank.components.iter().enumerate() {
            if component.slot != slot || bank.keys.get(&component.key) != Some(&slot) {
                return Err(SemanticTransitionError::ObservationMismatch);
            }
            for (aggregate, bound) in ceiling.iter_mut().zip(component.ceiling) {
                *aggregate = aggregate
                    .checked_add(bound)
                    .ok_or(SemanticTransitionError::GenerationExhausted)?;
            }
            entries.push(PublicationPolicyWorkEntry {
                tally: *component.tally.device_ptr(),
                ceiling: component.ceiling,
            });
        }
        let count = u64::try_from(entries.len())
            .map_err(|_| SemanticTransitionError::GenerationExhausted)?;
        bank.frozen = Some(FrozenPolicyWorkMetadata {
            entries,
            input: PublicationPolicyWorkInput {
                entries: bank.entries.device_ptr_value(),
                count,
            },
            ceiling,
        });
        Ok(())
    }

    pub(super) fn native_ceiling(&self, bank: usize) -> Result<[u64; 9], SemanticTransitionError> {
        Ok(Self::frozen(self.bank(bank)?)?.ceiling)
    }

    // The caller installs this immutable owner before staging and retains each
    // write in its original metadata ledger before any fallible enqueue. The
    // same ledger resolves retries; staging never enters device execution.
    pub(super) fn append_metadata_writes(
        &self,
        stream: &Arc<CudaStream>,
        writes: &mut Vec<state_restoration::OriginalDeviceWrite>,
    ) -> Result<(), SemanticTransitionError> {
        for bank in &self.banks {
            let frozen = Self::frozen(bank)?;
            writes.push(state_restoration::OriginalDeviceWrite::new(
                stream,
                &vec![0u64; bank.tallies.len()],
                bank.tallies.view(),
                false,
            )?);
            writes.push(state_restoration::OriginalDeviceWrite::new(
                stream,
                &frozen.entries,
                bank.entries.slice(..frozen.entries.len()),
                false,
            )?);
            writes.push(state_restoration::OriginalDeviceWrite::new(
                stream,
                &[frozen.input],
                bank.input.view(),
                false,
            )?);
        }
        Ok(())
    }

    pub(super) fn attach(
        &self,
        bank: usize,
        descriptor: &mut Descriptor,
        recorder: &mut LaunchRecorder,
    ) -> Result<(), SemanticTransitionError> {
        let bank = self.bank(bank)?;
        let frozen = Self::frozen(bank)?;
        descriptor.policy_work = if frozen.entries.is_empty() {
            0
        } else {
            recorder.read(&bank.input);
            recorder.read(&bank.entries.slice(..frozen.entries.len()));
            recorder.read(&bank.tallies);
            bank.input.device_ptr_value()
        };
        Ok(())
    }

    // The authentic completed original graph wait is checked by the enclosing
    // Session before these retained views enter its existing terminal reader.
    pub(super) fn components(
        &self,
        bank: usize,
    ) -> Result<Vec<PolicyWorkComponent>, SemanticTransitionError> {
        let bank = self.bank(bank)?;
        Self::frozen(bank)?;
        Ok(bank.components.clone())
    }

    fn bank(&self, bank: usize) -> Result<&PolicyWorkBank, SemanticTransitionError> {
        self.banks
            .get(bank)
            .ok_or(SemanticTransitionError::ObservationMismatch)
    }

    fn frozen(bank: &PolicyWorkBank) -> Result<&FrozenPolicyWorkMetadata, SemanticTransitionError> {
        bank.frozen.as_ref().ok_or_else(|| {
            publication_input_error("original policy work requires its complete immutable freeze")
        })
    }
}
