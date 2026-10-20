use super::*;
use std::collections::{BTreeMap, BTreeSet};
use std::sync::Weak;

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum SemanticSourceContentPhase {
    Model,
    PackPolicy,
    FinalizeContinuation,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SemanticSourceContentKind {
    BindModel,
    Capture,
    Verify,
}

#[derive(Clone)]
pub struct SemanticSourceContentFamily {
    geometry: Weak<prepared_source_content::SourceContentGeometryOwner>,
    phase: SemanticSourceContentPhase,
    kind: SemanticSourceContentKind,
    prepared_section: Option<(SemanticTransitionKind, usize)>,
}

impl SemanticSourceContentFamily {
    pub(super) fn new(
        geometry: &Arc<prepared_source_content::SourceContentGeometryOwner>,
        phase: SemanticSourceContentPhase,
        kind: SemanticSourceContentKind,
    ) -> Self {
        Self {
            geometry: Arc::downgrade(geometry),
            phase,
            kind,
            prepared_section: None,
        }
    }

    pub(super) fn for_prepared_section(
        geometry: &Arc<prepared_source_content::SourceContentGeometryOwner>,
        phase: SemanticSourceContentPhase,
        kind: SemanticSourceContentKind,
        prepared_section: (SemanticTransitionKind, usize),
    ) -> Self {
        Self {
            geometry: Arc::downgrade(geometry),
            phase,
            kind,
            prepared_section: Some(prepared_section),
        }
    }

    pub fn phase(&self) -> SemanticSourceContentPhase {
        self.phase
    }

    pub fn kind(&self) -> SemanticSourceContentKind {
        self.kind
    }

    pub fn prepared_section(&self) -> Option<(SemanticTransitionKind, usize)> {
        self.prepared_section
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SemanticSourceTensorDescriptor {
    pub layout: SemanticTensorLayout,
    pub logical_begin: u64,
    pub logical_end: u64,
    pub storage_slot: u64,
    // Only an actual registered producer version group has a slot. None makes
    // no counter claim; native byte/content guards remain mandatory either way.
    pub version_slot: Option<u64>,
    pub allocation_slot: u64,
    pub byte_offset: u64,
}

#[derive(Clone)]
pub struct SemanticSourceAllocationDescriptor<A = DeviceAllocationProvenance> {
    pub slot: u64,
    pub span_bytes: u64,
    pub native_allocation: Option<A>,
}

pub struct SemanticSourceContentEvent<A = DeviceAllocationProvenance> {
    pub family: SemanticSourceContentFamily,
    pub receiver_slot: u64,
    pub output_slot: Option<u64>,
    pub tensors: Vec<SemanticSourceTensorDescriptor>,
    pub allocations: Vec<SemanticSourceAllocationDescriptor<A>>,
}

/// One original prepared capture or verification, before any numerical effect.
#[derive(Clone)]
pub struct SemanticPreparedTensorContentEvent {
    pub kind: SemanticSourceContentKind,
    pub receiver_slot: u64,
    pub output_slot: Option<u64>,
    pub tensors: Vec<SemanticSourceTensorDescriptor>,
    pub allocations: Vec<SemanticSourceAllocationDescriptor<SemanticSourceContentAllocation>>,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct SemanticPreparedTensorContentBindings {
    pub policy_slot: Option<u64>,
    pub continuation_slot: Option<u64>,
    pub model_slot: Option<u64>,
}

#[derive(Clone, Copy)]
pub(super) enum PreparedContentBindingKind {
    Policy,
    Continuation,
    Model,
}

struct PreparedContentBank {
    events: Vec<SemanticPreparedTensorContentEvent>,
    owners: Vec<u64>,
    ceilings: Vec<[u64; 9]>,
    ceiling: [u64; 9],
    bindings: SemanticPreparedTensorContentBindings,
    state: Mutex<PreparedContentState>,
}

#[derive(Default)]
struct PreparedContentState {
    cursor: usize,
    pending: bool,
    receiver: Option<(u64, SemanticPreparedStep)>,
    witnesses: BTreeMap<u64, SourceWitnessIdentity>,
    allocations: BTreeMap<u64, (u64, u64, Option<DeviceAllocationProvenance>)>,
    storage: BTreeMap<u64, u64>,
}

pub(super) struct PreparedTensorContentPlan {
    banks: [PreparedContentBank; 2],
    _reservation: Option<SemanticReplayMaterialReservation>,
}

impl PreparedTensorContentPlan {
    pub(super) fn new<'a>(
        session: &SemanticTransitionSession,
        step: &'a SemanticPreparedStep,
        events: &[Vec<SemanticPreparedTensorContentEvent>; 2],
        bindings: &[SemanticPreparedTensorContentBindings; 2],
        capture_owner: impl Fn(
            usize,
            u64,
        )
            -> Result<Option<&'a SemanticPreparedStep>, SemanticTransitionError>,
    ) -> Result<Self, SemanticTransitionError> {
        // Charge the copied descriptors and all construction-custody maps to
        // the original material owner before cloning any caller roster.
        let node = OwnedPublicationMaterial::tree_node_bound(
            size_of::<(u64, (u64, u64, Option<DeviceAllocationProvenance>))>().max(size_of::<(
                u64,
                SourceWitnessIdentity,
            )>(
            )),
            0,
        )?;
        let mut bytes =
            size_of::<Self>() + 2 * size_of::<PreparedContentBank>() + 4 * size_of::<usize>();
        for event in events.iter().flatten() {
            bytes = bytes
                .checked_add(
                    size_of::<SemanticPreparedTensorContentEvent>()
                        + size_of::<[u64; 9]>()
                        + size_of::<u64>(),
                )
                .and_then(|bytes| bytes.checked_add(4 * node))
                .and_then(|bytes| {
                    bytes.checked_add(
                        event
                            .tensors
                            .len()
                            .checked_mul(size_of::<SemanticSourceTensorDescriptor>() + 4 * node)?,
                    )
                })
                .and_then(|bytes| {
                    bytes.checked_add(event.allocations.len().checked_mul(
                        size_of::<
                            SemanticSourceAllocationDescriptor<SemanticSourceContentAllocation>,
                        >() + 2 * node,
                    )?)
                })
                .ok_or(SemanticTransitionError::GenerationExhausted)?;
        }
        let reservation = session
            .replay_material_budget
            .as_ref()
            .map(|budget| {
                budget.reserve_auxiliary(
                    u64::try_from(bytes)
                        .map_err(|_| SemanticTransitionError::GenerationExhausted)?,
                )
            })
            .transpose()?;
        let mut banks = Vec::with_capacity(2);
        for (bank, events) in events.iter().enumerate() {
            let mut outputs = BTreeMap::new();
            let mut witness_slots = BTreeSet::new();
            let mut spans = BTreeMap::new();
            let mut storage = BTreeMap::new();
            let mut ceilings = Vec::with_capacity(events.len());
            let mut owners = Vec::with_capacity(events.len());
            let mut ceiling = [0; 9];
            for event in events {
                if event.tensors.is_empty()
                    || !matches!(
                        (event.kind, event.output_slot),
                        (SemanticSourceContentKind::Capture, Some(_))
                            | (SemanticSourceContentKind::Verify, None)
                    )
                {
                    return Err(publication_input_error(
                        "prepared content requires its complete typed capture/verify roster",
                    ));
                }
                validate_content_coordinates(
                    event
                        .tensors
                        .iter()
                        .map(|tensor| (tensor.layout.role, tensor.layout.index)),
                )?;
                if let Some(output) = event.output_slot {
                    if output == event.receiver_slot
                        || !witness_slots.insert(output)
                        || outputs.insert(output, &event.tensors).is_some()
                    {
                        return Err(publication_input_error(
                            "prepared content capture requires a unique original witness slot",
                        ));
                    }
                } else {
                    witness_slots.insert(event.receiver_slot);
                    if outputs
                        .get(&event.receiver_slot)
                        .is_some_and(|original| **original != event.tensors)
                    {
                        return Err(publication_input_error(
                            "prepared verification changed its original capture geometry",
                        ));
                    }
                }
                let mut used = BTreeSet::new();
                let mut allocations = BTreeMap::new();
                for allocation in &event.allocations {
                    if allocations
                        .insert(allocation.slot, allocation.span_bytes)
                        .is_some()
                        || spans
                            .insert(allocation.slot, allocation.span_bytes)
                            .is_some_and(|original| original != allocation.span_bytes)
                    {
                        return Err(publication_input_error(
                            "prepared content allocation slot changed its original complete span",
                        ));
                    }
                    if let Some(native) = allocation
                        .native_allocation
                        .as_ref()
                        .and_then(SourceContentAllocation::native_provenance)
                    {
                        if native.allocation_bytes() != allocation.span_bytes {
                            return Err(publication_input_error(
                                "prepared content native allocation has a different original span",
                            ));
                        }
                    }
                }
                for tensor in &event.tensors {
                    let bytes = tensor_layout_bytes(&tensor.layout)?;
                    tensor_content_range(
                        &tensor.layout,
                        tensor.logical_begin,
                        tensor.logical_end,
                        bytes,
                    )?;
                    if allocations.get(&tensor.allocation_slot).is_none_or(|span| {
                        tensor
                            .byte_offset
                            .checked_add(bytes as u64)
                            .is_none_or(|end| end > *span)
                    }) || storage
                        .insert(tensor.storage_slot, tensor.allocation_slot)
                        .is_some_and(|original| original != tensor.allocation_slot)
                    {
                        return Err(publication_input_error(
                            "prepared content typed view changed its complete backing allocation",
                        ));
                    }
                    used.insert(tensor.allocation_slot);
                }
                if used.len() != allocations.len() {
                    return Err(publication_input_error(
                        "prepared content has an unused allocation descriptor",
                    ));
                }
                let original = if event.kind == SemanticSourceContentKind::Verify {
                    match capture_owner(bank, event.receiver_slot)? {
                        Some(original) => original,
                        None if bindings[bank].model_slot == Some(event.receiver_slot) => step,
                        None => return Err(publication_input_error(
                            "prepared verification requires its original predeclared capture or structural model witness owner",
                        )),
                    }
                } else {
                    step
                };
                let bound = session.prepared_content_event_native_work_ceiling(
                    original,
                    bank,
                    event,
                    event.kind == SemanticSourceContentKind::Verify,
                    event.kind == SemanticSourceContentKind::Verify
                        && bindings[bank].model_slot == Some(event.receiver_slot),
                )?;
                native_work_bound::add_native_work_ceiling(&mut ceiling, bound)?;
                ceilings.push(bound);
                owners.push(original.token);
            }
            let binding = bindings[bank];
            let kind = session.prepared_transition_kind(step)?;
            let storage = session
                .publication
                .as_ref()
                .ok_or(SemanticTransitionError::NotBound)?;
            if binding.policy_slot.is_some() != (kind == SemanticTransitionKind::Proposal)
                || binding.continuation_slot.is_some() != (kind != SemanticTransitionKind::Drain)
                || binding.model_slot.is_some() != (kind != SemanticTransitionKind::Drain)
            {
                return Err(publication_input_error(
                    "prepared content bindings changed their original step services",
                ));
            }
            for (slot, copies) in [(binding.policy_slot, 1), (binding.continuation_slot, 2)] {
                if let Some(slot) = slot {
                    let event = events
                        .iter()
                        .find(|event| event.output_slot == Some(slot))
                        .ok_or_else(|| {
                            publication_input_error(
                                "prepared intrinsic binding has no original capture witness slot",
                            )
                        })?;
                    let guard = session.prepared_content_event_native_work_ceiling(
                        step, bank, event, true, false,
                    )?;
                    for _ in 0..copies {
                        native_work_bound::add_native_work_ceiling(&mut ceiling, guard)?;
                    }
                }
            }
            if binding.model_slot.is_some_and(|slot| {
                events.iter().any(|event| event.output_slot == Some(slot))
                    || !events.iter().any(|event| {
                        event.kind == SemanticSourceContentKind::Verify
                            && event.receiver_slot == slot
                    })
            }) {
                return Err(publication_input_error(
                    "prepared block binding has no original model verification witness slot",
                ));
            }
            if kind == SemanticTransitionKind::Proposal
                && storage.physical_parameter_roster.is_some()
            {
                native_work_bound::add_native_work_ceiling(
                    &mut ceiling,
                    native_work_bound::numerical_block_snapshot_native_work_ceiling(
                        storage
                            .physical_parameter_roster
                            .as_ref()
                            .expect("declared proposal block roster")
                            .len(),
                    )?,
                )?;
            }
            banks.push(PreparedContentBank {
                events: events.clone(),
                owners,
                ceilings,
                ceiling,
                bindings: binding,
                state: Mutex::new(PreparedContentState::default()),
            });
        }
        Ok(Self {
            banks: banks
                .try_into()
                .map_err(|_| SemanticTransitionError::ObservationMismatch)?,
            _reservation: reservation,
        })
    }

    pub(super) fn native_work_ceiling(&self, bank: usize) -> [u64; 9] {
        self.banks[bank].ceiling
    }

    pub(super) fn require_binding(
        &self,
        bank: usize,
        kind: PreparedContentBindingKind,
        witness: &SemanticTensorContentWitness,
    ) -> Result<(), SemanticTransitionError> {
        let plan = &self.banks[bank];
        let slot = match kind {
            PreparedContentBindingKind::Policy => plan.bindings.policy_slot,
            PreparedContentBindingKind::Continuation => plan.bindings.continuation_slot,
            PreparedContentBindingKind::Model => plan.bindings.model_slot,
        }
        .ok_or_else(|| {
            publication_input_error(
                "prepared intrinsic consumer has no original declared witness slot",
            )
        })?;
        let state = plan
            .state
            .lock()
            .map_err(|_| publication_input_error("prepared content custody is poisoned"))?;
        if state.pending
            || state
                .witnesses
                .get(&slot)
                .is_none_or(|original| !original.matches(witness))
        {
            return Err(publication_input_error(
                "prepared intrinsic consumer changed its original bound witness",
            ));
        }
        Ok(())
    }

    pub(super) fn claim(
        &self,
        session: &SemanticTransitionSession,
        step: &SemanticPreparedStep,
        bank: usize,
        witness: &SemanticTensorContentWitness,
        verify: bool,
    ) -> Result<(), SemanticTransitionError> {
        let plan = &self.banks[bank];
        let mut state = plan
            .state
            .lock()
            .map_err(|_| publication_input_error("prepared content custody is poisoned"))?;
        let event = plan.events.get(state.cursor).ok_or_else(|| {
            publication_input_error(
                "prepared content exceeds its original frozen occurrence roster",
            )
        })?;
        let content = &session.steps[&witness.reader_token].content[witness.index];
        if witness.reader_token != plan.owners[state.cursor] {
            return Err(publication_input_error(
                "prepared capture belongs to another original recorder",
            ));
        }
        if state.pending
            || verify != (event.kind == SemanticSourceContentKind::Verify)
            || !same_source_tensor_geometry(&event.tensors, &content.tensors)
            || content.native_work_ceiling(verify)? != plan.ceilings[state.cursor]
            || matches!(&content.seals, TensorContentSeals::Model(model)
                if model.prepared.as_ref().is_some_and(|original| original.bank == bank))
                != (verify && plan.bindings.model_slot == Some(event.receiver_slot))
        {
            return Err(publication_input_error(
                "prepared content changed its next original typed occurrence or native producer",
            ));
        }
        if verify {
            if witness.reader_token != step.token {
                let original = session.steps[&witness.reader_token]
                    .prepared
                    .as_ref()
                    .and_then(|prepared| prepared.content_plan.as_ref())
                    .ok_or(SemanticTransitionError::ObservationMismatch)?;
                let original = original.banks[bank].state.lock().map_err(|_| {
                    publication_input_error("original content producer custody is poisoned")
                })?;
                if original
                    .witnesses
                    .get(&event.receiver_slot)
                    .is_none_or(|original| !original.matches(witness))
                {
                    return Err(publication_input_error(
                        "prepared verification lost its authentic earlier capture witness",
                    ));
                }
            }
            if let Some(original) = state.witnesses.get(&event.receiver_slot) {
                if !original.matches(witness) {
                    return Err(publication_input_error(
                        "prepared verification changed its original witness receiver",
                    ));
                }
            } else {
                // A retained genuine witness can precede this step's plan.
                // The caller has already authenticated its native owner and
                // exact captured tensor roster, not fabricated a new witness.
                if state
                    .witnesses
                    .values()
                    .any(|original| original.matches(witness))
                {
                    return Err(publication_input_error(
                        "prepared witness slots changed their original alias partition",
                    ));
                }
                state.witnesses.insert(
                    event.receiver_slot,
                    SourceWitnessIdentity::from_witness(witness),
                );
            }
        } else {
            if state
                .witnesses
                .values()
                .any(|original| original.matches(witness))
            {
                return Err(publication_input_error(
                    "prepared capture reused an earlier native witness output",
                ));
            }
            if state.receiver.as_ref().is_some_and(|(slot, original)| {
                *slot != event.receiver_slot || !original.same_handle(step)
            }) {
                return Err(publication_input_error(
                    "prepared capture changed its original Step receiver",
                ));
            }
            state.receiver = Some((event.receiver_slot, step.clone()));
        }
        for (descriptor, tensor) in event.tensors.iter().zip(&content.tensors) {
            let allocation = event
                .allocations
                .iter()
                .find(|allocation| allocation.slot == descriptor.allocation_slot)
                .expect("validated original descriptor");
            if !verify || witness.reader_token == step.token {
                session.require_prepared_content_allocation(
                    step, bank, allocation, descriptor, tensor,
                )?;
            }
            let base = tensor
                .data
                .checked_sub(descriptor.byte_offset)
                .ok_or(SemanticTransitionError::ObservationMismatch)?;
            if allocation.span_bytes != 0 {
                cuda_backing_allocation(
                    base,
                    usize::try_from(allocation.span_bytes)
                        .map_err(|_| SemanticTransitionError::GenerationExhausted)?,
                )?;
            }
            for (&slot, (original_base, original_span, original_native)) in &state.allocations {
                if slot == allocation.slot
                    && (*original_base, *original_span) != (base, allocation.span_bytes)
                    || slot != allocation.slot
                        && allocation.span_bytes != 0
                        && (*original_base, *original_span) == (base, allocation.span_bytes)
                    || original_native
                        .as_ref()
                        .zip(tensor.native_allocation.as_ref())
                        .is_some_and(|(original, actual)| {
                            original.same_allocation(actual) != (slot == allocation.slot)
                        })
                {
                    return Err(publication_input_error(
                        "prepared content changed its original backing alias partition",
                    ));
                }
            }
            state.allocations.insert(
                allocation.slot,
                (
                    base,
                    allocation.span_bytes,
                    tensor.native_allocation.clone(),
                ),
            );
            if state
                .storage
                .insert(descriptor.storage_slot, allocation.slot)
                .is_some_and(|original| original != allocation.slot)
            {
                return Err(publication_input_error(
                    "prepared content storage changed its original allocation",
                ));
            }
        }
        state.pending = true;
        Ok(())
    }

    pub(super) fn complete(
        &self,
        bank: usize,
        witness: &SemanticTensorContentWitness,
    ) -> Result<(), SemanticTransitionError> {
        let plan = &self.banks[bank];
        let mut state = plan
            .state
            .lock()
            .map_err(|_| publication_input_error("prepared content custody is poisoned"))?;
        if !state.pending {
            return Err(SemanticTransitionError::ObservationMismatch);
        }
        let event = &plan.events[state.cursor];
        if let Some(output) = event.output_slot {
            state
                .witnesses
                .insert(output, SourceWitnessIdentity::from_witness(witness));
        }
        state.pending = false;
        state.cursor += 1;
        Ok(())
    }

    pub(super) fn require_complete(&self, bank: usize) -> Result<(), SemanticTransitionError> {
        let plan = &self.banks[bank];
        let state = plan
            .state
            .lock()
            .map_err(|_| publication_input_error("prepared content custody is poisoned"))?;
        if state.pending || state.cursor != plan.events.len() {
            return Err(publication_input_error(
                "prepared branch omitted an original content occurrence",
            ));
        }
        Ok(())
    }
}

impl SemanticTransitionSession {
    pub(super) fn require_prepared_content_binding(
        &self,
        step: &SemanticPreparedStep,
        bank: usize,
        kind: PreparedContentBindingKind,
        witness: &SemanticTensorContentWitness,
    ) -> Result<(), SemanticTransitionError> {
        self.steps
            .get(&step.token)
            .and_then(|owner| owner.prepared.as_ref())
            .and_then(|prepared| prepared.content_plan.as_ref())
            .ok_or_else(|| {
                publication_input_error("prepared intrinsic consumer has no original content plan")
            })?
            .require_binding(bank, kind, witness)
    }

    pub(super) fn current_prepared_content_recorder(
        &self,
    ) -> Result<(SemanticPreparedStep, usize), SemanticTransitionError> {
        let build = self
            .prepared_segment
            .as_ref()
            .ok_or(SemanticTransitionError::NotCaptured)?;
        let mut active = build.tokens.iter().filter_map(|token| {
            self.steps
                .get(token)
                .and_then(|owner| owner.prepared.as_ref())
                .and_then(|prepared| prepared.model_work.as_ref())
                .and_then(|work| work.capture_bank.map(|bank| (*token, bank)))
        });
        let (token, bank) = active.next().ok_or_else(|| {
            publication_input_error("prepared content has no original active recording bank")
        })?;
        if active.next().is_some() {
            return Err(SemanticTransitionError::ObservationMismatch);
        }
        let step = SemanticPreparedStep {
            issuer: Arc::clone(&build.issuer),
            scope: Arc::clone(&build.scope),
            token,
        };
        self.checked_prepared_step(&step, true)?;
        Ok((step, bank))
    }
    fn prepared_content_geometry_owner(
        &self,
        allocation: &prepared_source_content::SemanticSourceGeometryAllocation,
    ) -> Result<&Arc<prepared_source_content::SourceContentGeometryOwner>, SemanticTransitionError>
    {
        self.actor_refresh
            .as_ref()
            .and_then(|staged| {
                staged
                    .proof
                    .inner
                    .content_geometry
                    .iter()
                    .map(|geometry| &geometry.owner)
                    .find(|owner| allocation.belongs_to(owner))
            })
            .ok_or_else(|| {
                publication_input_error(
                    "prepared content geometry belongs to another original child",
                )
            })
    }

    fn require_prepared_content_allocation(
        &self,
        step: &SemanticPreparedStep,
        bank: usize,
        allocation: &SemanticSourceAllocationDescriptor<SemanticSourceContentAllocation>,
        descriptor: &SemanticSourceTensorDescriptor,
        tensor: &PreparedSemanticTensor,
    ) -> Result<(), SemanticTransitionError> {
        match &allocation.native_allocation {
            Some(SemanticSourceContentAllocation::Geometry(original)) => {
                let owner = self.prepared_content_geometry_owner(original)?;
                owner.require_allocation(original, allocation.span_bytes)?;
                owner.require_prepared_allocation(
                    self,
                    step,
                    (self.prepared_transition_kind(step)?, bank),
                    original,
                    descriptor,
                    tensor,
                )
            }
            Some(SemanticSourceContentAllocation::Native(original)) => {
                if tensor.native_allocation.as_ref().is_none_or(|actual| {
                    !original.same_allocation(actual)
                        || actual.allocation_bytes() != allocation.span_bytes
                }) {
                    return Err(publication_input_error(
                        "prepared content changed its original native allocation or byte offset",
                    ));
                }
                Ok(())
            }
            None if tensor.native_allocation.is_none() => Ok(()),
            None => Err(publication_input_error(
                "prepared native content has no original allocation descriptor",
            )),
        }
    }

    fn prepared_content_native_classification(
        &self,
        step: &SemanticPreparedStep,
        bank: usize,
        descriptor: &SemanticSourceTensorDescriptor,
        native: &DeviceAllocationProvenance,
    ) -> Result<(bool, bool), SemanticTransitionError> {
        let owner = &self.steps[&step.token];
        let inputs = owner
            .inputs
            .as_ref()
            .ok_or(SemanticTransitionError::NotBound)?;
        let bytes = tensor_layout_bytes(&descriptor.layout)?;
        let classify =
            |view: DeviceMemoryView<u8>| -> Result<Option<(bool, bool)>, SemanticTransitionError> {
                let Some(provenance) = view.allocation_provenance() else {
                    return Ok(None);
                };
                if !native.same_allocation(&provenance) {
                    return Ok(None);
                }
                let Some(offset) = descriptor.byte_offset.checked_sub(provenance.byte_offset())
                else {
                    return Ok(None);
                };
                let offset = usize::try_from(offset)
                    .map_err(|_| SemanticTransitionError::GenerationExhausted)?;
                let end = offset
                    .checked_add(bytes)
                    .ok_or(SemanticTransitionError::GenerationExhausted)?;
                let Some(source) = view.try_slice(offset..end) else {
                    return Ok(None);
                };
                let tensor = PreparedSemanticTensor {
                    layout: descriptor.layout,
                    logical_begin: descriptor.logical_begin,
                    logical_end: descriptor.logical_end,
                    data: *source.device_ptr(),
                    source: Some(source),
                    native_allocation: Some(native.clone()),
                    _empty_owner: None,
                };
                if self.owns_pending_model_tensor(step, &tensor)? {
                    return Ok(Some((false, false)));
                }
                if owner.feedback_origin_digest(&tensor)?.is_some() {
                    return Ok(Some((false, true)));
                }
                if inputs.owns_tensor(&tensor)? {
                    return Ok(Some((true, false)));
                }
                Ok(None)
            };
        for field in [
            PublicationBankField::Source,
            PublicationBankField::PrefixExtent,
            PublicationBankField::RingHead,
            PublicationBankField::Terminal,
        ] {
            if let Some(classification) = classify(inputs.field_view(field)?)? {
                return Ok(classification);
            }
        }
        for bank in &inputs.views {
            for view in bank.values() {
                if let Some(classification) = classify(view.clone())? {
                    return Ok(classification);
                }
            }
        }
        for allocation in &inputs.storage.allocations {
            if let Some(classification) = classify(allocation.slice()?.view())? {
                return Ok(classification);
            }
        }
        for allocation in &inputs.private {
            if let Some(classification) = classify(allocation.view())? {
                return Ok(classification);
            }
        }
        for feedback in &owner.feedback {
            for tensor in &feedback.content.tensors {
                if let Some(view) = &tensor.source {
                    if let Some(classification) = classify(view.clone())? {
                        return Ok(classification);
                    }
                }
            }
        }
        let branch = &owner
            .prepared
            .as_ref()
            .ok_or(SemanticTransitionError::NotBound)?
            .branches[bank];
        if let Some(buffers) = branch
            .policy_buffers
            .as_ref()
            .or_else(|| branch.policy.as_ref().map(|policy| &policy.buffers))
        {
            for domain in policy_uniform_domain_layouts(buffers, self.codebooks.input_cells)? {
                let mut layout = domain.layout;
                layout.index = descriptor.layout.index;
                if native.same_allocation(&domain.allocation)
                    && descriptor.layout == layout
                    && descriptor.byte_offset == domain.allocation.byte_offset()
                    && (descriptor.logical_begin, descriptor.logical_end) == (0, 0)
                {
                    return Ok((false, false));
                }
            }
        }
        Err(publication_input_error(
            "prepared content native allocation has no original typed producer",
        ))
    }

    fn prepared_content_event_native_work_ceiling(
        &self,
        step: &SemanticPreparedStep,
        bank: usize,
        event: &SemanticPreparedTensorContentEvent,
        verify: bool,
        model: bool,
    ) -> Result<[u64; 9], SemanticTransitionError> {
        let owner = &self.steps[&step.token];
        let inputs = owner
            .inputs
            .as_ref()
            .ok_or(SemanticTransitionError::NotBound)?;
        if model
            && (!verify
                || !owner.content.iter().any(|content| {
                    matches!(&content.seals, TensorContentSeals::Model(model)
                if model.prepared.as_ref().is_some_and(|original| original.bank == bank))
                        && same_source_tensor_geometry(&event.tensors, &content.tensors)
                }))
        {
            return Err(publication_input_error(
                "prepared model verification changed its original admitted bank roster",
            ));
        }
        let mut ceiling = [0; 9];
        if model {
            let contract = inputs.storage.bank_templates[bank]
                .iter()
                .find(|range| (range.role, range.index) == (44, 0))
                .ok_or(SemanticTransitionError::ObservationMismatch)?;
            native_work_bound::add_native_work_ceiling(
                &mut ceiling,
                native_work_bound::content_digest_ceiling(contract.length_bytes, 21, false)?,
            )?;
        }
        let mut input_guard = false;
        for descriptor in &event.tensors {
            let allocation = event
                .allocations
                .iter()
                .find(|allocation| allocation.slot == descriptor.allocation_slot)
                .ok_or(SemanticTransitionError::ObservationMismatch)?;
            let (native_input, sealed) = match &allocation.native_allocation {
                Some(SemanticSourceContentAllocation::Geometry(original)) => {
                    let geometry = self.prepared_content_geometry_owner(original)?;
                    geometry.require_allocation(original, allocation.span_bytes)?;
                    geometry.classify_tensor(
                        original,
                        descriptor,
                        Some((self.prepared_transition_kind(step)?, bank)),
                    )?
                }
                Some(SemanticSourceContentAllocation::Native(native)) => {
                    self.prepared_content_native_classification(step, bank, descriptor, native)?
                }
                None => (false, false),
            };
            if !model && native_input {
                if !input_guard {
                    native_work_bound::add_native_work_ceiling(
                        &mut ceiling,
                        inputs.native_work_ceiling()?,
                    )?;
                    input_guard = true;
                }
            } else {
                native_work_bound::add_native_work_ceiling(
                    &mut ceiling,
                    native_work_bound::tensor_layout_native_work_ceiling(
                        &descriptor.layout,
                        descriptor.logical_begin,
                        descriptor.logical_end,
                        tensor_layout_bytes(&descriptor.layout)?,
                        verify || sealed,
                    )?,
                )?;
            }
        }
        native_work_bound::native_work_ceiling_units(ceiling)?;
        Ok(ceiling)
    }
}

#[derive(Clone)]
pub enum SemanticSourceContentAllocation {
    Native(DeviceAllocationProvenance),
    Geometry(prepared_source_content::SemanticSourceGeometryAllocation),
}

mod allocation_seal {
    pub trait Sealed {}
    impl Sealed for super::DeviceAllocationProvenance {}
    impl Sealed for super::SemanticSourceContentAllocation {}
    impl Sealed for super::prepared_source_content::SemanticSourceGeometryAllocation {}
}

pub trait SourceContentAllocation: allocation_seal::Sealed + Clone {
    fn native_provenance(&self) -> Option<&DeviceAllocationProvenance>;
    fn original_geometry(
        &self,
    ) -> Option<&prepared_source_content::SemanticSourceGeometryAllocation>;
}

impl SourceContentAllocation for DeviceAllocationProvenance {
    fn native_provenance(&self) -> Option<&DeviceAllocationProvenance> {
        Some(self)
    }
    fn original_geometry(
        &self,
    ) -> Option<&prepared_source_content::SemanticSourceGeometryAllocation> {
        None
    }
}
impl SourceContentAllocation for prepared_source_content::SemanticSourceGeometryAllocation {
    fn native_provenance(&self) -> Option<&DeviceAllocationProvenance> {
        None
    }
    fn original_geometry(
        &self,
    ) -> Option<&prepared_source_content::SemanticSourceGeometryAllocation> {
        Some(self)
    }
}
impl SourceContentAllocation for SemanticSourceContentAllocation {
    fn native_provenance(&self) -> Option<&DeviceAllocationProvenance> {
        match self {
            Self::Native(value) => Some(value),
            Self::Geometry(_) => None,
        }
    }
    fn original_geometry(
        &self,
    ) -> Option<&prepared_source_content::SemanticSourceGeometryAllocation> {
        match self {
            Self::Native(_) => None,
            Self::Geometry(value) => Some(value),
        }
    }
}

struct SourceContentGeometry {
    phase: SemanticSourceContentPhase,
    kind: SemanticSourceContentKind,
    receiver_slot: u64,
    output_slot: Option<u64>,
    tensors: Vec<SemanticSourceTensorDescriptor>,
    allocations: Vec<(u64, u64)>,
    ceiling: [u64; 9],
    prepared_section: Option<(SemanticTransitionKind, usize)>,
    section: usize,
    section_index: usize,
}

#[derive(Clone)]
pub struct SemanticSourceContentOccurrence {
    frozen: Arc<FrozenSourceContentOccurrences>,
    index: usize,
}

impl SemanticSourceContentOccurrence {
    fn geometry(&self) -> &SourceContentGeometry {
        &self.frozen.events[self.index]
    }

    pub fn phase(&self) -> SemanticSourceContentPhase {
        self.geometry().phase
    }

    pub fn kind(&self) -> SemanticSourceContentKind {
        self.geometry().kind
    }

    pub fn prepared_section(&self) -> Option<(SemanticTransitionKind, usize)> {
        self.geometry().prepared_section
    }

    pub(super) fn claim_prepared(
        &self,
        session: &SemanticTransitionSession,
        step: &SemanticPreparedStep,
        receiver: Option<&SemanticTensorContentWitness>,
        tensors: &[PreparedSemanticTensor],
    ) -> Result<SourceContentClaim, SemanticTransitionError> {
        self.frozen
            .claim_prepared(session, step, self, receiver, tensors)
    }

    pub fn receiver_slot(&self) -> u64 {
        self.geometry().receiver_slot
    }

    pub fn output_slot(&self) -> Option<u64> {
        self.geometry().output_slot
    }

    pub fn tensor_layouts(&self) -> &[SemanticSourceTensorDescriptor] {
        &self.geometry().tensors
    }

    pub fn allocation_layouts(&self) -> &[(u64, u64)] {
        &self.geometry().allocations
    }

    pub(super) fn uses_uniform_domains(&self) -> bool {
        self.frozen.geometry.upgrade().is_some_and(|geometry| {
            self.geometry().tensors.iter().any(|tensor| {
                self.frozen.declared_allocations[self.geometry().section]
                    .get(&tensor.allocation_slot)
                    .is_some_and(|allocation| geometry.is_uniform_allocation(allocation))
            })
        })
    }

    pub fn same_handle(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.frozen, &other.frozen) && self.index == other.index
    }

    pub(super) fn native_work_ceiling(&self) -> [u64; 9] {
        self.geometry().ceiling
    }
}

#[derive(Clone)]
struct SourceWitnessIdentity {
    issuer: Arc<()>,
    token: u64,
    index: usize,
}

impl SourceWitnessIdentity {
    fn from_witness(witness: &SemanticTensorContentWitness) -> Self {
        Self {
            issuer: Arc::clone(&witness.issuer),
            token: witness.reader_token,
            index: witness.index,
        }
    }

    fn matches(&self, witness: &SemanticTensorContentWitness) -> bool {
        Arc::ptr_eq(&self.issuer, &witness.issuer)
            && self.token == witness.reader_token
            && self.index == witness.index
    }
}

struct SourcePhysicalCustody {
    allocations: BTreeMap<u64, Option<DeviceAllocationProvenance>>,
    backing_allocations: BTreeMap<u64, (u64, u64)>,
    storage_allocations: BTreeMap<u64, (u64, u64)>,
    witnesses: BTreeMap<u64, SourceWitnessIdentity>,
    prepared_receiver: Option<(u64, SemanticPreparedStep)>,
    tensors: Vec<Vec<PreparedSemanticTensor>>,
}

#[derive(Clone)]
pub(super) struct SourceContentClaim {
    frozen: Arc<FrozenSourceContentOccurrences>,
    index: usize,
    identity: Arc<()>,
}

struct SourceOccurrenceState {
    cursor: usize,
    pending: Option<(usize, Arc<()>, usize)>,
    native_pending: Option<(SourceNativeContentGuard, Arc<()>)>,
    native_completed: BTreeSet<SourceNativeContentGuard>,
    physical: Option<SourcePhysicalCustody>,
}

/// The original native consumers, not additional compiler-visible events.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub(super) enum SourceNativeContentGuard {
    PolicyBinding,
    PolicyAttach,
    ContinuationBinding,
    ContinuationAttach,
    Launch,
    ImportedGroup(usize),
    Retirement,
}

#[derive(Clone)]
pub(super) struct SourceNativeContentClaim {
    frozen: Arc<FrozenSourceContentOccurrences>,
    kind: SourceNativeContentGuard,
    identity: Arc<()>,
}

pub(super) struct FrozenSourceContentOccurrences {
    geometry: Weak<prepared_source_content::SourceContentGeometryOwner>,
    declared_allocations:
        [BTreeMap<u64, prepared_source_content::SemanticSourceGeometryAllocation>; 4],
    events: Vec<SourceContentGeometry>,
    ceiling: [u64; 9],
    native_guards: BTreeMap<SourceNativeContentGuard, (u64, [u64; 9])>,
    state: Mutex<[SourceOccurrenceState; 4]>,
    section_lengths: [usize; 4],
    _reservation: SemanticReplayMaterialReservation,
}

impl FrozenSourceContentOccurrences {
    pub(super) fn policy_binding_dma_ceiling(&self) -> Result<[u64; 9], SemanticTransitionError> {
        if !self
            .native_guards
            .contains_key(&SourceNativeContentGuard::PolicyBinding)
        {
            return Ok([0; 9]);
        }
        let geometry = self
            .geometry
            .upgrade()
            .ok_or(SemanticTransitionError::ObservationMismatch)?;
        let mut ceiling = [0; 9];
        ceiling[0] = 9;
        ceiling[8] = geometry.publication.policy_producer_ceiling[8];
        Ok(ceiling)
    }
    pub(super) fn require_original_occurrence(
        self: &Arc<Self>,
        occurrence: &SemanticSourceContentOccurrence,
    ) -> Result<(), SemanticTransitionError> {
        if !Arc::ptr_eq(self, &occurrence.frozen) {
            return Err(publication_input_error(
                "source content occurrence belongs to another frozen original roster",
            ));
        }
        Ok(())
    }

    pub(super) fn new<A: SourceContentAllocation>(
        recipe: &Arc<prepared_source_content::SourceContentGeometryOwner>,
        events: Vec<SemanticSourceContentEvent<A>>,
    ) -> Result<Arc<Self>, SemanticTransitionError> {
        if events.is_empty() {
            return Err(publication_input_error(
                "source content freeze requires its complete original occurrence roster",
            ));
        }
        let mut bytes = size_of::<Self>()
            .checked_add(2 * size_of::<usize>())
            .ok_or(SemanticTransitionError::GenerationExhausted)?;
        let node = OwnedPublicationMaterial::tree_node_bound(
            size_of::<(u64, Vec<SemanticSourceTensorDescriptor>)>()
                .max(size_of::<(u64, Option<DeviceAllocationProvenance>)>())
                .max(size_of::<(
                    u64,
                    prepared_source_content::SemanticSourceGeometryAllocation,
                )>()),
            0,
        )?;
        for event in &events {
            let tensors = event
                .tensors
                .len()
                .checked_mul(
                    4 * size_of::<SemanticSourceTensorDescriptor>()
                        + 2 * size_of::<PreparedSemanticTensor>()
                        + 8 * node,
                )
                .ok_or(SemanticTransitionError::GenerationExhausted)?;
            let allocations = event
                .allocations
                .len()
                .checked_mul(
                    2 * size_of::<SemanticSourceAllocationDescriptor<A>>()
                        + size_of::<(u64, u64)>()
                        + 4 * node,
                )
                .ok_or(SemanticTransitionError::GenerationExhausted)?;
            bytes = bytes
                .checked_add(tensors)
                .and_then(|bytes| bytes.checked_add(allocations))
                .and_then(|bytes| {
                    bytes.checked_add(
                        size_of::<SourceContentGeometry>()
                            + 2 * size_of::<Vec<PreparedSemanticTensor>>()
                            + 2 * size_of::<SemanticSourceContentOccurrence>()
                            + 4 * size_of::<usize>(),
                    )
                })
                .ok_or(SemanticTransitionError::GenerationExhausted)?;
        }
        bytes = bytes
            .checked_add(8 * (2 * node + 2 * size_of::<usize>()))
            .ok_or(SemanticTransitionError::GenerationExhausted)?;
        let reservation = recipe.seed.proof.owner.budget.reserve_auxiliary(
            u64::try_from(bytes).map_err(|_| SemanticTransitionError::GenerationExhausted)?,
        )?;
        let current = recipe.kind.is_current();
        let mut outputs: [BTreeMap<u64, Vec<SemanticSourceTensorDescriptor>>; 4] =
            std::array::from_fn(|_| BTreeMap::new());
        let mut baselines = [None; 4];
        let mut prepared_receivers = [None; 4];
        let mut previous_phases = [SemanticSourceContentPhase::Model; 4];
        let mut spans = std::array::from_fn::<_, 4, _>(|_| BTreeMap::new());
        let mut provenance: [BTreeMap<u64, Option<DeviceAllocationProvenance>>; 4] =
            std::array::from_fn(|_| BTreeMap::new());
        let mut declared_allocations: [BTreeMap<
            u64,
            prepared_source_content::SemanticSourceGeometryAllocation,
        >; 4] = std::array::from_fn(|_| BTreeMap::new());
        let mut storage_allocations = std::array::from_fn::<_, 4, _>(|_| BTreeMap::new());
        let mut section_lengths = [0usize; 4];
        let mut section_phases = [0u8; 4];
        let mut geometry = Vec::with_capacity(events.len());
        let mut ceiling = [0; 9];
        let mut native_guards = BTreeMap::new();
        for event in events {
            let family_recipe = event.family.geometry.upgrade().ok_or_else(|| {
                publication_input_error("source content family no longer has its original recipe")
            })?;
            if !Arc::ptr_eq(recipe, &family_recipe) {
                return Err(publication_input_error(
                    "source content family belongs to another original recipe",
                ));
            }
            let section = match (current, event.family.prepared_section) {
                (false, None) => 0,
                (true, Some((SemanticTransitionKind::Recompute, bank)))
                    if bank < 2 && event.family.phase != SemanticSourceContentPhase::PackPolicy =>
                {
                    bank
                }
                (true, Some((SemanticTransitionKind::Proposal, bank))) if bank < 2 => 2 + bank,
                _ => {
                    return Err(publication_input_error(
                        "content occurrence changed its original prepared child operation or bank",
                    ))
                }
            };
            let event_index = section_lengths[section];
            section_lengths[section] = event_index
                .checked_add(1)
                .ok_or(SemanticTransitionError::GenerationExhausted)?;
            let outputs = &mut outputs[section];
            let baseline = &mut baselines[section];
            let spans = &mut spans[section];
            let provenance = &mut provenance[section];
            let declared_allocations = &mut declared_allocations[section];
            let storage_allocations = &mut storage_allocations[section];
            if event.family.phase < previous_phases[section] {
                return Err(publication_input_error(
                    "source content occurrences changed their original phase order",
                ));
            }
            previous_phases[section] = event.family.phase;
            section_phases[section] |= match event.family.phase {
                SemanticSourceContentPhase::Model => 1,
                SemanticSourceContentPhase::PackPolicy => 2,
                SemanticSourceContentPhase::FinalizeContinuation => 4,
            };
            match (event.family.kind, event.output_slot) {
                (SemanticSourceContentKind::BindModel, Some(output))
                    if !current
                        && event_index == 0
                        && event.family.phase == SemanticSourceContentPhase::Model
                        && output != event.receiver_slot =>
                {
                    *baseline = Some(output);
                }
                (SemanticSourceContentKind::Verify, None)
                    if current
                        && event_index == 0
                        && event.family.phase == SemanticSourceContentPhase::Model =>
                {
                    *baseline = Some(event.receiver_slot);
                    outputs.insert(event.receiver_slot, event.tensors.clone());
                }
                (SemanticSourceContentKind::Capture, Some(output))
                    if event_index != 0
                        && !outputs.contains_key(&output)
                        && Some(output) != *baseline
                        && output != event.receiver_slot => {}
                (SemanticSourceContentKind::Verify, None) if event_index != 0 => {}
                _ => {
                    return Err(publication_input_error(
                        "source content begins with its original model binding; later captures have new outputs and verification has no output",
                    ));
                }
            }
            if current && event.family.kind == SemanticSourceContentKind::Capture {
                if outputs.contains_key(&event.receiver_slot)
                    || prepared_receivers[section]
                        .is_some_and(|original| original != event.receiver_slot)
                {
                    return Err(publication_input_error(
                        "CURRENT capture receiver must remain its original prepared Step",
                    ));
                }
                prepared_receivers[section] = Some(event.receiver_slot);
            } else if event.family.kind == SemanticSourceContentKind::BindModel {
                // This sole structural receiver denotes the authenticated
                // original parent, not a witness fabricated before execution.
            } else if let Some(receiver) = outputs.get(&event.receiver_slot) {
                if event.family.kind == SemanticSourceContentKind::Verify
                    && receiver != &event.tensors
                {
                    return Err(publication_input_error(
                        "source content verification changed its original capture geometry or aliases",
                    ));
                }
            } else {
                return Err(publication_input_error(
                    "source content receiver is not its original model output or prior capture",
                ));
            }
            if event.tensors.is_empty() {
                return Err(publication_input_error(
                    "source content occurrence requires an actual tensor roster",
                ));
            }
            validate_content_coordinates(
                event
                    .tensors
                    .iter()
                    .map(|tensor| (tensor.layout.role, tensor.layout.index)),
            )?;
            let mut event_allocations = BTreeMap::new();
            for allocation in &event.allocations {
                if event_allocations
                    .insert(allocation.slot, allocation.span_bytes)
                    .is_some()
                    || spans
                        .get(&allocation.slot)
                        .is_some_and(|span| *span != allocation.span_bytes)
                {
                    return Err(publication_input_error(
                        "source content allocation slots must have one original full backing span",
                    ));
                }
                if let Some(original) = allocation
                    .native_allocation
                    .as_ref()
                    .and_then(SourceContentAllocation::original_geometry)
                {
                    recipe.require_allocation(original, allocation.span_bytes)?;
                    recipe.require_allocation_section(original, event.family.prepared_section)?;
                    if declared_allocations.get(&allocation.slot).is_some_and(
                        |previous: &prepared_source_content::SemanticSourceGeometryAllocation| {
                            !previous.same_handle(original)
                        },
                    ) || declared_allocations.iter().any(|(&slot, previous)| {
                        previous.same_handle(original) != (slot == allocation.slot)
                    }) {
                        return Err(publication_input_error(
                            "source content changed its original geometry alias partition",
                        ));
                    }
                    declared_allocations.insert(allocation.slot, original.clone());
                }
                if let Some(actual) = allocation
                    .native_allocation
                    .as_ref()
                    .and_then(SourceContentAllocation::native_provenance)
                {
                    if actual.allocation_bytes() != allocation.span_bytes {
                        return Err(publication_input_error(
                            "source content allocation span differs from its original native allocation",
                        ));
                    }
                    for (&slot, previous) in provenance.iter() {
                        if let Some(previous) = previous {
                            if actual.same_allocation(previous) != (slot == allocation.slot) {
                                return Err(publication_input_error(
                                    "source content allocation slots changed their original backing alias partition",
                                ));
                            }
                        }
                    }
                    provenance.insert(allocation.slot, Some(actual.clone()));
                } else {
                    provenance.entry(allocation.slot).or_insert(None);
                }
                spans.insert(allocation.slot, allocation.span_bytes);
            }
            let mut used_allocations = BTreeSet::new();
            for tensor in &event.tensors {
                let bytes = tensor_layout_bytes(&tensor.layout)?;
                tensor_content_range(
                    &tensor.layout,
                    tensor.logical_begin,
                    tensor.logical_end,
                    bytes,
                )?;
                let span = event_allocations
                    .get(&tensor.allocation_slot)
                    .ok_or_else(|| {
                        publication_input_error(
                            "source content tensor has no original allocation descriptor",
                        )
                    })?;
                let bytes = u64::try_from(bytes)
                    .map_err(|_| SemanticTransitionError::GenerationExhausted)?;
                if tensor
                    .byte_offset
                    .checked_add(bytes)
                    .is_none_or(|end| end > *span)
                {
                    return Err(publication_input_error(
                        "source content tensor exceeds its full original backing allocation",
                    ));
                }
                if storage_allocations
                    .insert(tensor.storage_slot, tensor.allocation_slot)
                    .is_some_and(|slot| slot != tensor.allocation_slot)
                {
                    return Err(publication_input_error(
                        "source content storage slot changed its original allocation",
                    ));
                }
                used_allocations.insert(tensor.allocation_slot);
            }
            if used_allocations.len() != event_allocations.len() {
                return Err(publication_input_error(
                    "source content occurrence contains an unused allocation descriptor",
                ));
            }
            let event_ceiling =
                prepared_source_content::source_geometry_occurrence_native_work_ceiling(
                    recipe,
                    &event,
                    event.family.kind == SemanticSourceContentKind::BindModel
                        || (event.family.kind == SemanticSourceContentKind::Verify
                            && Some(event.receiver_slot) == *baseline),
                )?;
            if !current {
                native_work_bound::add_native_work_ceiling(&mut ceiling, event_ceiling)?;
            }
            if !current && event.family.kind == SemanticSourceContentKind::Capture {
                let policy = event.family.phase == SemanticSourceContentPhase::PackPolicy
                    && event.tensors.len() == recipe.publication.policy_layouts.len()
                    && event
                        .tensors
                        .iter()
                        .zip(recipe.publication.policy_layouts)
                        .all(|(actual, expected)| {
                            actual.layout == expected
                                && actual.logical_begin == 0
                                && actual.logical_end == 0
                        });
                let handle = prepared_source_content::SemanticPreparedSourceContentGeometry {
                    owner: Arc::clone(recipe),
                };
                let blocks = handle.continuation_blocks()?;
                let services = 6 + usize::from(blocks.is_some());
                let continuation = if event.family.phase
                    == SemanticSourceContentPhase::FinalizeContinuation
                    && event.tensors.len() >= services
                {
                    let first = event.tensors.len() - services;
                    let expected = continuation_service_layouts(
                        first,
                        handle.continuation_service_capacity()?,
                        blocks,
                    )?;
                    event.tensors[first..]
                        .iter()
                        .zip(expected)
                        .all(|(actual, expected)| {
                            actual.layout == expected
                                && actual.logical_begin == 0
                                && actual.logical_end == 0
                        })
                } else {
                    false
                };
                if continuation {
                    let original = &recipe.seed.proof.owner;
                    let storage = &recipe.publication.storage;
                    if let Some(blocks) = blocks {
                        native_work_bound::add_native_work_ceiling(
                            &mut ceiling,
                            native_work_bound::numerical_block_snapshot_native_work_ceiling(
                                blocks,
                            )?,
                        )?;
                    }
                    native_work_bound::add_native_work_ceiling(
                        &mut ceiling,
                        native_work_bound::continuation_preparation_geometry_native_work_ceiling(
                            &original.contract,
                            &storage.bank_templates[0],
                            storage.continuation_templates.len(),
                            original.layouts.len(),
                            storage.allocation_bytes.len(),
                            native_work_bound::continuation_copy_capacity_native_work_ceiling(
                                &storage.continuation_templates,
                                |slot| {
                                    storage
                                        .allocation_bytes
                                        .get(slot)
                                        .copied()
                                        .map(|bytes| bytes as u64)
                                        .ok_or(SemanticTransitionError::ObservationMismatch)
                                },
                            )?,
                        )?,
                    )?;
                }
                let kinds: &[SourceNativeContentGuard] = if policy {
                    native_work_bound::add_native_work_ceiling(
                        &mut ceiling,
                        recipe.publication.policy_producer_ceiling,
                    )?;
                    &[
                        SourceNativeContentGuard::PolicyBinding,
                        SourceNativeContentGuard::PolicyAttach,
                    ]
                } else if continuation {
                    &[
                        SourceNativeContentGuard::ContinuationBinding,
                        SourceNativeContentGuard::ContinuationAttach,
                        SourceNativeContentGuard::Launch,
                        SourceNativeContentGuard::ImportedGroup(0),
                        SourceNativeContentGuard::ImportedGroup(1),
                        SourceNativeContentGuard::Retirement,
                    ]
                } else {
                    &[]
                };
                if !kinds.is_empty() {
                    let verification = SemanticSourceContentEvent {
                        family: SemanticSourceContentFamily::new(
                            recipe,
                            event.family.phase,
                            SemanticSourceContentKind::Verify,
                        ),
                        receiver_slot: event
                            .output_slot
                            .expect("validated original capture output"),
                        output_slot: None,
                        tensors: event.tensors.clone(),
                        allocations: event.allocations.clone(),
                    };
                    let native_ceiling =
                        prepared_source_content::source_geometry_occurrence_native_work_ceiling(
                            recipe,
                            &verification,
                            false,
                        )?;
                    for &kind in kinds {
                        if native_guards
                            .insert(kind, (verification.receiver_slot, native_ceiling))
                            .is_some()
                        {
                            return Err(publication_input_error("SOURCE native consumer requires its sole original canonical producer"));
                        }
                        native_work_bound::add_native_work_ceiling(&mut ceiling, native_ceiling)?;
                    }
                }
            }
            if let Some(output) = event.output_slot {
                outputs.insert(output, event.tensors.clone());
            }
            geometry.push(SourceContentGeometry {
                phase: event.family.phase,
                kind: event.family.kind,
                receiver_slot: event.receiver_slot,
                output_slot: event.output_slot,
                tensors: event.tensors,
                allocations: event
                    .allocations
                    .iter()
                    .map(|allocation| (allocation.slot, allocation.span_bytes))
                    .collect(),
                ceiling: event_ceiling,
                prepared_section: event.family.prepared_section,
                section,
                section_index: event_index,
            });
        }
        if current
            && (section_phases[..2].iter().any(|phases| *phases != 5)
                || section_phases[2..].iter().any(|phases| *phases != 7))
        {
            return Err(publication_input_error("CURRENT content freeze requires all four original Recompute/Proposal bank sections"));
        }
        native_work_bound::native_work_ceiling_units(ceiling)?;
        Ok(Arc::new(Self {
            geometry: Arc::downgrade(recipe),
            declared_allocations,
            events: geometry,
            ceiling,
            native_guards,
            state: Mutex::new(std::array::from_fn(|section| SourceOccurrenceState {
                cursor: 0,
                pending: None,
                native_pending: None,
                native_completed: BTreeSet::new(),
                physical: (section_lengths[section] != 0).then(|| SourcePhysicalCustody {
                    allocations: std::mem::take(&mut provenance[section]),
                    backing_allocations: BTreeMap::new(),
                    storage_allocations: BTreeMap::new(),
                    witnesses: BTreeMap::new(),
                    prepared_receiver: None,
                    tensors: Vec::new(),
                }),
            })),
            section_lengths,
            _reservation: reservation,
        }))
    }

    pub(super) fn claim_native(
        self: &Arc<Self>,
        kind: SourceNativeContentGuard,
        witness: &SemanticTensorContentWitness,
    ) -> Result<SourceNativeContentClaim, SemanticTransitionError> {
        let (slot, _) = self.native_guards.get(&kind).ok_or_else(|| {
            publication_input_error(
                "SOURCE native consumer lacks its original frozen producer geometry",
            )
        })?;
        let mut state = self.state.lock().map_err(|_| {
            publication_input_error("original SOURCE native consumer custody is poisoned")
        })?;
        let state = &mut state[0];
        if state.pending.is_some() || state.native_completed.contains(&kind) {
            return Err(publication_input_error(
                "SOURCE native consumer repeated or overlaps another original occurrence",
            ));
        }
        let physical = state
            .physical
            .as_ref()
            .ok_or(SemanticTransitionError::ObservationMismatch)?;
        if physical
            .witnesses
            .get(slot)
            .is_none_or(|original| !original.matches(witness))
        {
            return Err(publication_input_error(
                "SOURCE native consumer changed its original canonical output witness",
            ));
        }
        let identity = if let Some((original, identity)) = &state.native_pending {
            if *original != kind {
                return Err(publication_input_error(
                    "SOURCE native consumer cannot replace its unresolved original operation",
                ));
            }
            Arc::clone(identity)
        } else {
            let identity = Arc::new(());
            state.native_pending = Some((kind, Arc::clone(&identity)));
            identity
        };
        Ok(SourceNativeContentClaim {
            frozen: Arc::clone(self),
            kind,
            identity,
        })
    }

    pub(super) fn occurrence_handles(self: &Arc<Self>) -> Vec<SemanticSourceContentOccurrence> {
        (0..self.events.len())
            .map(|index| SemanticSourceContentOccurrence {
                frozen: Arc::clone(self),
                index,
            })
            .collect()
    }

    pub(super) fn phase_occurrences(
        self: &Arc<Self>,
        phase: SemanticSourceContentPhase,
    ) -> Vec<SemanticSourceContentOccurrence> {
        self.occurrence_handles()
            .into_iter()
            .filter(|occurrence| occurrence.phase() == phase)
            .collect()
    }

    pub(super) fn native_work_ceiling(&self) -> [u64; 9] {
        self.ceiling
    }

    pub(super) fn claim_prepared(
        self: &Arc<Self>,
        session: &SemanticTransitionSession,
        step: &SemanticPreparedStep,
        occurrence: &SemanticSourceContentOccurrence,
        receiver: Option<&SemanticTensorContentWitness>,
        tensors: &[PreparedSemanticTensor],
    ) -> Result<SourceContentClaim, SemanticTransitionError> {
        self.require_original_occurrence(occurrence)?;
        let section = occurrence.prepared_section().ok_or_else(|| {
            publication_input_error("prepared content requires its original CURRENT section")
        })?;
        let geometry = self
            .geometry
            .upgrade()
            .ok_or(SemanticTransitionError::ObservationMismatch)?;
        geometry.require_current_section(session, step, section)?;
        for (descriptor, tensor) in occurrence.geometry().tensors.iter().zip(tensors) {
            if let Some(declared) = self.declared_allocations[occurrence.geometry().section]
                .get(&descriptor.allocation_slot)
            {
                geometry.require_prepared_allocation(
                    session, step, section, declared, descriptor, tensor,
                )?;
            }
        }
        match (occurrence.kind(), receiver) {
            (SemanticSourceContentKind::Capture, None) => {
                let mut state = self.state.lock().map_err(|_| {
                    publication_input_error("source content occurrence custody is poisoned")
                })?;
                let physical = state[occurrence.geometry().section]
                    .physical.as_mut().ok_or_else(|| {
                        publication_input_error("source content physical custody has already retired")
                    })?;
                let slot = occurrence.receiver_slot();
                if physical.prepared_receiver.as_ref().is_some_and(|(original_slot, original)| {
                    *original_slot != slot || !original.same_handle(step)
                }) {
                    return Err(publication_input_error(
                        "CURRENT capture changed its original prepared Step receiver",
                    ));
                }
                physical.prepared_receiver = Some((slot, step.clone()));
            }
            (SemanticSourceContentKind::Verify, Some(receiver)) => {
                session.checked_prepared_content_witness(step, receiver)?;
                if occurrence.geometry().section_index == 0 {
                    let content = &session.steps[&step.token].content[receiver.index];
                    if !matches!(&content.seals, TensorContentSeals::Model(model)
                        if model.prepared.as_ref().is_some_and(|original| original.bank == section.1))
                    {
                        return Err(publication_input_error(
                            "CURRENT first verification requires its original recorded bank model witness",
                        ));
                    }
                }
            }
            _ => return Err(publication_input_error(
                "CURRENT content requires its original Step capture or witness verification receiver",
            )),
        }
        self.claim(occurrence, receiver, tensors)
    }

    pub(super) fn require_complete(&self) -> Result<(), SemanticTransitionError> {
        let state = self.state.lock().map_err(|_| {
            publication_input_error("source content occurrence custody is poisoned")
        })?;
        if state
            .iter()
            .zip(self.section_lengths)
            .any(|(state, length)| {
                state.cursor != length || state.pending.is_some() || state.native_pending.is_some()
            })
        {
            return Err(publication_input_error(
                "source content still has an unresolved original frozen occurrence suffix",
            ));
        }
        Ok(())
    }

    pub(super) fn require_original_recipe(
        &self,
        recipe: &Arc<prepared_source_content::PreparedSourceContentOwner>,
    ) -> Result<(), SemanticTransitionError> {
        if self
            .geometry
            .upgrade()
            .is_none_or(|original| !Arc::ptr_eq(&original, &recipe.geometry))
        {
            return Err(publication_input_error(
                "source content occurrence belongs to another original recipe",
            ));
        }
        Ok(())
    }

    pub(super) fn receiver_content_identity(
        self: &Arc<Self>,
        occurrence: &SemanticSourceContentOccurrence,
    ) -> Result<(Arc<()>, u64, usize), SemanticTransitionError> {
        if !Arc::ptr_eq(self, &occurrence.frozen) {
            return Err(publication_input_error(
                "source content occurrence belongs to another frozen original roster",
            ));
        }
        let state = self.state.lock().map_err(|_| {
            publication_input_error("source content occurrence custody is poisoned")
        })?;
        let state = &state[occurrence.geometry().section];
        let original = state
            .physical
            .as_ref()
            .and_then(|physical| physical.witnesses.get(&occurrence.receiver_slot()))
            .ok_or_else(|| {
                publication_input_error("source content receiver has no original native witness")
            })?;
        Ok((Arc::clone(&original.issuer), original.token, original.index))
    }

    pub(super) fn claim(
        self: &Arc<Self>,
        occurrence: &SemanticSourceContentOccurrence,
        receiver: Option<&SemanticTensorContentWitness>,
        tensors: &[PreparedSemanticTensor],
    ) -> Result<SourceContentClaim, SemanticTransitionError> {
        if !Arc::ptr_eq(self, &occurrence.frozen) {
            return Err(publication_input_error(
                "source content occurrence belongs to another frozen original roster",
            ));
        }
        let event = &self.events[occurrence.index];
        if !same_source_tensor_geometry(&event.tensors, tensors) {
            return Err(publication_input_error(
                "source content occurrence changed its frozen original typed tensor roster",
            ));
        }
        let mut state = self.state.lock().map_err(|_| {
            publication_input_error("source content occurrence custody is poisoned")
        })?;
        let state = &mut state[event.section];
        if state.cursor != event.section_index
            || state.pending.is_some()
            || state.native_pending.is_some()
        {
            return Err(publication_input_error(
                "source content occurrence must enter once in its original frozen order",
            ));
        }
        let physical = state.physical.as_mut().ok_or_else(|| {
            publication_input_error("source content physical custody has already retired")
        })?;
        let geometry = self
            .geometry
            .upgrade()
            .ok_or(SemanticTransitionError::ObservationMismatch)?;
        for (&slot, declared) in &self.declared_allocations[event.section] {
            if let Some(actual) = if event.prepared_section.is_none() {
                geometry.bound_allocation(declared)?
            } else {
                // The CURRENT prepared recorder authenticates its actual child
                // producer. No historical recipe owns that child's allocations.
                None
            } {
                if physical
                    .allocations
                    .get(&slot)
                    .and_then(Option::as_ref)
                    .is_some_and(|previous| !previous.same_allocation(&actual))
                {
                    return Err(publication_input_error(
                        "source content changed its original bound allocation",
                    ));
                }
                physical.allocations.insert(slot, Some(actual));
            }
        }
        if event.prepared_section.is_some() && event.section_index == 0 {
            let receiver = receiver.ok_or(SemanticTransitionError::ObservationMismatch)?;
            physical.witnesses.insert(
                event.receiver_slot,
                SourceWitnessIdentity::from_witness(receiver),
            );
        }
        if if event.prepared_section.is_some() && event.kind == SemanticSourceContentKind::Capture {
            receiver.is_some()
                || physical
                    .prepared_receiver
                    .as_ref()
                    .is_none_or(|(slot, _)| *slot != event.receiver_slot)
        } else if event.kind == SemanticSourceContentKind::BindModel {
            occurrence.index != 0 || receiver.is_some()
        } else {
            receiver.is_none_or(|receiver| {
                physical
                    .witnesses
                    .get(&event.receiver_slot)
                    .is_none_or(|original| !original.matches(receiver))
            })
        } {
            return Err(publication_input_error(
                "source content receiver is not its exact original native witness",
            ));
        }
        // Preserve the entire genuine producer roster before any fallible
        // backing query or later command construction. A rejected handoff is
        // still owned until the authentic terminal cleanup boundary.
        let tensor_roster = physical.tensors.len();
        physical.tensors.push(tensors.to_vec());
        for (descriptor, tensor) in event.tensors.iter().zip(tensors) {
            let span = event
                .allocations
                .iter()
                .find_map(|&(slot, span)| (slot == descriptor.allocation_slot).then_some(span))
                .expect("validated original allocation descriptor");
            if let Some(native) = &tensor.native_allocation {
                if !self.declared_allocations[event.section]
                    .contains_key(&descriptor.allocation_slot)
                {
                    return Err(publication_input_error(
                        "native content requires its original pre-factory allocation port",
                    ));
                }
                if native.allocation_bytes() != span {
                    return Err(publication_input_error(
                        "source content tensor changed its original native backing allocation or offset",
                    ));
                }
                for (&slot, original) in &physical.allocations {
                    if let Some(original) = original {
                        if native.same_allocation(original) != (slot == descriptor.allocation_slot)
                        {
                            return Err(publication_input_error(
                                "source content tensor changed its frozen native allocation alias partition",
                            ));
                        }
                    }
                }
                physical
                    .allocations
                    .insert(descriptor.allocation_slot, Some(native.clone()));
            } else if physical
                .allocations
                .get(&descriptor.allocation_slot)
                .is_some_and(Option::is_some)
            {
                return Err(publication_input_error(
                    "source content tensor lost its original native allocation provenance",
                ));
            }
            let base = tensor
                .data
                .checked_sub(descriptor.byte_offset)
                .ok_or_else(|| {
                    publication_input_error(
                        "source content byte offset exceeds its actual original pointer",
                    )
                })?;
            let backing = if span == 0 {
                (base, 0)
            } else {
                // The driver block may contain several genuine producer
                // allocations. It proves containment, not the size or identity
                // of an original native allocation or Torch storage object.
                let bytes = usize::try_from(span)
                    .map_err(|_| SemanticTransitionError::GenerationExhausted)?;
                cuda_backing_allocation(base, bytes)?;
                (base, span)
            };
            for (&slot, original) in &physical.backing_allocations {
                if span != 0 && *original == backing && slot != descriptor.allocation_slot
                    || slot == descriptor.allocation_slot && *original != backing
                {
                    return Err(publication_input_error(
                        "source content tensor changed its complete backing allocation alias partition",
                    ));
                }
            }
            physical
                .backing_allocations
                .insert(descriptor.allocation_slot, backing);
            if physical
                .storage_allocations
                .get(&descriptor.storage_slot)
                .is_some_and(|original| *original != backing)
            {
                return Err(publication_input_error(
                    "source content storage slot changed its original physical backing",
                ));
            }
            physical
                .storage_allocations
                .insert(descriptor.storage_slot, backing);
        }
        let identity = Arc::new(());
        state.pending = Some((occurrence.index, Arc::clone(&identity), tensor_roster));
        Ok(SourceContentClaim {
            frozen: Arc::clone(self),
            index: occurrence.index,
            identity,
        })
    }

    pub(super) fn retire_physical_custody(&self) {
        let retired = match self.state.lock() {
            Ok(mut state) => Some(std::array::from_fn::<_, 4, _>(|section| {
                state[section].physical.take()
            })),
            Err(_) => None,
        };
        drop(retired);
    }
}

impl SourceNativeContentClaim {
    pub(super) fn native_work_ceiling(&self) -> [u64; 9] {
        self.frozen.native_guards[&self.kind].1
    }

    pub(super) fn same_claim(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.frozen, &other.frozen)
            && self.kind == other.kind
            && Arc::ptr_eq(&self.identity, &other.identity)
    }

    pub(super) fn is_retirement(&self) -> bool {
        self.kind == SourceNativeContentGuard::Retirement
    }

    pub(super) fn complete(
        &self,
        original: &native_work_bound::OriginalContentBatch,
    ) -> Result<(), SemanticTransitionError> {
        if !original.verify
            || original.cursor != original.commands.len()
            || original.commands.iter().any(|command| !command.completed())
            || original
                .source_native_guard
                .as_ref()
                .is_none_or(|claim| !self.same_claim(claim))
        {
            return Err(publication_input_error(
                "SOURCE native consumer still has unresolved original commands",
            ));
        }
        let mut state = self.frozen.state.lock().map_err(|_| {
            publication_input_error("original SOURCE native consumer custody is poisoned")
        })?;
        let state = &mut state[0];
        if state
            .native_pending
            .as_ref()
            .is_none_or(|(kind, identity)| {
                *kind != self.kind || !Arc::ptr_eq(identity, &self.identity)
            })
        {
            return Err(publication_input_error(
                "SOURCE native consumer lost its exact retained original claim",
            ));
        }
        state.native_pending = None;
        state.native_completed.insert(self.kind);
        Ok(())
    }
}

fn same_source_tensor_geometry(
    descriptors: &[SemanticSourceTensorDescriptor],
    tensors: &[PreparedSemanticTensor],
) -> bool {
    descriptors.len() == tensors.len()
        && descriptors.iter().zip(tensors).all(|(descriptor, tensor)| {
            descriptor.layout == tensor.layout
                && descriptor.logical_begin == tensor.logical_begin
                && descriptor.logical_end == tensor.logical_end
        })
}

impl SourceContentClaim {
    pub(super) fn same_claim(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.frozen, &other.frozen)
            && self.index == other.index
            && Arc::ptr_eq(&self.identity, &other.identity)
    }

    pub(super) fn native_work_ceiling(&self) -> [u64; 9] {
        self.frozen.events[self.index].ceiling
    }

    fn require_pending<'a>(
        &self,
        state: &'a mut SourceOccurrenceState,
    ) -> Result<&'a mut SourcePhysicalCustody, SemanticTransitionError> {
        if state.cursor != self.frozen.events[self.index].section_index
            || state.pending.as_ref().is_none_or(|(index, identity, _)| {
                *index != self.index || !Arc::ptr_eq(identity, &self.identity)
            })
        {
            return Err(publication_input_error(
                "source content command lost its exact original pending occurrence claim",
            ));
        }
        state.physical.as_mut().ok_or_else(|| {
            publication_input_error("source content physical custody has already retired")
        })
    }

    pub(super) fn bind_output(
        &self,
        witness: &SemanticTensorContentWitness,
        content: &TensorContentBuffers,
    ) -> Result<(), SemanticTransitionError> {
        let event = &self.frozen.events[self.index];
        let slot = event.output_slot.ok_or_else(|| {
            publication_input_error("source content verification cannot create an output witness")
        })?;
        if (event.prepared_section.is_none()
            && !self
                .frozen
                .geometry
                .upgrade()
                .ok_or(SemanticTransitionError::ObservationMismatch)?
                .matches_reader(&witness.issuer, witness.reader_token)?)
            || !match (&content.seals, event.kind) {
                (TensorContentSeals::Model(_), SemanticSourceContentKind::BindModel) => true,
                (TensorContentSeals::Captured(_), SemanticSourceContentKind::Capture) => true,
                _ => false,
            }
            || !same_source_tensor_geometry(&event.tensors, &content.tensors)
        {
            return Err(publication_input_error(
                "source content capture output belongs to another original reader",
            ));
        }
        let mut state = self.frozen.state.lock().map_err(|_| {
            publication_input_error("source content occurrence custody is poisoned")
        })?;
        let state = &mut state[event.section];
        let tensor_roster = state
            .pending
            .as_ref()
            .map(|(_, _, roster)| *roster)
            .ok_or_else(|| {
                publication_input_error(
                    "source content capture has no original pending tensor roster",
                )
            })?;
        let physical = self.require_pending(state)?;
        if event.prepared_section.is_some()
            && !physical.witnesses.values().any(|original| {
                Arc::ptr_eq(&original.issuer, &witness.issuer)
                    && original.token == witness.reader_token
            })
        {
            return Err(publication_input_error(
                "CURRENT capture changed its original prepared child reader",
            ));
        }
        let original_tensors = &physical.tensors[tensor_roster];
        if original_tensors.len() != content.tensors.len()
            || original_tensors
                .iter()
                .zip(&content.tensors)
                .any(|(original, actual)| !same_tensor_content_owner(original, actual))
        {
            return Err(publication_input_error(
                "source content capture output changed its exact original native tensor owners",
            ));
        }
        if let Some(original) = physical.witnesses.get(&slot) {
            if !original.matches(witness) {
                return Err(publication_input_error(
                    "source content capture output changed its original native witness",
                ));
            }
            return Ok(());
        }
        if physical
            .witnesses
            .values()
            .any(|original| original.matches(witness))
        {
            return Err(publication_input_error(
                "source content capture must issue a new original output witness",
            ));
        }
        physical
            .witnesses
            .insert(slot, SourceWitnessIdentity::from_witness(witness));
        Ok(())
    }

    // Called only by the canonical batch completion boundary, after its real
    // command waits and original consumer ordering. No public completion flag
    // or new command owner can discharge this retained claim.
    pub(super) fn complete(
        &self,
        original: &native_work_bound::OriginalContentBatch,
    ) -> Result<(), SemanticTransitionError> {
        if original
            .source_occurrence
            .as_ref()
            .is_none_or(|claim| !self.same_claim(claim))
            || original.cursor != original.commands.len()
            || original.commands.iter().any(|command| !command.completed())
            || original.verify
                != (self.frozen.events[self.index].kind != SemanticSourceContentKind::Capture)
        {
            return Err(publication_input_error(
                "source content occurrence still has unresolved original native commands",
            ));
        }
        let mut state = self.frozen.state.lock().map_err(|_| {
            publication_input_error("source content occurrence custody is poisoned")
        })?;
        let state = &mut state[self.frozen.events[self.index].section];
        self.finish_pending(state)
    }

    pub(super) fn complete_prepared(
        &self,
        session: &SemanticTransitionSession,
        step: &SemanticPreparedStep,
        witness: &SemanticTensorContentWitness,
        verify: bool,
    ) -> Result<(), SemanticTransitionError> {
        let event = &self.frozen.events[self.index];
        let section = event
            .prepared_section
            .ok_or(SemanticTransitionError::ObservationMismatch)?;
        self.frozen
            .geometry
            .upgrade()
            .ok_or(SemanticTransitionError::ObservationMismatch)?
            .require_current_section(session, step, section)?;
        session.checked_prepared_content_witness(step, witness)?;
        if verify != (event.kind == SemanticSourceContentKind::Verify)
            || !same_source_tensor_geometry(
                &event.tensors,
                &session.steps[&step.token].content[witness.index].tensors,
            )
        {
            return Err(publication_input_error(
                "CURRENT recorded content changed its original typed occurrence",
            ));
        }
        // This cursor proves successful binding by the original prepared
        // recorder, not GPU execution or cold expenditure. The same recorded
        // witness and native tally remain responsible for actual later replay.
        let mut state = self.frozen.state.lock().map_err(|_| {
            publication_input_error("original prepared content custody is poisoned")
        })?;
        self.finish_pending(&mut state[event.section])
    }

    fn finish_pending(
        &self,
        state: &mut SourceOccurrenceState,
    ) -> Result<(), SemanticTransitionError> {
        let physical = self.require_pending(state)?;
        if self.frozen.events[self.index]
            .output_slot
            .is_some_and(|slot| !physical.witnesses.contains_key(&slot))
        {
            return Err(publication_input_error(
                "source content capture has not bound its original output witness",
            ));
        }
        state.cursor = state
            .cursor
            .checked_add(1)
            .ok_or(SemanticTransitionError::GenerationExhausted)?;
        state.pending = None;
        Ok(())
    }
}
