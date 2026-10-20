//! The original delivery and terminal receiver event share one acknowledgement.

use super::*;

#[repr(C)]
#[derive(Clone, Copy, Default)]
pub(super) struct AcknowledgementEntry {
    pub stable_identity: Identity256,
    pub receipt_digest: Identity256,
    pub receipt_len: u64,
    pub receiver_outcome_digest: Identity256,
    pub receiver_outcome_len: u64,
    pub previous_chain: Identity256,
    pub chain: Identity256,
}

// SAFETY: this padding-free C record contains initialized integral words only.
unsafe impl DeviceRepr for AcknowledgementEntry {}
const _: () = assert!(size_of::<AcknowledgementEntry>() == 176);

pub(super) struct ArchivedAcknowledgement {
    pub stable_identity: Identity256,
    pub delivery_receipt: Vec<u8>,
    pub receiver_outcome: Vec<u8>,
}

fn header(
    bytes: &[u8],
    capacity: usize,
) -> Result<AcknowledgementQueueHeader, SemanticTransitionError> {
    if bytes.len() < size_of::<AcknowledgementQueueHeader>() {
        return Err(publication_input_error(
            "acknowledgement queue has no native header",
        ));
    }
    // SAFETY: the complete padding-free header extent was checked above.
    let header = unsafe {
        bytes
            .as_ptr()
            .cast::<AcknowledgementQueueHeader>()
            .read_unaligned()
    };
    let declared = usize::try_from(header.capacity)
        .ok()
        .and_then(|count| count.checked_mul(size_of::<AcknowledgementEntry>()))
        .and_then(|entries| {
            entries.checked_add(usize::try_from(header.payload_capacity_bytes).ok()?)
        })
        .and_then(|body| body.checked_add(size_of::<AcknowledgementQueueHeader>()));
    if header.abi != 1
        || header.count > header.capacity
        || declared != Some(capacity)
        || bytes.len() > capacity
        || header.payload_used_bytes
            != (bytes.len() - size_of::<AcknowledgementQueueHeader>()) as u64
        || header.count > header.payload_used_bytes / size_of::<AcknowledgementEntry>() as u64
    {
        return Err(publication_input_error(
            "acknowledgement queue extent is inconsistent",
        ));
    }
    Ok(header)
}

fn entry(
    bytes: &[u8],
    cursor: usize,
) -> Result<(AcknowledgementEntry, usize, usize, usize), SemanticTransitionError> {
    let payload = cursor
        .checked_add(size_of::<AcknowledgementEntry>())
        .filter(|end| *end <= bytes.len())
        .ok_or_else(|| publication_input_error("acknowledgement entry is incomplete"))?;
    // SAFETY: the complete padding-free entry extent was checked above.
    let entry = unsafe {
        bytes[cursor..]
            .as_ptr()
            .cast::<AcknowledgementEntry>()
            .read_unaligned()
    };
    let outcome = payload
        .checked_add(
            usize::try_from(entry.receipt_len)
                .map_err(|_| SemanticTransitionError::GenerationExhausted)?,
        )
        .filter(|end| *end <= bytes.len())
        .ok_or_else(|| publication_input_error("acknowledgement delivery receipt is incomplete"))?;
    let end = outcome
        .checked_add(
            usize::try_from(entry.receiver_outcome_len)
                .map_err(|_| SemanticTransitionError::GenerationExhausted)?,
        )
        .filter(|end| *end <= bytes.len())
        .ok_or_else(|| publication_input_error("acknowledgement receiver outcome is incomplete"))?;
    Ok((entry, payload, outcome, end))
}

/// Check the original finite destination before hashing or parsing new bytes.
/// An exact retry replaces its own retained extent for this capacity calculation;
/// authentication and exact byte equality are still required by the ACK owner.
pub(super) fn admit_payload(
    bytes: &[u8],
    capacity: usize,
    stable: Identity256,
    delivery_bytes: usize,
    outcome_bytes: usize,
) -> Result<usize, SemanticTransitionError> {
    let header = header(bytes, capacity)?;
    let mut cursor = size_of::<AcknowledgementQueueHeader>();
    let mut used = 0usize;
    let mut previous = None;
    for _ in 0..header.count {
        let (item, payload, _, end) = entry(bytes, cursor)?;
        used = used
            .checked_add(end - payload)
            .ok_or(SemanticTransitionError::GenerationExhausted)?;
        if item.stable_identity == stable && previous.replace(end - payload).is_some() {
            return Err(publication_input_error(
                "acknowledgement repeats one stable intent",
            ));
        }
        cursor = end;
    }
    let payload_capacity = usize::try_from(header.payload_capacity_bytes)
        .map_err(|_| SemanticTransitionError::GenerationExhausted)?;
    if cursor != bytes.len()
        || used > payload_capacity
        || (previous.is_none() && header.count == header.capacity)
    {
        return Err(publication_input_error(
            "acknowledgement archive exceeds its original capacity",
        ));
    }
    let available = used
        .checked_sub(previous.unwrap_or(0))
        .and_then(|used| used.checked_add(delivery_bytes))
        .and_then(|used| payload_capacity.checked_sub(used))
        .ok_or_else(|| {
            publication_input_error("acknowledgement delivery exceeds its original capacity")
        })?;
    if outcome_bytes > available {
        return Err(publication_input_error(
            "receiver outcome exceeds its original acknowledgement capacity",
        ));
    }
    Ok(available)
}

pub(super) fn allocation_bound(
    bytes: &[u8],
    capacity: usize,
) -> Result<u64, SemanticTransitionError> {
    let header = header(bytes, capacity)?;
    let width = size_of::<ArchivedAcknowledgement>()
        .checked_add(OwnedPublicationMaterial::tree_node_bound(
            size_of::<[u8; 32]>(),
            0,
        )?)
        .ok_or(SemanticTransitionError::GenerationExhausted)? as u64;
    let payload_bytes = header
        .payload_used_bytes
        .checked_sub(
            header
                .count
                .checked_mul(size_of::<AcknowledgementEntry>() as u64)
                .ok_or(SemanticTransitionError::GenerationExhausted)?,
        )
        .ok_or(SemanticTransitionError::GenerationExhausted)?;
    header
        .count
        .checked_mul(width)
        .and_then(|bytes| bytes.checked_add(payload_bytes))
        .ok_or(SemanticTransitionError::GenerationExhausted)
}

pub(super) fn acknowledgement_records(
    bytes: &[u8],
    capacity: usize,
) -> Result<Vec<ArchivedAcknowledgement>, SemanticTransitionError> {
    let header = header(bytes, capacity)?;
    let mut cursor = size_of::<AcknowledgementQueueHeader>();
    let mut used = 0usize;
    let mut chain = Identity256::default();
    let mut seen = BTreeSet::new();
    let mut records = Vec::with_capacity(header.count as usize);
    for ordinal in 0..header.count {
        let (item, payload, outcome, end) = entry(bytes, cursor)?;
        let delivery = &bytes[payload..outcome];
        let terminal = &bytes[outcome..end];
        used = used
            .checked_add(end - payload)
            .ok_or(SemanticTransitionError::GenerationExhausted)?;
        let previous = chain;
        chain = publication_fold_digest(chain, 32, ordinal, item.stable_identity);
        chain = publication_fold_digest(chain, 32, ordinal, item.receipt_digest);
        chain = publication_fold_digest(chain, 32, ordinal, item.receiver_outcome_digest);
        let outcome_digest = if terminal.is_empty() {
            Identity256::default()
        } else {
            Identity256::from_bytes(Sha256::digest(terminal).into())
        };
        if item.stable_identity == Identity256::default()
            || !seen.insert(*item.stable_identity.as_bytes())
            || item.previous_chain != previous
            || item.chain != chain
            || item.receipt_digest != Identity256::from_bytes(Sha256::digest(delivery).into())
            || item.receiver_outcome_digest != outcome_digest
        {
            return Err(publication_input_error(
                "acknowledgement differs from its native paired chain",
            ));
        }
        records.push(ArchivedAcknowledgement {
            stable_identity: item.stable_identity,
            delivery_receipt: delivery.to_vec(),
            receiver_outcome: terminal.to_vec(),
        });
        cursor = end;
    }
    if cursor != bytes.len()
        || chain != header.chain_head
        || used as u64 > header.payload_capacity_bytes
    {
        return Err(publication_input_error(
            "acknowledgement chain does not cover its original payloads",
        ));
    }
    Ok(records)
}
