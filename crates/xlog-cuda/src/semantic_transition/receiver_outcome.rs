//! Authenticate a receiver's original terminal outcome independently of replay rows.

use super::*;

const OUTCOME_DOMAIN: &[u8] = b"xlog.receiver.outcome.v1\0";
const IDENTITY_FIELDS: usize = 6;
const HEADER_BYTES: usize = OUTCOME_DOMAIN.len() + IDENTITY_FIELDS * 32 + 2 * size_of::<u64>();
const MAC_BYTES: usize = 32;
const MAX_REFUSAL_BYTES: usize = 4096;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum ReceiverDisposition {
    Completed,
    Refused,
}

pub(super) struct AuthenticatedReceiverOutcome<'a> {
    pub identity: Identity256,
    pub checkpoint_identity: Identity256,
    pub disposition: ReceiverDisposition,
    pub content: &'a [u8],
}

fn invalid_outcome() -> SemanticTransitionError {
    publication_input_error("receiver outcome differs from its original authenticated delivery")
}

/// Validate the exact signed terminal record without allocating or reconstructing it.
///
/// The caller retains these original bytes through the one acknowledgement and
/// archives them with its delivery receipt. A started or unknown request has no
/// terminal record and cannot be represented by either disposition here.
pub(super) fn verify_receiver_outcome<'a>(
    bytes: &'a [u8],
    stable_intent: Identity256,
    recipient: Identity256,
    effect: &[u8],
    payload: &[u8],
    delivery_receipt: &[u8],
    key: &[u8; 32],
    max_outcome_bytes: usize,
) -> Result<AuthenticatedReceiverOutcome<'a>, SemanticTransitionError> {
    if bytes.len() > max_outcome_bytes
        || bytes.len() < HEADER_BYTES + MAC_BYTES
        || !bytes.starts_with(OUTCOME_DOMAIN)
    {
        return Err(invalid_outcome());
    }
    let mac_offset = bytes.len() - MAC_BYTES;
    if !super::late_replay::receiver_mac_matches(&bytes[..mac_offset], &bytes[mac_offset..], key) {
        return Err(invalid_outcome());
    }
    let identities = &bytes[OUTCOME_DOMAIN.len()..OUTCOME_DOMAIN.len() + IDENTITY_FIELDS * 32];
    for (actual, expected) in identities.chunks_exact(32).zip([
        stable_intent,
        recipient,
        Identity256::from_bytes(Sha256::digest(effect).into()),
        Identity256::from_bytes(Sha256::digest(payload).into()),
        Identity256::from_bytes(Sha256::digest(delivery_receipt).into()),
    ]) {
        if actual != expected.as_bytes() {
            return Err(invalid_outcome());
        }
    }
    let checkpoint_identity = Identity256::from_bytes(
        identities[5 * 32..6 * 32]
            .try_into()
            .map_err(|_| invalid_outcome())?,
    );
    if checkpoint_identity.as_bytes().iter().all(|byte| *byte == 0) {
        return Err(invalid_outcome());
    }
    let status_offset = OUTCOME_DOMAIN.len() + IDENTITY_FIELDS * 32;
    let status = u64::from_le_bytes(
        bytes[status_offset..status_offset + 8]
            .try_into()
            .map_err(|_| invalid_outcome())?,
    );
    let content_bytes = u64::from_le_bytes(
        bytes[status_offset + 8..HEADER_BYTES]
            .try_into()
            .map_err(|_| invalid_outcome())?,
    );
    let content = &bytes[HEADER_BYTES..mac_offset];
    if usize::try_from(content_bytes).ok() != Some(content.len()) || content.is_empty() {
        return Err(invalid_outcome());
    }
    let disposition = match status {
        1 => ReceiverDisposition::Completed,
        2 if content.len() <= MAX_REFUSAL_BYTES && std::str::from_utf8(content).is_ok() => {
            ReceiverDisposition::Refused
        }
        _ => return Err(invalid_outcome()),
    };
    Ok(AuthenticatedReceiverOutcome {
        identity: Identity256::from_bytes(Sha256::digest(bytes).into()),
        checkpoint_identity,
        disposition,
        content,
    })
}
