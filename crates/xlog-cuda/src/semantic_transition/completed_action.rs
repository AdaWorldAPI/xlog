//! The original native action witness codec. Decoding does not grant execution authority.

use super::*;

pub(super) struct OriginalCompletedAction {
    pub(super) predecessor: SemanticPublishedIdentity,
    pub(super) successor: SemanticPublishedIdentity,
    pub(super) bank: u64,
    pub(super) state: DeviceState,
    pub(super) receipts: [SemanticTransitionReceipt; COMPONENT_COUNT],
}

pub(super) fn decode(bytes: &[u8]) -> Result<OriginalCompletedAction, SemanticTransitionError> {
    let mut reader = SemanticMaterialReader::new(bytes);
    let magic = b"XLOG-COMPLETED-ACTION-WITNESSES\0";
    if reader
        .take(magic.len())
        .map_err(SemanticTransitionError::Semantic)?
        != magic
        || reader.u64().map_err(SemanticTransitionError::Semantic)? != 2
    {
        return Err(publication_input_error(
            "completed action witness has another native envelope",
        ));
    }
    let predecessor = decode_publication_identity(&mut reader)?;
    let successor = decode_publication_identity(&mut reader)?;
    let bank = reader.u64().map_err(SemanticTransitionError::Semantic)?;
    if bank > 1
        || reader.u64().map_err(SemanticTransitionError::Semantic)?
            != size_of::<DeviceState>() as u64
    {
        return Err(publication_input_error(
            "completed action witness has another native state extent",
        ));
    }
    let state_bytes = reader
        .take(size_of::<DeviceState>())
        .map_err(SemanticTransitionError::Semantic)?;
    // SAFETY: the complete fixed native state consists only of initialized integer ABI records.
    let state = unsafe { state_bytes.as_ptr().cast::<DeviceState>().read_unaligned() };
    if reader.u64().map_err(SemanticTransitionError::Semantic)? != COMPONENT_COUNT as u64 {
        return Err(publication_input_error(
            "completed action witness lost its full native receipt roster",
        ));
    }
    let bytes = reader
        .take(COMPONENT_COUNT * size_of::<SemanticTransitionReceipt>())
        .map_err(SemanticTransitionError::Semantic)?;
    let receipts = std::array::from_fn(|index| {
        // SAFETY: the checked exact roster bounds each padding-free scalar receipt.
        unsafe {
            bytes[index * size_of::<SemanticTransitionReceipt>()..]
                .as_ptr()
                .cast::<SemanticTransitionReceipt>()
                .read_unaligned()
        }
    });
    reader.finish().map_err(SemanticTransitionError::Semantic)?;
    Ok(OriginalCompletedAction {
        predecessor,
        successor,
        bank,
        state,
        receipts,
    })
}
