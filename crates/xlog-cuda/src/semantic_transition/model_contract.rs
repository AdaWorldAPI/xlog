//! Authenticated host model contracts of acquired publication readers.

use super::*;

impl SemanticTransitionSession {
    /// Return the complete structural contract of this actual acquired parent.
    ///
    /// The finalizer projects structural schema, generation and origin identity
    /// from the acquired header. This
    /// performs no device read and grants neither numerical nor import access.
    pub fn published_model_contract_metadata(
        &self,
        lease: &SemanticPublishedLease,
    ) -> Result<(Vec<u8>, SemanticModelContractLayout), SemanticTransitionError> {
        self.checked_reader(lease)?;
        let header = lease.model_context_header()?;
        let storage = self
            .publication
            .as_ref()
            .ok_or(SemanticTransitionError::NotBound)?;
        let range = *lease
            .directory
            .iter()
            .find(|range| (range.role, range.index) == (SemanticStateRole::ModelContract as u64, 0))
            .ok_or(SemanticTransitionError::ObservationMismatch)?;
        let allocation = storage
            .allocations
            .get(range.storage_slot as usize)
            .ok_or(SemanticTransitionError::ObservationMismatch)?;
        let layout = storage.contract_value.model_contract_layout;
        let original = &storage.model_contract_record;
        if storage.instance != header.instance
            || storage.contract_value.abi != 1
            || storage.contract_value.model_generation == 0
            || header.model_generation < storage.contract_value.model_generation
            || header.model_generation > u64::from(u32::MAX)
            || header.range_count != lease.directory.len() as u64
            || range.generation != 1
            || range.logical_begin != 0
            || range.logical_end != 0
            || range.length_bytes != original.len() as u64
            || range
                .offset_bytes
                .checked_add(range.length_bytes)
                .is_none_or(|end| end > allocation.len() as u64)
            || storage.model_memory.digest(&storage.layouts)? != header.model_geometry_digest
        {
            return Err(SemanticTransitionError::ObservationMismatch);
        }
        layout.validate(range.length_bytes)?;
        if layout.schema_bytes > u64::MAX / 8 {
            return Err(SemanticTransitionError::ObservationMismatch);
        }

        let mut bytes = original.to_vec();
        let schema_begin = layout.schema_begin as usize;
        let schema_end = schema_begin + layout.schema_bytes as usize;
        let schema_digest: [u8; 32] = Sha256::digest(&bytes[schema_begin..schema_end]).into();
        let generation = header.model_generation.to_le_bytes();
        let mut identity = Sha256::new();
        identity.update(b"xlog.semantic.model-identity.v2\0");
        identity.update(schema_digest);
        identity.update(generation);
        identity.update(header.original_instance.as_bytes());
        let identity: [u8; 32] = identity.finalize().into();
        for (offset, value) in [
            (layout.schema_digest_offset, schema_digest.as_slice()),
            (layout.generation_offset, generation.as_slice()),
            (layout.identity_offset, identity.as_slice()),
        ] {
            let begin = offset as usize;
            bytes[begin..begin + value.len()].copy_from_slice(value);
        }
        let record = PublicationMaterialRange {
            range,
            capacity: allocation.len(),
            bytes,
        };
        Ok((record.bytes, layout))
    }
}
