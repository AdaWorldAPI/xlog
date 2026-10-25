//! Content lineage for a completed prefill held before its task is published.

use super::*;

#[derive(Clone, Copy)]
struct PrefillTensorGeometry {
    layout: SemanticTensorLayout,
    logical_begin: u64,
    logical_end: u64,
    physical_bytes: usize,
}

impl PrefillTensorGeometry {
    fn from_tensor(tensor: &PreparedSemanticTensor) -> Result<Self, SemanticTransitionError> {
        let physical_bytes = tensor.source.as_ref().map_or(0, DeviceMemoryView::len);
        tensor_content_range(
            &tensor.layout,
            tensor.logical_begin,
            tensor.logical_end,
            physical_bytes,
        )?;
        Ok(Self {
            layout: tensor.layout,
            logical_begin: tensor.logical_begin,
            logical_end: tensor.logical_end,
            physical_bytes,
        })
    }

    fn encode(&self, bytes: &mut Vec<u8>) {
        for word in [
            self.layout.role,
            self.layout.index,
            self.layout.element_bytes,
            self.layout.scalar_type,
            self.layout.rank,
            self.layout.logical_axis,
        ] {
            material_u64(bytes, word);
        }
        for word in self
            .layout
            .dimensions
            .iter()
            .chain(&self.layout.strides_bytes)
        {
            material_u64(bytes, *word);
        }
        material_u64(bytes, self.logical_begin);
        material_u64(bytes, self.logical_end);
        material_u64(bytes, self.physical_bytes as u64);
    }

    fn decode(reader: &mut SemanticMaterialReader<'_>) -> Result<Self, SemanticTransitionError> {
        let mut word = || reader.u64().map_err(SemanticTransitionError::Semantic);
        let layout = SemanticTensorLayout {
            role: word()?,
            index: word()?,
            element_bytes: word()?,
            scalar_type: word()?,
            rank: word()?,
            logical_axis: word()?,
            dimensions: [word()?, word()?, word()?, word()?],
            strides_bytes: [word()?, word()?, word()?, word()?],
        };
        let logical_begin = word()?;
        let logical_end = word()?;
        let physical_bytes =
            usize::try_from(word()?).map_err(|_| SemanticTransitionError::GenerationExhausted)?;
        tensor_content_range(&layout, logical_begin, logical_end, physical_bytes)?;
        Ok(Self {
            layout,
            logical_begin,
            logical_end,
            physical_bytes,
        })
    }
}

pub(super) struct OriginalDormantPrefillCheckpoint {
    continuation: Arc<ModelContextContinuationOwner>,
    storage: Arc<PublicationStorage>,
    scope: Arc<()>,
    source_parent: SemanticPublishedIdentity,
    content: OriginalInitialPrefillContent,
    geometry: [Vec<PrefillTensorGeometry>; 2],
    loaded: Option<SemanticReplayLoadedMaterial>,
    bindings: Option<(Identity256, Identity256)>,
    allowance: Arc<Mutex<native_work_bound::ColdNativeAllowance>>,
    reservation: Arc<SemanticReplayMaterialReservation>,
}

impl SemanticTransitionSession {
    fn completed_restored_prefill_for_checkpoint(
        &self,
        continuation: &SemanticModelContextContinuation,
    ) -> Result<Option<SemanticRestoredModelContextPrefill>, SemanticTransitionError> {
        let original = continuation
            .inner
            .original
            .lock()
            .map_err(|_| SemanticTransitionError::Poisoned)?;
        let stage = original
            .parent
            .as_ref()
            .and_then(|parent| parent.prefill.as_ref())
            .ok_or(SemanticTransitionError::NotBound)?;
        if stage.input.is_some() {
            return Ok(None);
        }
        let restore = original
            .checkpoint_restore
            .clone()
            .ok_or(SemanticTransitionError::NotBound)?;
        drop(original);
        let proof = restore
            .lock()
            .map_err(|_| SemanticTransitionError::Poisoned)?
            .completed_prefill_proof()?;
        self.require_restored_model_context_prefill(continuation, &proof)?;
        Ok(Some(proof))
    }

    pub(super) fn dormant_prefill_checkpoint_allocation_bound(
        &self,
        continuation: &SemanticModelContextContinuation,
    ) -> Result<u64, SemanticTransitionError> {
        continuation.inner.require_dormant_parent(self)?;
        let original = continuation
            .inner
            .original
            .lock()
            .map_err(|_| SemanticTransitionError::Poisoned)?;
        let parent = original
            .parent
            .as_ref()
            .ok_or(SemanticTransitionError::NotBound)?;
        let stage = parent
            .prefill
            .as_ref()
            .ok_or(SemanticTransitionError::NotBound)?;
        let model = stage
            .model_content
            .as_ref()
            .ok_or(SemanticTransitionError::NotBound)?;
        let output = stage
            .output_content
            .as_ref()
            .ok_or(SemanticTransitionError::NotBound)?;
        let count = model
            .tensors
            .len()
            .checked_add(output.tensors.len())
            .ok_or(SemanticTransitionError::GenerationExhausted)?;
        let bytes = initial_prefill_geometry_content_extent(
            &stage.layout.prefill_tokens,
            &stage.layout.prefill_mapping,
        )?
        .checked_mul(3)
        .and_then(|bytes| {
            bytes.checked_add(count.checked_mul(
                2 * size_of::<PrefillTensorGeometry>() + 2 * size_of::<usize>() + 17 * 8,
            )?)
        })
        .and_then(|bytes| {
            bytes.checked_add(
                size_of::<OriginalDormantPrefillCheckpoint>()
                    + 2 * size_of::<usize>()
                    + CheckpointInitialPrefill::MAGIC.len()
                    + 1
                    + 4 * 32
                    + 5 * 8,
            )
        })
        .ok_or(SemanticTransitionError::GenerationExhausted)?;
        u64::try_from(bytes).map_err(|_| SemanticTransitionError::GenerationExhausted)
    }

    pub(super) fn stage_dormant_prefill_checkpoint(
        &self,
        continuation: &SemanticModelContextContinuation,
        source_parent: SemanticPublishedIdentity,
        reservation: Arc<SemanticReplayMaterialReservation>,
        allowance: Arc<Mutex<native_work_bound::ColdNativeAllowance>>,
    ) -> Result<OriginalDormantPrefillCheckpoint, SemanticTransitionError> {
        self.ensure_quiescent()?;
        continuation.inner.require_dormant_parent(self)?;
        self.require_model_context_numerical_contract(continuation)?;
        let restored = self.completed_restored_prefill_for_checkpoint(continuation)?;
        if self
            .graph
            .cold_work_allowance()
            .as_ref()
            .is_none_or(|actual| !Arc::ptr_eq(actual, &allowance))
        {
            return Err(SemanticTransitionError::ObservationMismatch);
        }
        let original = continuation
            .inner
            .original
            .lock()
            .map_err(|_| SemanticTransitionError::Poisoned)?;
        let parent = original
            .parent
            .as_ref()
            .ok_or(SemanticTransitionError::NotBound)?;
        let stage = parent
            .prefill
            .as_ref()
            .ok_or(SemanticTransitionError::NotBound)?;
        if parent.bank.header.abi != 0
            || parent.bank.header.publication_word != 0
            || parent.bank.header.authority_generation != stage.task_epoch
            || parent.bank.header.model_generation != stage.model_generation
            || stage.task_epoch != self.task_epoch
            || self.task_evaluation_identity() != Some(stage.task_identity)
            || stage.layout.prefill_tokens.is_empty()
            || stage.layout.prefill_tokens.len() as u64 != parent.bank.header.prefix_extent
            || (restored.is_none() && (stage.input.is_none() || stage.consumer_stream.is_none()))
        {
            return Err(publication_input_error(
                "dormant prefill is not its original completed input/output boundary",
            ));
        }
        let model = stage
            .model_content
            .as_ref()
            .ok_or(SemanticTransitionError::NotBound)?;
        let output = stage
            .output_content
            .as_ref()
            .ok_or(SemanticTransitionError::NotBound)?;
        let record = InitialPrefillRecord {
            predecessor: source_parent,
            semantic_root: parent.bank.header.semantic_digest,
            task_identity: stage.task_identity,
            task_epoch: stage.task_epoch,
            task_content: self
                .task
                .as_ref()
                .ok_or(SemanticTransitionError::NotBound)?
                .0
                .content_identity(),
            model_generation: stage.model_generation,
            tokens: stage.layout.prefill_tokens.clone(),
            mapping: stage.layout.prefill_mapping.clone(),
        };
        let bytes = encode_initial_prefill_record_prefix(&record)?;
        let mut geometry = [Vec::new(), Vec::new()];
        for (group, content) in [model, output].into_iter().enumerate() {
            geometry[group] = content
                .tensors
                .iter()
                .map(PrefillTensorGeometry::from_tensor)
                .collect::<Result<_, _>>()?;
        }
        Ok(OriginalDormantPrefillCheckpoint {
            continuation: Arc::clone(&continuation.inner),
            storage: Arc::clone(&parent.storage),
            scope: Arc::clone(&stage.scope),
            source_parent,
            content: OriginalInitialPrefillContent {
                predecessor: source_parent,
                bytes,
                installed: false,
            },
            geometry,
            loaded: None,
            bindings: None,
            allowance,
            reservation,
        })
    }

    pub(super) fn resolve_dormant_prefill_checkpoint(
        &mut self,
        continuation: &SemanticModelContextContinuation,
        original: &mut OriginalDormantPrefillCheckpoint,
        source_native_digest: Identity256,
        target_native_digest: Identity256,
    ) -> Result<SemanticReplayLoadedMaterial, SemanticTransitionError> {
        continuation.inner.require_dormant_parent(self)?;
        let retained = continuation
            .inner
            .original
            .lock()
            .map_err(|_| SemanticTransitionError::Poisoned)?;
        let parent = retained
            .parent
            .as_ref()
            .ok_or(SemanticTransitionError::NotBound)?;
        if !Arc::ptr_eq(&original.continuation, &continuation.inner)
            || !Arc::ptr_eq(&original.storage, &parent.storage)
            || parent
                .prefill
                .as_ref()
                .is_none_or(|stage| !Arc::ptr_eq(&original.scope, &stage.scope))
            || self
                .graph
                .cold_work_allowance()
                .as_ref()
                .is_none_or(|actual| !Arc::ptr_eq(actual, &original.allowance))
            || original
                .bindings
                .is_some_and(|binding| binding != (source_native_digest, target_native_digest))
        {
            return Err(SemanticTransitionError::ObservationMismatch);
        }
        drop(retained);
        original.bindings = Some((source_native_digest, target_native_digest));
        if let Some(loaded) = &original.loaded {
            return Ok(SemanticReplayLoadedMaterial {
                bytes: loaded.bytes.clone(),
                decoded: None,
            });
        }

        let mut bytes = Vec::new();
        let extent = CheckpointInitialPrefill::MAGIC.len()
            + 1
            + 5 * 8
            + 4 * 32
            + original.content.bytes.len()
            + (original.geometry[0].len() + original.geometry[1].len()) * 17 * 8;
        bytes
            .try_reserve_exact(extent)
            .map_err(|_| publication_input_error("dormant prefill section allocation failed"))?;
        bytes.extend_from_slice(CheckpointInitialPrefill::MAGIC);
        material_u64(&mut bytes, 4);
        bytes.push(1);
        encode_publication_identity(&mut bytes, original.source_parent);
        bytes.extend_from_slice(source_native_digest.as_bytes());
        bytes.extend_from_slice(target_native_digest.as_bytes());
        material_u64(&mut bytes, original.content.bytes.len() as u64);
        bytes.extend_from_slice(&original.content.bytes);
        for group in &original.geometry {
            material_u64(&mut bytes, group.len() as u64);
            for geometry in group {
                geometry.encode(&mut bytes);
            }
        }
        original.loaded = Some(SemanticReplayLoadedMaterial {
            bytes: PublicationMaterialBytes::readback(
                bytes,
                Some(Arc::clone(&original.reservation)),
            ),
            decoded: None,
        });
        Ok(SemanticReplayLoadedMaterial {
            bytes: original
                .loaded
                .as_ref()
                .expect("retained prefill section")
                .bytes
                .clone(),
            decoded: None,
        })
    }
}

/// A boundary retaining the restored model owners and complete persistent cache
/// buffers after their actual native copies finish. It authorizes no model
/// call or publication; activation still owns the actual PublishedParent.
#[derive(Clone)]
pub struct SemanticRestoredModelContextPrefill {
    inner: Arc<RestoredModelContextPrefillOwner>,
}

impl SemanticRestoredModelContextPrefill {
    pub fn same_handle(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.inner, &other.inner)
    }

    pub fn material_budget(&self) -> &SemanticReplayMaterialBudget {
        &self.inner._reservation.budget
    }

    pub fn source_layout(&self) -> &SemanticInitialSourceLayout {
        &self.inner.layout
    }

    pub fn source_layout_clone_extent(&self) -> Result<usize, SemanticTransitionError> {
        initial_source_layout_clone_extent(&self.inner.layout)
    }

    pub fn source_mapping(&self) -> &[SemanticSourceMapping] {
        &self.inner.layout.prefill_mapping
    }

    pub fn tensor_count(&self) -> usize {
        self.inner.output_tensors.len()
    }

    pub fn capacity_tensor_count(&self) -> usize {
        self.inner.storage.layouts.len()
    }

    /// Checkpoint identities are lineage only. The separately returned current
    /// identity belongs to the genuinely restored SOURCE publication.
    pub fn completion_lineage(
        &self,
    ) -> (
        SemanticPublishedIdentity,
        Identity256,
        Identity256,
        Identity256,
        u64,
        u64,
    ) {
        (
            self.inner.checkpoint_source_parent,
            self.inner.source_native_digest,
            self.inner.target_native_digest,
            self.inner.task_identity,
            self.inner.task_epoch,
            self.inner.model_generation,
        )
    }

    pub fn restored_source_parent(&self) -> SemanticPublishedIdentity {
        self.inner.source_parent
    }
}

fn initial_source_layout_clone_extent(
    layout: &SemanticInitialSourceLayout,
) -> Result<usize, SemanticTransitionError> {
    let bytes = layout
        .prefill_tokens
        .len()
        .checked_mul(size_of::<u64>())
        .and_then(|bytes| {
            bytes.checked_add(
                layout
                    .prefix
                    .len()
                    .checked_mul(size_of::<SemanticTextSlot>())?,
            )
        })
        .and_then(|bytes| {
            bytes.checked_add(
                layout
                    .prefill_mapping
                    .len()
                    .checked_mul(size_of::<SemanticSourceMapping>())?,
            )
        })
        .and_then(|bytes| bytes.checked_add(layout.provenance_bytes.len()))
        .ok_or(SemanticTransitionError::GenerationExhausted)?;
    layout.prefill_mapping.iter().try_fold(bytes, |bytes, row| {
        bytes
            .checked_add(row.source.len())
            .ok_or(SemanticTransitionError::GenerationExhausted)
    })
}

struct RestoredModelContextPrefillOwner {
    continuation: std::sync::Weak<ModelContextContinuationOwner>,
    issuer: Arc<()>,
    storage: Arc<PublicationStorage>,
    scope: Arc<()>,
    task_identity: Identity256,
    task_epoch: u64,
    model_generation: u64,
    source_parent: SemanticPublishedIdentity,
    checkpoint_source_parent: SemanticPublishedIdentity,
    source_native_digest: Identity256,
    target_native_digest: Identity256,
    layout: SemanticInitialSourceLayout,
    output_tensors: Vec<PreparedSemanticTensor>,
    declaration: SemanticModelContextPrefillRestoreDeclaration,
    _cache_copies: Vec<crate::device::RetainedDeviceMemoryCommand>,
    _section: SemanticReplayLoadedMaterial,
    _reservation: Arc<SemanticReplayMaterialReservation>,
    _copy_reservation: Arc<SemanticReplayMaterialReservation>,
}

impl ModelContextContinuationOwner {
    pub(in crate::semantic_transition) fn with_dormant_prefill_stage<R>(
        &self,
        issuer: &Arc<()>,
        scope: &Arc<()>,
        operation: impl FnOnce(&mut InitialPrefillStage) -> Result<R, SemanticTransitionError>,
    ) -> Result<R, SemanticTransitionError> {
        let mut original = self
            .original
            .lock()
            .map_err(|_| SemanticTransitionError::Poisoned)?;
        if !Arc::ptr_eq(issuer, &self.target_issuer)
            || original.activating
            || original.activated.is_some()
            || original.refused
            || original
                .initial_prefill_scope
                .as_ref()
                .is_none_or(|actual| !Arc::ptr_eq(actual, scope))
        {
            return Err(SemanticTransitionError::ObservationMismatch);
        }
        let parent = original
            .parent
            .as_mut()
            .ok_or(SemanticTransitionError::NotBound)?;
        if !parent.initialized
            || parent.bank.header.abi != 0
            || parent.bank.header.publication_word != 0
        {
            return Err(SemanticTransitionError::ObservationMismatch);
        }
        let stage = parent
            .prefill
            .as_mut()
            .ok_or(SemanticTransitionError::NotBound)?;
        if !Arc::ptr_eq(&stage.scope, scope)
            || stage.model_generation != parent.bank.header.model_generation
        {
            return Err(SemanticTransitionError::ObservationMismatch);
        }
        operation(stage)
    }
}

/// Authenticated completed-prefill geometry before restoration initializes
/// any device counter. These ports declare future owners; they grant no reads.
#[derive(Clone)]
pub struct SemanticModelContextPrefillRestoreDeclaration {
    inner: Arc<ModelContextPrefillRestoreDeclarationOwner>,
}

struct ModelContextPrefillRestoreDeclarationOwner {
    scope: Arc<()>,
    source_publication: Arc<PublicationStorage>,
    source_models: Vec<Arc<ModelGenerationOwner>>,
    layout: SemanticInitialSourceLayout,
    lineage: (
        SemanticPublishedIdentity,
        Identity256,
        Identity256,
        Identity256,
        u64,
        u64,
    ),
    rows: [Vec<prepared_source_content::SemanticSourceGeometryTensor>; 2],
    binding: Mutex<Option<[Vec<PreparedSemanticTensor>; 2]>>,
    _reservation: Arc<SemanticReplayMaterialReservation>,
}

impl SemanticModelContextPrefillRestoreDeclaration {
    pub fn same_handle(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.inner, &other.inner)
    }
    pub fn material_budget(&self) -> &SemanticReplayMaterialBudget {
        &self.inner._reservation.budget
    }
    pub fn source_layout(&self) -> &SemanticInitialSourceLayout {
        &self.inner.layout
    }
    pub fn source_mapping(&self) -> &[SemanticSourceMapping] {
        &self.inner.layout.prefill_mapping
    }
    pub fn source_layout_clone_extent(&self) -> Result<usize, SemanticTransitionError> {
        initial_source_layout_clone_extent(&self.inner.layout)
    }
    pub fn completion_lineage(
        &self,
    ) -> (
        SemanticPublishedIdentity,
        Identity256,
        Identity256,
        Identity256,
        u64,
        u64,
    ) {
        self.inner.lineage
    }
    pub fn tensor_count(&self) -> usize {
        self.inner.rows[1].len()
    }
    pub fn model_tensor_count(&self) -> usize {
        self.inner.rows[0].len()
    }

    pub fn model_storage_layouts(&self) -> &[SemanticModelStorage] {
        &self.inner.source_publication.model_memory.storages
    }

    pub fn model_view_layouts(
        &self,
    ) -> impl Iterator<Item = (SemanticTensorLayout, u64, u64)> + '_ {
        let storage = &self.inner.source_publication;
        storage.model_memory.views.iter().map(|view| {
            (
                storage.layouts[&(view.role, view.index)],
                view.storage,
                view.byte_offset,
            )
        })
    }

    pub fn model_allocation_layouts(
        &self,
    ) -> impl Iterator<
        Item = (
            prepared_source_content::SemanticSourceGeometryAllocation,
            u64,
        ),
    > + '_ {
        self.inner
            .source_publication
            .model_memory
            .allocation_bytes
            .iter()
            .enumerate()
            .map(|(index, &bytes)| {
                (
                    prepared_source_content::SemanticSourceGeometryAllocation::restored_prefill(
                        &self.inner.scope,
                        true,
                        index,
                        bytes,
                    ),
                    bytes,
                )
            })
    }
    pub fn tensor_metadata(
        &self,
    ) -> Result<
        Vec<(
            SemanticTensorLayout,
            u64,
            u64,
            prepared_source_content::SemanticSourceGeometryAllocation,
            u64,
        )>,
        SemanticTransitionError,
    > {
        self.metadata(1)
    }
    pub fn model_tensor_metadata(
        &self,
    ) -> Result<
        Vec<(
            SemanticTensorLayout,
            u64,
            u64,
            prepared_source_content::SemanticSourceGeometryAllocation,
            u64,
        )>,
        SemanticTransitionError,
    > {
        self.metadata(0)
    }
    fn metadata(
        &self,
        group: usize,
    ) -> Result<
        Vec<(
            SemanticTensorLayout,
            u64,
            u64,
            prepared_source_content::SemanticSourceGeometryAllocation,
            u64,
        )>,
        SemanticTransitionError,
    > {
        let mut rows = Vec::new();
        rows.try_reserve_exact(self.inner.rows[group].len())
            .map_err(|_| {
                publication_input_error("restored prefill declaration projection allocation failed")
            })?;
        rows.extend(self.inner.rows[group].iter().map(|row| {
            (
                row.layout,
                row.logical_begin,
                row.logical_end,
                row.allocation.clone(),
                row.byte_offset,
            )
        }));
        Ok(rows)
    }
    pub fn require_bound(
        &self,
        proof: &SemanticRestoredModelContextPrefill,
    ) -> Result<(), SemanticTransitionError> {
        if !self.same_handle(&proof.inner.declaration)
            || !Arc::ptr_eq(&self.inner.scope, &proof.inner.scope)
            || self
                .inner
                .binding
                .lock()
                .map_err(|_| SemanticTransitionError::Poisoned)?
                .is_none()
        {
            return Err(SemanticTransitionError::ObservationMismatch);
        }
        Ok(())
    }
    pub fn allocation_matches(
        &self,
        declared: &prepared_source_content::SemanticSourceGeometryAllocation,
        actual: &DeviceAllocationProvenance,
    ) -> Result<bool, SemanticTransitionError> {
        let row = self
            .inner
            .rows
            .iter()
            .enumerate()
            .find_map(|(group, rows)| {
                rows.iter()
                    .enumerate()
                    .find(|(_, row)| row.allocation.same_handle(declared))
                    .map(|(ordinal, row)| (group, ordinal, row))
            })
            .ok_or(SemanticTransitionError::ObservationMismatch)?;
        if declared.span_bytes() != row.2.allocation.span_bytes() {
            return Err(SemanticTransitionError::ObservationMismatch);
        }
        let binding = self
            .inner
            .binding
            .lock()
            .map_err(|_| SemanticTransitionError::Poisoned)?;
        let Some(bound) = binding.as_ref() else {
            return Ok(false);
        };
        let tensor = bound[row.0]
            .get(row.1)
            .ok_or(SemanticTransitionError::ObservationMismatch)?;
        let expected = tensor
            .native_allocation
            .as_ref()
            .ok_or(SemanticTransitionError::ObservationMismatch)?;
        Ok(expected.same_allocation(actual)
            && expected.allocation_bytes() == actual.allocation_bytes())
    }
    fn bind(
        &self,
        groups: [&InitialPrefillTensorOwners; 2],
    ) -> Result<(), SemanticTransitionError> {
        let mut binding = self
            .inner
            .binding
            .lock()
            .map_err(|_| SemanticTransitionError::Poisoned)?;
        if binding.is_some() {
            return Err(SemanticTransitionError::ObservationMismatch);
        }
        for (group, content) in groups.iter().enumerate() {
            if content.tensors.len() != self.inner.rows[group].len() {
                return Err(SemanticTransitionError::ObservationMismatch);
            }
            for (row, tensor) in self.inner.rows[group].iter().zip(&content.tensors) {
                let actual = tensor
                    .native_allocation
                    .as_ref()
                    .ok_or(SemanticTransitionError::ObservationMismatch)?;
                if row.layout != tensor.layout
                    || row.logical_begin != tensor.logical_begin
                    || row.logical_end != tensor.logical_end
                    || row.allocation.span_bytes() != actual.allocation_bytes() as u64
                    || row.byte_offset != actual.byte_offset() as u64
                {
                    return Err(SemanticTransitionError::ObservationMismatch);
                }
            }
        }
        *binding = Some([groups[0].tensors.clone(), groups[1].tensors.clone()]);
        Ok(())
    }
}

pub(super) struct OriginalRestoredModelContextPrefill {
    continuation: std::sync::Weak<ModelContextContinuationOwner>,
    source_parent: SemanticPublishedIdentity,
    checkpoint_source_parent: SemanticPublishedIdentity,
    source_native_digest: Identity256,
    target_native_digest: Identity256,
    record: DecodedInitialPrefillContent,
    geometry: [Vec<PrefillTensorGeometry>; 2],
    copy_plans: Vec<Vec<(u64, u64, usize)>>,
    cache_copies: Vec<crate::device::RetainedDeviceMemoryCommand>,
    copy_cursor: usize,
    copy_dma_recorded: bool,
    layout: SemanticInitialSourceLayout,
    declaration: SemanticModelContextPrefillRestoreDeclaration,
    section: Option<SemanticReplayLoadedMaterial>,
    reservation: Arc<SemanticReplayMaterialReservation>,
    copy_reservation: Arc<SemanticReplayMaterialReservation>,
    stage: Option<InitialPrefillStage>,
    storage: Option<Arc<PublicationStorage>>,
    allowance: Option<Arc<Mutex<native_work_bound::ColdNativeAllowance>>>,
    work: Option<SemanticColdModelWork>,
    result: Option<SemanticRestoredModelContextPrefill>,
    poisoned: bool,
}

fn require_prefill_roster(
    material: &OwnedPublicationMaterial,
    groups: &[Vec<PrefillTensorGeometry>; 2],
) -> Result<(), SemanticTransitionError> {
    for (model, geometry) in [(true, &groups[0]), (false, &groups[1])] {
        let expected = material
            .layouts
            .keys()
            .filter(|(role, _)| {
                if model {
                    matches!(*role, 18..=25)
                } else {
                    matches!(*role, 4..=13)
                }
            })
            .copied()
            .chain(
                (0..if model {
                    material.model_allocations.len()
                } else {
                    0
                })
                    .map(|index| (0, index as u64)),
            )
            .collect::<BTreeSet<_>>();
        let mut seen = BTreeSet::new();
        if geometry.len() != expected.len() {
            return Err(publication_input_error(
                "restored prefill requires the full original tensor roster",
            ));
        }
        for geometry in geometry {
            let key = (geometry.layout.role, geometry.layout.index);
            if !expected.contains(&key) || !seen.insert(key) {
                return Err(publication_input_error(
                    "restored prefill changed its tensor coordinates",
                ));
            }
            if key.0 == 0 {
                if geometry.physical_bytes as u64
                    != material.model_memory.allocation_bytes[key.1 as usize]
                    || geometry.logical_begin != 0
                    || geometry.logical_end != 0
                {
                    return Err(publication_input_error(
                        "restored prefill changed its full model backing allocation",
                    ));
                }
            } else {
                let range = material
                    .ranges
                    .iter()
                    .find(|item| (item.range.role, item.range.index) == key)
                    .ok_or(SemanticTransitionError::ObservationMismatch)?
                    .range;
                let expected_layout = if model {
                    geometry.layout
                } else if matches!(key.0, 4 | 5) {
                    prefix_capacity_layout(geometry.layout, material.contract.prefix_capacity)?
                } else {
                    canonical_tensor_layout(geometry.layout)?
                };
                if material.layouts.get(&key) != Some(&expected_layout)
                    || range.logical_begin != geometry.logical_begin
                    || range.logical_end != geometry.logical_end
                    || range.length_bytes != tensor_layout_bytes(&expected_layout)? as u64
                {
                    return Err(publication_input_error(
                        "restored prefill changed its tensor type, layout or coverage interval",
                    ));
                }
            }
        }
        if seen != expected {
            return Err(SemanticTransitionError::ObservationMismatch);
        }
    }
    Ok(())
}

fn require_prefill_source_layout(
    material: &OwnedPublicationMaterial,
    layout: &SemanticInitialSourceLayout,
    record: &DecodedInitialPrefillContent,
) -> Result<(), SemanticTransitionError> {
    let header = material.bank.header;
    let prefix = material
        .ranges
        .iter()
        .find(|item| {
            (item.range.role, item.range.index) == (SemanticStateRole::PrefixSource as u64, 0)
        })
        .ok_or(SemanticTransitionError::ObservationMismatch)?;
    let provenance = material
        .ranges
        .iter()
        .find(|item| {
            (item.range.role, item.range.index)
                == (SemanticStateRole::TokenProvenanceRecords as u64, 0)
        })
        .ok_or(SemanticTransitionError::ObservationMismatch)?;
    if layout.prefill_tokens != record.tokens
        || layout.prefill_mapping.len() != record.mapping.len()
        || layout
            .prefill_mapping
            .iter()
            .zip(&record.mapping)
            .any(|(actual, saved)| {
                actual.source != saved.source
                    || actual.offset != saved.offset
                    || actual.logical_position != saved.logical_position
            })
        || layout.prefix.len() != record.tokens.len()
        || header.prefix_extent != record.tokens.len() as u64
        || header.ring_head != layout.ring_head
        || !same_initial_source_rows(&layout.source, &material.bank.source)
        || prefix.range.logical_begin != 0
        || prefix.range.logical_end != header.prefix_extent
        || prefix.bytes.len()
            != layout
                .prefix
                .len()
                .checked_mul(size_of::<SemanticTextSlot>())
                .ok_or(SemanticTransitionError::GenerationExhausted)?
        || provenance.range.logical_end != layout.provenance_records
        || provenance.capacity != layout.provenance_capacity_bytes
        || provenance.bytes.as_ref() != layout.provenance_bytes.as_slice()
    {
        return Err(publication_input_error(
            "restored prefill differs from its admitted source, token mapping or provenance",
        ));
    }
    for (ordinal, bytes) in prefix
        .bytes
        .as_chunks::<{ size_of::<SemanticTextSlot>() }>()
        .0
        .iter()
        .enumerate()
    {
        // SAFETY: each checked chunk is one complete padding-free integer ABI
        // row; an unaligned material allocation is never borrowed as a C row.
        let actual = unsafe { bytes.as_ptr().cast::<SemanticTextSlot>().read_unaligned() };
        if !same_initial_source_rows(
            std::slice::from_ref(&layout.prefix[ordinal]),
            std::slice::from_ref(&actual),
        ) || actual.token != record.tokens[ordinal]
            || actual.logical_position != ordinal as u64
            || actual.kind != 1
            || actual.provenance != 1
            || actual.valid != 1
            || actual.committed != 1
        {
            return Err(publication_input_error(
                "restored prefill ordered prefix content differs",
            ));
        }
    }
    Ok(())
}

impl OriginalRestoredModelContextPrefill {
    pub(super) fn require_source_layout(
        &self,
        layout: &SemanticInitialSourceLayout,
    ) -> Result<(), SemanticTransitionError> {
        let saved = &self.layout;
        if !same_initial_source_rows(&saved.source, &layout.source)
            || !same_initial_source_rows(&saved.prefix, &layout.prefix)
            || saved.prefill_tokens != layout.prefill_tokens
            || saved.ring_head != layout.ring_head
            || saved.provenance_bytes != layout.provenance_bytes
            || saved.provenance_capacity_bytes != layout.provenance_capacity_bytes
            || saved.provenance_records != layout.provenance_records
            || saved.prefill_mapping.len() != layout.prefill_mapping.len()
            || saved
                .prefill_mapping
                .iter()
                .zip(&layout.prefill_mapping)
                .any(|(saved, actual)| {
                    saved.source != actual.source
                        || saved.offset != actual.offset
                        || saved.logical_position != actual.logical_position
                })
        {
            return Err(SemanticTransitionError::ObservationMismatch);
        }
        Ok(())
    }

    /// Decode under the original pool before admitting any restoration effects.
    /// `layout` is produced by the canonical admitted-observation planner, not
    /// by a replacement prefill or model-output callback.
    pub(super) fn prepare(
        continuation: &SemanticModelContextContinuation,
        target: &SemanticTransitionSession,
        source_parent: (SemanticPublishedIdentity, SemanticPublishedIdentity),
        source_native_digest: Identity256,
        target_native_digest: Identity256,
        section: SemanticReplayLoadedMaterial,
        material: &dormant_material::DormantPublicationMaterial,
        layout: SemanticInitialSourceLayout,
    ) -> Result<Self, SemanticTransitionError> {
        let budget = target
            .replay_material_budget
            .as_ref()
            .ok_or(SemanticTransitionError::NotBound)?;
        section.require_budget(budget)?;
        let bound = section
            .as_bytes()
            .len()
            .checked_mul(12)
            .and_then(|bytes| {
                bytes.checked_add(
                    size_of::<Self>()
                        + size_of::<RestoredModelContextPrefillOwner>()
                        + 8 * size_of::<usize>()
                        + state_restoration::OriginalDeviceWrite::metadata_allocation_bytes::<u64>(
                        ),
                )
            })
            .and_then(|bytes| u64::try_from(bytes).ok())
            .ok_or(SemanticTransitionError::GenerationExhausted)?;
        budget.require_host_allocation(bound)?;
        let reservation = Arc::new(budget.reserve_auxiliary(bound)?);
        let mut reader = SemanticMaterialReader::new(section.as_bytes());
        if reader
            .take(CheckpointInitialPrefill::MAGIC.len())
            .map_err(SemanticTransitionError::Semantic)?
            != CheckpointInitialPrefill::MAGIC
            || reader.u64().map_err(SemanticTransitionError::Semantic)? != 4
            || reader.u8().map_err(SemanticTransitionError::Semantic)? != 1
            || decode_publication_identity(&mut reader)? != source_parent.1
            || prefill_identity(&mut reader)? != source_native_digest
            || prefill_identity(&mut reader)? != target_native_digest
            || material.source.native_digest != source_native_digest
        {
            return Err(publication_input_error(
                "dormant prefill belongs to another complete source or target checkpoint",
            ));
        }
        let content_extent = prefill_extent(&mut reader)?;
        let record = CheckpointInitialPrefill::parse_content(
            reader
                .take(content_extent)
                .map_err(SemanticTransitionError::Semantic)?,
        )?;
        let mut geometry = [Vec::new(), Vec::new()];
        for group in &mut geometry {
            let count = prefill_extent(&mut reader)?;
            if count > section.as_bytes().len() / (17 * 8) {
                return Err(publication_input_error(
                    "dormant prefill geometry exceeds its complete section",
                ));
            }
            group.try_reserve_exact(count).map_err(|_| {
                publication_input_error("restored prefill geometry allocation failed")
            })?;
            for _ in 0..count {
                group.push(PrefillTensorGeometry::decode(&mut reader)?);
            }
        }
        reader.finish().map_err(SemanticTransitionError::Semantic)?;
        let native = &material.material;
        let task_content = target
            .task
            .as_ref()
            .ok_or(SemanticTransitionError::NotBound)?
            .0
            .content_identity();
        if native.bank.header.abi != 0
            || native.bank.header.publication_word != 0
            || native.bank.header.authority_generation != target.task_epoch
            || native.contract.task_identity != record.task_identity
            || native.contract.model_generation != material.source.generation
            || native.bank.header.model_generation != material.source.generation
            || native.bank.header.model_geometry_digest != material.source.geometry_digest
            || native.model_memory.digest(&native.layouts)? != material.source.geometry_digest
            || target.task_evaluation_identity() != Some(record.task_identity)
            || native.model_memory != material.source.model_memory
            || native.contract.model_contract_layout != material.source.model_contract_layout
        {
            return Err(SemanticTransitionError::ObservationMismatch);
        }
        let model_contract = native
            .ranges
            .iter()
            .find(|item| {
                (item.range.role, item.range.index) == (SemanticStateRole::ModelContract as u64, 0)
            })
            .ok_or(SemanticTransitionError::ObservationMismatch)?;
        if model_contract.range.generation != 1
            || model_contract.bytes.as_ref() != material.source.model_contract_record.as_ref()
            || native
                .layouts
                .iter()
                .filter(|((role, _), _)| matches!(*role, 18..=25))
                .any(|(key, layout)| material.source.layouts.get(key) != Some(layout))
            || native
                .layouts
                .keys()
                .filter(|(role, _)| matches!(*role, 18..=25))
                .count()
                != material
                    .source
                    .layouts
                    .keys()
                    .filter(|(role, _)| matches!(*role, 18..=25))
                    .count()
        {
            return Err(publication_input_error(
                "restored prefill changed its full source model contract or backing view geometry",
            ));
        }
        record.verify(
            native.bank.header.semantic_digest,
            material.source.generation,
            native.contract.task_identity,
            target.task_epoch,
            task_content,
            native.bank.header.prefix_extent,
        )?;
        require_prefill_roster(native, &geometry)?;
        require_prefill_source_layout(native, &layout, &record)?;
        let declaration_scratch = geometry[0]
            .len()
            .checked_add(geometry[1].len())
            .and_then(|count| count.checked_mul(size_of::<(bool, usize, u64, u64)>()))
            .and_then(|bytes| bytes.checked_add(2 * size_of::<Vec<(bool, usize, u64, u64)>>()))
            .and_then(|bytes| u64::try_from(bytes).ok())
            .ok_or(SemanticTransitionError::GenerationExhausted)?;
        budget.require_host_allocation(declaration_scratch)?;
        let _declaration_scratch = budget.reserve_auxiliary(declaration_scratch)?;
        let declaration_rows = geometry
            .iter()
            .enumerate()
            .map(|(group, rows)| {
                rows.iter()
                    .enumerate()
                    .map(|(ordinal, geometry)| {
                        let key = (geometry.layout.role, geometry.layout.index);
                        let (model, index, span, offset) = if group == 0 {
                            let (index, offset) = if key.0 == 0 {
                                (
                                    usize::try_from(key.1).map_err(|_| {
                                        SemanticTransitionError::ObservationMismatch
                                    })?,
                                    0,
                                )
                            } else {
                                material.source.model_memory.location(key.0, key.1)?
                            };
                            let span = *material
                                .source
                                .model_memory
                                .allocation_bytes
                                .get(index)
                                .ok_or(SemanticTransitionError::ObservationMismatch)?;
                            (true, index, span, offset as u64)
                        } else {
                            let range = native
                                .ranges
                                .iter()
                                .find(|item| (item.range.role, item.range.index) == key)
                                .ok_or(SemanticTransitionError::ObservationMismatch)?;
                            if native.layouts.get(&key) == Some(&geometry.layout) {
                                (
                                    false,
                                    usize::try_from(range.range.storage_slot).map_err(|_| {
                                        SemanticTransitionError::ObservationMismatch
                                    })?,
                                    range.capacity as u64,
                                    range.range.offset_bytes,
                                )
                            } else {
                                (
                                    false,
                                    native
                                        .ranges
                                        .len()
                                        .checked_add(ordinal)
                                        .ok_or(SemanticTransitionError::GenerationExhausted)?,
                                    geometry.physical_bytes as u64,
                                    0,
                                )
                            }
                        };
                        if offset
                            .checked_add(tensor_layout_bytes(&geometry.layout)? as u64)
                            .is_none_or(|end| end > span)
                        {
                            return Err(SemanticTransitionError::ObservationMismatch);
                        }
                        Ok((model, index, span, offset))
                    })
                    .collect::<Result<Vec<_>, SemanticTransitionError>>()
            })
            .collect::<Result<Vec<_>, _>>()?;
        let declaration_count = geometry[0]
            .len()
            .checked_add(geometry[1].len())
            .ok_or(SemanticTransitionError::GenerationExhausted)?;
        let declaration_bound = declaration_count
            .checked_mul(
                2 * (size_of::<prepared_source_content::SemanticSourceGeometryTensor>()
                    + size_of::<PreparedSemanticTensor>()),
            )
            .and_then(|bytes| {
                bytes.checked_add(size_of::<ModelContextPrefillRestoreDeclarationOwner>())
            })
            .and_then(|bytes| {
                bytes.checked_add(
                    initial_source_layout_clone_extent(&layout)
                        .ok()?
                        .checked_mul(2)?,
                )
            })
            .and_then(|bytes| u64::try_from(bytes).ok())
            .ok_or(SemanticTransitionError::GenerationExhausted)?;
        budget.require_host_allocation(declaration_bound)?;
        let declaration_reservation = Arc::new(budget.reserve_auxiliary(declaration_bound)?);
        let scope = Arc::new(());
        let mut rows = [Vec::new(), Vec::new()];
        for (group, saved) in geometry.iter().enumerate() {
            rows[group].try_reserve_exact(saved.len()).map_err(|_| {
                publication_input_error("prefill declaration row allocation failed")
            })?;
            for (geometry, &(model, index, span, offset)) in
                saved.iter().zip(&declaration_rows[group])
            {
                rows[group].push(prepared_source_content::SemanticSourceGeometryTensor {
                    layout: geometry.layout,
                    logical_begin: geometry.logical_begin,
                    logical_end: geometry.logical_end,
                    byte_offset: offset,
                    allocation:
                        prepared_source_content::SemanticSourceGeometryAllocation::restored_prefill(
                            &scope, model, index, span,
                        ),
                });
            }
        }
        let declaration = SemanticModelContextPrefillRestoreDeclaration {
            inner: Arc::new(ModelContextPrefillRestoreDeclarationOwner {
                scope,
                source_publication: Arc::clone(&continuation.inner.source_publication),
                source_models: continuation.inner.models.clone(),
                layout: layout.clone(),
                lineage: (
                    source_parent.1,
                    source_native_digest,
                    target_native_digest,
                    record.task_identity,
                    record.task_epoch,
                    record.model_generation,
                ),
                rows,
                binding: Mutex::new(None),
                _reservation: declaration_reservation,
            }),
        };

        let copy_rows = geometry[1].iter().try_fold(0usize, |rows, geometry| {
            let layout = native
                .layouts
                .get(&(geometry.layout.role, geometry.layout.index))
                .ok_or(SemanticTransitionError::ObservationMismatch)?;
            let count = if *layout == geometry.layout {
                0
            } else {
                tensor_copy_plan_geometry(&geometry.layout, layout)?.2
            };
            rows.checked_add(count)
                .ok_or(SemanticTransitionError::GenerationExhausted)
        })?;
        let copy_bound = copy_rows
            .checked_mul(
                2 * (size_of::<(u64, u64, usize)>()
                    + size_of::<crate::device::RetainedDeviceMemoryCommand>()),
            )
            .and_then(|bytes| u64::try_from(bytes).ok())
            .ok_or(SemanticTransitionError::GenerationExhausted)?;
        budget.require_host_allocation(copy_bound)?;
        let copy_reservation = Arc::new(budget.reserve_auxiliary(copy_bound)?);
        let mut cache_copies = Vec::new();
        cache_copies.try_reserve_exact(copy_rows).map_err(|_| {
            publication_input_error("restored prefill cache copy custody allocation failed")
        })?;
        // Use the original canonical copy geometry in reverse. This preserves
        // the saved producer's exact shape/strides while reading the genuine
        // restored capacity layout, including all attention heads.
        let copy_plans = geometry[1]
            .iter()
            .map(|geometry| {
                let layout = native
                    .layouts
                    .get(&(geometry.layout.role, geometry.layout.index))
                    .ok_or(SemanticTransitionError::ObservationMismatch)?;
                if *layout == geometry.layout {
                    return Ok(Vec::new());
                }
                tensor_copy_plan(&geometry.layout, layout)
            })
            .collect::<Result<Vec<_>, SemanticTransitionError>>()?;
        Ok(Self {
            continuation: Arc::downgrade(&continuation.inner),
            source_parent: source_parent.0,
            checkpoint_source_parent: source_parent.1,
            source_native_digest,
            target_native_digest,
            record,
            geometry,
            copy_plans,
            cache_copies,
            copy_cursor: 0,
            copy_dma_recorded: false,
            layout,
            declaration,
            section: Some(section),
            reservation,
            copy_reservation,
            stage: None,
            storage: None,
            allowance: None,
            work: None,
            result: None,
            poisoned: false,
        })
    }

    pub(super) fn native_ceilings(&self) -> Result<([u64; 9], [u64; 9]), SemanticTransitionError> {
        let mut dma = [0u64; 9];
        for (_, _, bytes) in self.copy_plans.iter().flatten() {
            dma[0] = dma[0]
                .checked_add(1)
                .ok_or(SemanticTransitionError::GenerationExhausted)?;
            dma[8] = dma[8]
                .checked_add(*bytes as u64)
                .ok_or(SemanticTransitionError::GenerationExhausted)?;
        }
        Ok((dma, dma))
    }

    pub(super) fn completion_pending(&self) -> Result<bool, SemanticTransitionError> {
        Ok(self
            .cache_copies
            .iter()
            .any(crate::device::RetainedDeviceMemoryCommand::completion_pending))
    }

    pub(super) fn proof(
        &self,
    ) -> Result<SemanticRestoredModelContextPrefill, SemanticTransitionError> {
        self.result.clone().ok_or(SemanticTransitionError::NotBound)
    }

    pub(super) fn declaration(
        &self,
    ) -> Result<SemanticModelContextPrefillRestoreDeclaration, SemanticTransitionError> {
        Ok(self.declaration.clone())
    }

    pub(super) fn take_verified_stage(
        &mut self,
    ) -> Result<InitialPrefillStage, SemanticTransitionError> {
        if self.result.is_none() {
            return Err(SemanticTransitionError::ObservationMismatch);
        }
        self.stage.take().ok_or(SemanticTransitionError::NotBound)
    }
}

fn restored_prefill_tensor(
    geometry: &PrefillTensorGeometry,
    storage: &PublicationStorage,
    owner: &ModelContextContinuationOwner,
) -> Result<PreparedSemanticTensor, SemanticTransitionError> {
    let key = (geometry.layout.role, geometry.layout.index);
    let view = if key.0 == 0 {
        let allocation = owner
            .models
            .get(usize::try_from(key.1).map_err(|_| SemanticTransitionError::ObservationMismatch)?)
            .ok_or(SemanticTransitionError::ObservationMismatch)?;
        if allocation.allocation.len() != geometry.physical_bytes {
            return Err(SemanticTransitionError::ObservationMismatch);
        }
        allocation.allocation.view()
    } else {
        let range = storage.bank_templates[0]
            .iter()
            .find(|range| (range.role, range.index) == key)
            .ok_or(SemanticTransitionError::ObservationMismatch)?;
        if range.logical_begin != geometry.logical_begin
            || range.logical_end != geometry.logical_end
        {
            return Err(SemanticTransitionError::ObservationMismatch);
        }
        let allocation = storage
            .allocations
            .get(
                usize::try_from(range.storage_slot)
                    .map_err(|_| SemanticTransitionError::ObservationMismatch)?,
            )
            .ok_or(SemanticTransitionError::ObservationMismatch)?;
        let begin = usize::try_from(range.offset_bytes)
            .map_err(|_| SemanticTransitionError::ObservationMismatch)?;
        let bytes = usize::try_from(range.length_bytes)
            .map_err(|_| SemanticTransitionError::ObservationMismatch)?;
        let end = begin
            .checked_add(bytes)
            .filter(|end| *end <= allocation.len())
            .ok_or(SemanticTransitionError::ObservationMismatch)?;
        let view = allocation.slice()?.view().slice(begin..end);
        if matches!(key.0, 18..=25) {
            let (ordinal, offset) = owner
                .source_publication
                .model_memory
                .location(key.0, key.1)?;
            let expected = owner
                .models
                .get(ordinal)
                .ok_or(SemanticTransitionError::ObservationMismatch)?;
            let expected_view = expected.allocation.view().slice(
                offset
                    ..offset
                        .checked_add(geometry.physical_bytes)
                        .filter(|end| *end <= expected.allocation.len())
                        .ok_or(SemanticTransitionError::ObservationMismatch)?,
            );
            if view.device_ptr() != expected_view.device_ptr()
                || view
                    .allocation_provenance()
                    .zip(expected_view.allocation_provenance())
                    .is_none_or(|(actual, expected)| !actual.same_allocation(&expected))
            {
                return Err(publication_input_error(
                    "restored prefill model view replaced its actual source owner",
                ));
            }
        }
        view
    };
    let layout = if key.0 == 0 {
        geometry.layout
    } else {
        *storage
            .layouts
            .get(&key)
            .ok_or(SemanticTransitionError::ObservationMismatch)?
    };
    Ok(PreparedSemanticTensor {
        layout,
        logical_begin: geometry.logical_begin,
        logical_end: geometry.logical_end,
        data: *view.device_ptr(),
        native_allocation: view.allocation_provenance(),
        source: Some(view),
        _empty_owner: None,
    })
}

impl SemanticTransitionSession {
    pub(super) fn resolve_restored_model_context_prefill(
        &mut self,
        continuation: &SemanticModelContextContinuation,
        source_parent: SemanticPublishedIdentity,
        source_native_digest: Identity256,
        target_native_digest: Identity256,
        work: &SemanticColdModelWork,
        original: &mut OriginalRestoredModelContextPrefill,
    ) -> Result<SemanticRestoredModelContextPrefill, SemanticTransitionError> {
        let owner = &continuation.inner;
        let allowance = self.require_dormant_model_context_restore_work(continuation, work)?;
        let native_work = self
            .graph
            .active_cold_work()
            .ok_or(SemanticTransitionError::NotBound)?;
        if !original.continuation.ptr_eq(&Arc::downgrade(owner))
            || original.source_parent != source_parent
            || original.source_native_digest != source_native_digest
            || original.target_native_digest != target_native_digest
            || original
                .work
                .as_ref()
                .is_some_and(|actual| !actual.same_invocation(work))
            || original
                .allowance
                .as_ref()
                .is_some_and(|actual| !Arc::ptr_eq(actual, &allowance))
            || self
                .graph
                .cold_work_allowance()
                .as_ref()
                .is_none_or(|actual| !Arc::ptr_eq(actual, &allowance))
        {
            return Err(SemanticTransitionError::ObservationMismatch);
        }
        if let Some(proof) = &original.result {
            return Ok(proof.clone());
        }
        self.require_model_context_numerical_contract(continuation)?;
        let (bank, _, _, _) = self.dormant_state_material_restore_recipe()?;
        if bank.header.model_generation != original.record.model_generation
            || bank.header.authority_generation != original.record.task_epoch
            || bank.header.semantic_digest != original.record.semantic_root
        {
            return Err(SemanticTransitionError::ObservationMismatch);
        }
        let storage = Arc::clone(
            self.publication
                .as_ref()
                .ok_or(SemanticTransitionError::NotBound)?,
        );
        if original
            .storage
            .as_ref()
            .is_some_and(|actual| !Arc::ptr_eq(actual, &storage))
        {
            return Err(SemanticTransitionError::ObservationMismatch);
        }
        original.storage = Some(Arc::clone(&storage));
        original.allowance = Some(Arc::clone(&allowance));
        original.work = Some(work.clone());
        if original.stage.is_none() {
            let scope = Arc::clone(&original.declaration.inner.scope);
            let mut contents = Vec::with_capacity(2);
            if original
                .cache_copies
                .iter()
                .any(crate::device::RetainedDeviceMemoryCommand::entered)
            {
                return Err(SemanticTransitionError::ObservationMismatch);
            }
            original.cache_copies.clear();
            for (group, geometry) in original.geometry.iter().enumerate() {
                let mut tensors = geometry
                    .iter()
                    .map(|geometry| restored_prefill_tensor(geometry, &storage, owner))
                    .collect::<Result<Vec<_>, _>>()?;
                if group == 1 {
                    for ((tensor, saved), plan) in
                        tensors.iter_mut().zip(geometry).zip(&original.copy_plans)
                    {
                        if plan.is_empty() {
                            continue;
                        }
                        let source = tensor
                            .source
                            .as_ref()
                            .ok_or(SemanticTransitionError::ObservationMismatch)?;
                        let compact =
                            allocate_publication::<u8>(&self.provider, saved.physical_bytes)?;
                        let destination = compact.view();
                        for &(compact_offset, capacity_offset, bytes) in plan {
                            let source_offset = usize::try_from(capacity_offset)
                                .map_err(|_| SemanticTransitionError::GenerationExhausted)?;
                            let destination_offset = usize::try_from(compact_offset)
                                .map_err(|_| SemanticTransitionError::GenerationExhausted)?;
                            let source = source
                                .try_slice(
                                    source_offset
                                        ..source_offset
                                            .checked_add(bytes)
                                            .ok_or(SemanticTransitionError::GenerationExhausted)?,
                                )
                                .ok_or(SemanticTransitionError::ObservationMismatch)?;
                            let destination = destination
                                .try_slice(
                                    destination_offset
                                        ..destination_offset
                                            .checked_add(bytes)
                                            .ok_or(SemanticTransitionError::GenerationExhausted)?,
                                )
                                .ok_or(SemanticTransitionError::ObservationMismatch)?;
                            original.cache_copies.push(
                                crate::device::RetainedDeviceMemoryCommand::copy(
                                    &self.stream,
                                    source,
                                    destination,
                                )
                                .map_err(|error| {
                                    runtime_error("restored prefill cache copy staging", error)
                                })?,
                            );
                        }
                        *tensor = PreparedSemanticTensor {
                            layout: saved.layout,
                            logical_begin: saved.logical_begin,
                            logical_end: saved.logical_end,
                            data: *destination.device_ptr(),
                            native_allocation: destination.allocation_provenance(),
                            source: Some(destination),
                            _empty_owner: None,
                        };
                    }
                }
                contents.push(InitialPrefillTensorOwners { tensors });
            }
            let output_content = contents.pop().expect("two restored content groups");
            let model_content = contents.pop().expect("two restored content groups");
            original.stage = Some(InitialPrefillStage {
                scope,
                task_identity: original.record.task_identity,
                task_epoch: original.record.task_epoch,
                model_generation: original.record.model_generation,
                layout: original.layout.clone(),
                input: None,
                model_content: Some(model_content),
                output_content: Some(output_content),
                consumer_stream: None,
            });
        }
        while original.copy_cursor < original.cache_copies.len() {
            let copy = &mut original.cache_copies[original.copy_cursor];
            allowance
                .lock()
                .map_err(|_| SemanticTransitionError::Poisoned)?
                .require_submitted_dma(copy.destination().len(), original.copy_dma_recorded)?;
            let mut submitted = Ok(());
            if !copy.entered() {
                let mut recorder = self.domain.new_strict_recorder();
                recorder.write(copy.destination());
                recorder.read(
                    copy.source()
                        .ok_or(SemanticTransitionError::ObservationMismatch)?,
                );
                submitted =
                    enqueue_recorded(&self.domain, &mut original.poisoned, recorder, |_| {
                        copy.enqueue()
                    });
            }
            if copy.submitted() && !original.copy_dma_recorded {
                allowance
                    .lock()
                    .map_err(|_| SemanticTransitionError::Poisoned)?
                    .record_submitted_dma(
                        copy.destination().len(),
                        &mut original.copy_dma_recorded,
                    )?;
            }
            submitted?;
            copy.resolve().map_err(|error| {
                runtime_error("original restored prefill cache copy completion", error)
            })?;
            original.copy_cursor += 1;
            original.copy_dma_recorded = false;
        }
        let groups = [
            stage
                .model_content
                .as_ref()
                .expect("restored original model owner"),
            stage
                .output_content
                .as_ref()
                .expect("restored original output owner"),
        ];
        original.declaration.bind(groups)?;
        let proof = SemanticRestoredModelContextPrefill {
            inner: Arc::new(RestoredModelContextPrefillOwner {
                continuation: Arc::downgrade(owner),
                issuer: Arc::clone(&self.publication_issuer),
                storage,
                scope: Arc::clone(&stage.scope),
                task_identity: stage.task_identity,
                task_epoch: stage.task_epoch,
                model_generation: stage.model_generation,
                source_parent: original.source_parent,
                checkpoint_source_parent: original.checkpoint_source_parent,
                source_native_digest: original.source_native_digest,
                target_native_digest: original.target_native_digest,
                layout: stage.layout.clone(),
                output_tensors: groups[1].tensors.clone(),
                declaration: original.declaration.clone(),
                _cache_copies: std::mem::take(&mut original.cache_copies),
                _section: original
                    .section
                    .take()
                    .ok_or(SemanticTransitionError::ObservationMismatch)?,
                _reservation: Arc::clone(&original.reservation),
                _copy_reservation: Arc::clone(&original.copy_reservation),
            }),
        };
        original.result = Some(proof.clone());
        Ok(proof)
    }

    pub fn require_restored_model_context_prefill(
        &self,
        continuation: &SemanticModelContextContinuation,
        proof: &SemanticRestoredModelContextPrefill,
    ) -> Result<(), SemanticTransitionError> {
        self.require_model_context_numerical_contract(continuation)?;
        let actual = &proof.inner;
        let original = continuation
            .inner
            .original
            .lock()
            .map_err(|_| SemanticTransitionError::Poisoned)?;
        let scope = original
            .parent
            .as_ref()
            .and_then(|parent| parent.prefill.as_ref())
            .map(|stage| &stage.scope)
            .or_else(|| {
                self.retained_initial_prefill
                    .as_ref()
                    .map(|stage| &stage.scope)
            });
        let activated_basis = if let Some(identity) = original.activated {
            self.retained_initial_prefill
                .as_ref()
                .is_some_and(|retained| {
                    retained.record.predecessor == identity
                        && retained.model_generation == actual.model_generation
                        && retained.task_identity == actual.task_identity
                        && retained.task_epoch == actual.task_epoch
                        && original
                            .initial_prefill_scope
                            .as_ref()
                            .is_some_and(|scope| Arc::ptr_eq(scope, &actual.scope))
                })
                && self.steps.iter().any(|(token, step)| {
                    step.identity == Some(identity)
                        && self.readers.get(token).is_some_and(|reader| {
                            !reader.acquisition_pending
                                && !reader.retirement_entered
                                && !reader.retirement_pending
                                && reader.model_owners.len() == continuation.inner.models.len()
                                && reader
                                    .model_owners
                                    .iter()
                                    .zip(&continuation.inner.models)
                                    .all(|(actual, expected)| Arc::ptr_eq(actual, expected))
                        })
                })
        } else {
            !original.activating
                && original.parent.as_ref().is_some_and(|parent| {
                    parent.initialized
                        && parent.bank.header.model_generation == actual.model_generation
                })
        };
        if !actual
            .continuation
            .ptr_eq(&Arc::downgrade(&continuation.inner))
            || !Arc::ptr_eq(&actual.issuer, &self.publication_issuer)
            || self
                .model_context_continuation
                .as_ref()
                .is_none_or(|owner| !Arc::ptr_eq(owner, &continuation.inner))
            || self
                .publication
                .as_ref()
                .is_none_or(|storage| !Arc::ptr_eq(storage, &actual.storage))
            || scope.is_none_or(|scope| !Arc::ptr_eq(scope, &actual.scope))
            || self.task_evaluation_identity() != Some(actual.task_identity)
            || self.task_epoch != actual.task_epoch
            || original.refused
            || !activated_basis
        {
            return Err(publication_input_error(
                "restored prefill proof belongs to another native owner or task boundary",
            ));
        }
        Ok(())
    }

    pub fn project_restored_model_context_prefill_boundary(
        &self,
        continuation: &SemanticModelContextContinuation,
        proof: &SemanticRestoredModelContextPrefill,
    ) -> Result<
        (
            SemanticInitialPrefillLease,
            Arc<()>,
            SemanticInitialSourceLayout,
        ),
        SemanticTransitionError,
    > {
        self.require_restored_model_context_prefill(continuation, proof)?;
        let owner = &proof.inner;
        Ok((
            SemanticInitialPrefillLease {
                issuer: Arc::clone(&owner.issuer),
                scope: Arc::clone(&owner.scope),
            },
            Arc::clone(&owner.scope),
            owner.layout.clone(),
        ))
    }

    pub fn restored_model_context_prefill_tensor(
        &mut self,
        continuation: &SemanticModelContextContinuation,
        proof: &SemanticRestoredModelContextPrefill,
        role: SemanticStateRole,
        index: u64,
        consumer_stream: u64,
    ) -> Result<DlpackManagedTensor, SemanticTransitionError> {
        self.require_restored_model_context_prefill(continuation, proof)?;
        dlpack_consumer_stream(consumer_stream)?;
        let tensor = proof
            .inner
            .output_tensors
            .iter()
            .find(|tensor| (tensor.layout.role, tensor.layout.index) == (role as u64, index))
            .ok_or(SemanticTransitionError::ObservationMismatch)?;
        let view = tensor
            .source
            .clone()
            .ok_or(SemanticTransitionError::ObservationMismatch)?;
        let (shape, strides, dtype) = semantic_tensor_export_geometry(&tensor.layout)?;
        let retained: Arc<dyn Send + Sync> = Arc::clone(&proof.inner) as Arc<dyn Send + Sync>;
        self.export_owned_view(
            view,
            shape,
            strides,
            dtype,
            Arc::clone(&proof.inner.scope),
            consumer_stream,
            Some(retained),
        )
    }

    pub fn restored_model_context_prefill_tensor_layouts(
        &self,
        continuation: &SemanticModelContextContinuation,
        proof: &SemanticRestoredModelContextPrefill,
    ) -> Result<Vec<SemanticTensorLayout>, SemanticTransitionError> {
        self.require_restored_model_context_prefill(continuation, proof)?;
        Ok(proof.inner.storage.layouts.values().copied().collect())
    }

    pub fn restored_model_context_prefill_tensor_metadata(
        &self,
        continuation: &SemanticModelContextContinuation,
        proof: &SemanticRestoredModelContextPrefill,
        role: SemanticStateRole,
        index: u64,
    ) -> Result<(SemanticTensorLayout, u64, u64), SemanticTransitionError> {
        self.require_restored_model_context_prefill(continuation, proof)?;
        let tensor = proof
            .inner
            .output_tensors
            .iter()
            .find(|tensor| (tensor.layout.role, tensor.layout.index) == (role as u64, index))
            .ok_or(SemanticTransitionError::ObservationMismatch)?;
        Ok((tensor.layout, tensor.logical_begin, tensor.logical_end))
    }

    pub fn restored_model_context_prefill_tensor_allocation(
        &mut self,
        continuation: &SemanticModelContextContinuation,
        proof: &SemanticRestoredModelContextPrefill,
        role: SemanticStateRole,
        index: u64,
        consumer_stream: u64,
    ) -> Result<(DeviceAllocationProvenance, DlpackManagedTensor), SemanticTransitionError> {
        self.require_restored_model_context_prefill(continuation, proof)?;
        dlpack_consumer_stream(consumer_stream)?;
        let tensor = proof
            .inner
            .output_tensors
            .iter()
            .find(|tensor| (tensor.layout.role, tensor.layout.index) == (role as u64, index))
            .ok_or(SemanticTransitionError::ObservationMismatch)?;
        let view = tensor
            .source
            .as_ref()
            .ok_or(SemanticTransitionError::ObservationMismatch)?;
        let provenance = view
            .allocation_provenance()
            .ok_or(SemanticTransitionError::ObservationMismatch)?;
        let allocation = view
            .allocation_view()
            .ok_or(SemanticTransitionError::ObservationMismatch)?;
        let bytes = i64::try_from(allocation.len())
            .map_err(|_| SemanticTransitionError::ObservationMismatch)?;
        let retained: Arc<dyn Send + Sync> = Arc::clone(&proof.inner) as Arc<dyn Send + Sync>;
        let tensor = self.export_owned_view(
            allocation,
            vec![bytes],
            vec![1],
            (1, 8),
            Arc::clone(&proof.inner.scope),
            consumer_stream,
            Some(retained),
        )?;
        Ok((provenance, tensor))
    }

    pub fn restored_model_context_prefill_record(
        &mut self,
        continuation: &SemanticModelContextContinuation,
        proof: &SemanticRestoredModelContextPrefill,
        role: SemanticStateRole,
        index: u64,
        consumer_stream: u64,
    ) -> Result<DlpackManagedTensor, SemanticTransitionError> {
        self.require_restored_model_context_prefill(continuation, proof)?;
        dlpack_consumer_stream(consumer_stream)?;
        if !matches!(
            role,
            SemanticStateRole::TokenProvenanceRecords | SemanticStateRole::TensorLayout
        ) {
            return Err(publication_input_error(
                "restored prefill record requires its source provenance or native tensor table",
            ));
        }
        let storage = &proof.inner.storage;
        let range = storage.bank_templates[0]
            .iter()
            .find(|range| (range.role, range.index) == (role as u64, index))
            .ok_or(SemanticTransitionError::ObservationMismatch)?;
        let allocation = storage
            .allocations
            .get(
                usize::try_from(range.storage_slot)
                    .map_err(|_| SemanticTransitionError::ObservationMismatch)?,
            )
            .ok_or(SemanticTransitionError::ObservationMismatch)?
            .slice()?;
        let span =
            publication_export_span(range, &publication_record_layout(range), allocation.len())?;
        let view = allocation.view().slice(span);
        let (shape, strides, dtype) =
            semantic_tensor_export_geometry(&publication_record_layout(range))?;
        let retained: Arc<dyn Send + Sync> = Arc::clone(&proof.inner) as Arc<dyn Send + Sync>;
        self.export_owned_view(
            view,
            shape,
            strides,
            dtype,
            Arc::clone(&proof.inner.scope),
            consumer_stream,
            Some(retained),
        )
    }

    pub fn restored_model_context_prefill_native_capacity_tensor_metadata(
        &self,
        continuation: &SemanticModelContextContinuation,
        proof: &SemanticRestoredModelContextPrefill,
        role: SemanticStateRole,
        index: u64,
    ) -> Result<(SemanticTensorLayout, u64, u64), SemanticTransitionError> {
        self.require_restored_model_context_prefill(continuation, proof)?;
        if !matches!(role as u64, 4..=13) {
            return Err(SemanticTransitionError::ObservationMismatch);
        }
        let storage = &proof.inner.storage;
        let range = storage.bank_templates[0]
            .iter()
            .find(|range| (range.role, range.index) == (role as u64, index))
            .ok_or(SemanticTransitionError::ObservationMismatch)?;
        Ok((
            *storage
                .layouts
                .get(&(range.role, range.index))
                .ok_or(SemanticTransitionError::ObservationMismatch)?,
            range.logical_begin,
            range.logical_end,
        ))
    }

    pub fn restored_model_context_prefill_native_capacity_tensor(
        &mut self,
        continuation: &SemanticModelContextContinuation,
        proof: &SemanticRestoredModelContextPrefill,
        role: SemanticStateRole,
        index: u64,
        consumer_stream: u64,
    ) -> Result<DlpackManagedTensor, SemanticTransitionError> {
        self.require_restored_model_context_prefill(continuation, proof)?;
        let (layout, _, _) = self.restored_model_context_prefill_native_capacity_tensor_metadata(
            continuation,
            proof,
            role,
            index,
        )?;
        let storage = &proof.inner.storage;
        let range = storage.bank_templates[0]
            .iter()
            .find(|range| (range.role, range.index) == (role as u64, index))
            .ok_or(SemanticTransitionError::ObservationMismatch)?;
        let allocation = storage
            .allocations
            .get(
                usize::try_from(range.storage_slot)
                    .map_err(|_| SemanticTransitionError::ObservationMismatch)?,
            )
            .ok_or(SemanticTransitionError::ObservationMismatch)?
            .slice()?;
        let span = publication_export_span(range, &layout, allocation.len())?;
        let view = allocation.view().slice(span);
        let (shape, strides, dtype) = semantic_tensor_export_geometry(&layout)?;
        let retained: Arc<dyn Send + Sync> = Arc::clone(&proof.inner) as Arc<dyn Send + Sync>;
        self.export_owned_view(
            view,
            shape,
            strides,
            dtype,
            Arc::clone(&proof.inner.scope),
            consumer_stream,
            Some(retained),
        )
    }
}

pub(super) fn dormant_prefill_source_geometry(
    material: &dormant_material::DormantPublicationMaterial,
) -> Result<(u64, u64, u64, u64, u64), SemanticTransitionError> {
    let native = &material.material;
    let provenance = native
        .ranges
        .iter()
        .find(|item| {
            (item.range.role, item.range.index)
                == (SemanticStateRole::TokenProvenanceRecords as u64, 0)
        })
        .ok_or(SemanticTransitionError::ObservationMismatch)?;
    if !provenance
        .capacity
        .is_multiple_of(size_of::<TokenProvenance>())
    {
        return Err(SemanticTransitionError::ObservationMismatch);
    }
    Ok((
        native.contract.prefix_capacity,
        native.contract.feedback_capacity,
        native.contract.max_position,
        native.contract.pad_token,
        (provenance.capacity / size_of::<TokenProvenance>()) as u64,
    ))
}
