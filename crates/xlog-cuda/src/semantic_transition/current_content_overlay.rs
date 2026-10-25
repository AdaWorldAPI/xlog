//! One backing for mutually exclusive CURRENT child content producers.

use super::super::source_content_occurrences::SemanticSourceTensorDescriptor;
use super::*;

/// Private custody rooted in one original decoded instruction. It survives
/// retirement of the cold report without a global allocation or actor lookup.
pub(super) struct ContentOverlayInstruction {
    groups: Mutex<Vec<Arc<CurrentContentOverlay>>>,
    binding: Mutex<()>,
}

impl ContentOverlayInstruction {
    pub(super) fn metadata_bytes(count: usize) -> Result<usize, SemanticTransitionError> {
        count
            .checked_mul(2 * size_of::<Arc<CurrentContentOverlay>>())
            .and_then(|bytes| bytes.checked_add(size_of::<Self>() + 2 * size_of::<usize>()))
            .ok_or(SemanticTransitionError::GenerationExhausted)
    }

    pub(super) fn new(count: usize) -> Result<Arc<Self>, SemanticTransitionError> {
        let count = count
            .checked_mul(2)
            .ok_or(SemanticTransitionError::GenerationExhausted)?;
        Ok(Arc::new(Self {
            groups: Mutex::new(Vec::with_capacity(count)),
            binding: Mutex::new(()),
        }))
    }
}

pub(super) struct CurrentContentOverlay {
    scope: Arc<()>,
    kind: SemanticTransitionKind,
    owners: [std::sync::Weak<SourceContentGeometryOwner>; 2],
    layouts: [Vec<(u64, u64, u64)>; 4],
    placements: [Vec<(u64, u64, u64)>; 4],
    span_bytes: u64,
    alignment: u64,
    state: Mutex<ContentOverlayState>,
    _reservation: Arc<SemanticReplayMaterialReservation>,
}

struct ContentOverlayState {
    backing: Option<ContentOverlayBacking>,
    binding_entered: bool,
    bound: bool,
    actor: Option<std::sync::Weak<actor_refresh::PreparedActorRefreshOwner>>,
}

struct ContentOverlayBacking {
    tensor: PreparedSemanticTensor,
    consumer_stream: u64,
    device: usize,
    _retirement: ModelExportOwner,
}

fn overlay_index(kind: SemanticTransitionKind) -> Result<usize, SemanticTransitionError> {
    match kind {
        SemanticTransitionKind::Recompute => Ok(0),
        SemanticTransitionKind::Proposal => Ok(1),
        _ => Err(publication_input_error(
            "content overlay requires its original Recompute or Proposal operation",
        )),
    }
}

impl SemanticPreparedSourceContentGeometry {
    /// Declare the original allocation alternatives for both SOURCE owners.
    /// Outer Update banks and each child's conditional banks provide the
    /// alternatives; the caller supplies no exclusion claim.
    pub fn content_overlay(
        &self,
        other: &Self,
        prepared_kind: SemanticTransitionKind,
        allocation_layouts: [Vec<(u64, u64, u64)>; 4],
    ) -> Result<
        (SemanticSourceGeometryAllocation, [Vec<(u64, u64, u64)>; 4]),
        SemanticTransitionError,
    > {
        let index = overlay_index(prepared_kind)?;
        self.require_content_overlay_pair(other)?;
        let mut first =
            self.owner.overlays.lock().map_err(|_| {
                publication_input_error("original content overlay custody is poisoned")
            })?;
        let mut second =
            other.owner.overlays.lock().map_err(|_| {
                publication_input_error("original content overlay custody is poisoned")
            })?;
        if let Some(original) = &first[index] {
            if second[index]
                .as_ref()
                .is_none_or(|value| !Arc::ptr_eq(original, value))
                || original.layouts != allocation_layouts
            {
                return Err(publication_input_error(
                    "content overlay cannot replace its original complete allocation rosters",
                ));
            }
            return Ok((original.allocation(index), original.placements.clone()));
        }
        if second[index].is_some() {
            return Err(SemanticTransitionError::ObservationMismatch);
        }
        for owner in [&self.owner, &other.owner] {
            if owner
                .current
                .lock()
                .map_err(|_| {
                    publication_input_error("original actor geometry custody is poisoned")
                })?
                .is_some()
                || owner
                    .reader
                    .lock()
                    .map_err(|_| {
                        publication_input_error("original content reader custody is poisoned")
                    })?
                    .is_some()
            {
                return Err(publication_input_error(
                    "content overlay must precede the original content freeze and ActorRefresh binding",
                ));
            }
        }
        let count = allocation_layouts.iter().try_fold(0usize, |count, rows| {
            count
                .checked_add(rows.len())
                .ok_or(SemanticTransitionError::GenerationExhausted)
        })?;
        let bytes = count
            .checked_mul(3 * size_of::<(u64, u64, u64)>())
            .and_then(|bytes| {
                bytes.checked_add(
                    size_of::<CurrentContentOverlay>()
                        + size_of::<SemanticReplayMaterialReservation>()
                        + 6 * size_of::<usize>(),
                )
            })
            .ok_or(SemanticTransitionError::GenerationExhausted)?;
        let reservation = Arc::new(self.material_budget().reserve_auxiliary(
            u64::try_from(bytes).map_err(|_| SemanticTransitionError::GenerationExhausted)?,
        )?);
        let mut placements: [Vec<(u64, u64, u64)>; 4] = std::array::from_fn(|alternative| {
            Vec::with_capacity(allocation_layouts[alternative].len())
        });
        let mut span_bytes = 0u64;
        let mut alignment = 1u64;
        for (rows, placed) in allocation_layouts.iter().zip(&mut placements) {
            let mut end = 0u64;
            for (position, &(slot, span, align)) in rows.iter().enumerate() {
                if align == 0
                    || !align.is_power_of_two()
                    || rows[..position].iter().any(|row| row.0 == slot)
                {
                    return Err(publication_input_error(
                        "content overlay needs unique original slots and power-of-two byte alignment",
                    ));
                }
                let offset = end
                    .checked_add(align - 1)
                    .map(|value| value & !(align - 1))
                    .ok_or(SemanticTransitionError::GenerationExhausted)?;
                end = offset
                    .checked_add(span)
                    .ok_or(SemanticTransitionError::GenerationExhausted)?;
                placed.push((slot, offset, span));
                alignment = alignment.max(align);
            }
            span_bytes = span_bytes.max(end);
        }
        if span_bytes == 0 || span_bytes > isize::MAX as u64 {
            return Err(publication_input_error(
                "content overlay requires one nonempty addressable complete backing",
            ));
        }
        let owner = Arc::new(CurrentContentOverlay {
            scope: Arc::new(()),
            kind: prepared_kind,
            owners: [Arc::downgrade(&self.owner), Arc::downgrade(&other.owner)],
            layouts: allocation_layouts,
            placements,
            span_bytes,
            alignment,
            state: Mutex::new(ContentOverlayState {
                backing: None,
                binding_entered: false,
                bound: false,
                actor: None,
            }),
            _reservation: reservation,
        });
        let result = (owner.allocation(index), owner.placements.clone());
        let mut groups = self.owner.overlay_instruction.groups.lock().map_err(|_| {
            publication_input_error("original instruction overlay roster custody is poisoned")
        })?;
        if groups.len() == groups.capacity() {
            return Err(SemanticTransitionError::ObservationMismatch);
        }
        // The original instruction retains every physical claim through its
        // last geometry. Groups retain only weak geometry/actor links, so this
        // cannot form a cycle or permit reuse after a single pair retires.
        groups.push(Arc::clone(&owner));
        first[index] = Some(Arc::clone(&owner));
        second[index] = Some(owner);
        Ok(result)
    }

    fn require_content_overlay_pair(&self, other: &Self) -> Result<(), SemanticTransitionError> {
        if !self.owner.kind.is_current()
            || self.owner.kind != other.owner.kind
            || self.owner.bank != Some(0)
            || other.owner.bank != Some(1)
            || self.owner.step_index != other.owner.step_index
            || self.owner.ordinal != other.owner.ordinal
            || !self.owner.work.same_invocation(&other.owner.work)
            || !Arc::ptr_eq(&self.owner.instruction, &other.owner.instruction)
            || !Arc::ptr_eq(&self.owner.allowance, &other.owner.allowance)
            || !Arc::ptr_eq(&self.owner.seed, &other.owner.seed)
            || !Arc::ptr_eq(&self.owner.publication, &other.owner.publication)
            || !Arc::ptr_eq(
                &self.owner.overlay_instruction,
                &other.owner.overlay_instruction,
            )
            || self
                .owner
                .input_generation
                .as_ref()
                .zip(other.owner.input_generation.as_ref())
                .is_none_or(|(left, right)| !left.same_handle(right))
        {
            return Err(publication_input_error(
                "content overlay requires both genuine CURRENT banks of one original Update and member",
            ));
        }
        Ok(())
    }

    /// Authenticate the immutable producer stream before a foreign callback.
    /// The canonical DLPack decoder preserves legacy stream 1 as CUDA stream 0.
    pub fn require_content_overlay_backing_stream(
        &self,
        session: &SemanticTransitionSession,
        instruction_admission: &SemanticSegmentInstructionAdmission,
        consumer_stream: u64,
    ) -> Result<(), SemanticTransitionError> {
        let instruction = session.instruction_admission(instruction_admission)?;
        if !Arc::ptr_eq(&self.owner.instruction, &instruction.instruction)
            || session
                .graph
                .cold_work_allowance()
                .as_ref()
                .is_none_or(|allowance| !Arc::ptr_eq(allowance, &self.owner.allowance))
        {
            return Err(publication_input_error(
                "content overlay backing changed its original instruction",
            ));
        }
        let stream = dlpack_consumer_stream(consumer_stream)?;
        if stream != session.stream.cu_stream()
            || stream != session.provider.device().inner().stream().cu_stream()
        {
            return Err(publication_input_error(
                "content overlay producer must use its original native import and capture stream",
            ));
        }
        Ok(())
    }

    /// Import the caller's original contiguous UInt8 pool block. The managed
    /// producer remains foreign and is retained by both original geometries.
    pub fn bind_content_overlay(
        &self,
        other: &Self,
        session: &SemanticTransitionSession,
        instruction_admission: &SemanticSegmentInstructionAdmission,
        prepared_kind: SemanticTransitionKind,
        backing: DlpackManagedTensor,
        consumer_stream: u64,
    ) -> Result<(), SemanticTransitionError> {
        // Retain even an invalid handoff until the canonical stream retirement
        // proves that dropping its real managed producer is safe.
        let mut pending = RetainedTensorAdmission {
            inputs: Some(vec![backing]),
            stream: Arc::clone(&session.stream),
            consumer_stream: Some(consumer_stream),
            ready: false,
        };
        let index = overlay_index(prepared_kind)?;
        self.require_content_overlay_pair(other)?;
        self.require_content_overlay_backing_stream(
            session,
            instruction_admission,
            consumer_stream,
        )?;
        if session.prepared_segment.as_ref().is_some_and(|build| {
            build.capturing || build.submitted || build.finished || build.graph_call_entered
        }) {
            return Err(publication_input_error(
                "content overlay backing changed its original instruction or entered after recording",
            ));
        }
        let owner = self.owner.content_overlay_owner(index)?;
        let other_owner = other.owner.content_overlay_owner(index)?;
        if !Arc::ptr_eq(&owner, &other_owner) {
            return Err(SemanticTransitionError::ObservationMismatch);
        }
        let _binding = self.owner.overlay_instruction.binding.lock().map_err(|_| {
            publication_input_error("original instruction overlay binding custody is poisoned")
        })?;
        let mut state = owner.state.lock().map_err(|_| {
            publication_input_error("original content overlay backing custody is poisoned")
        })?;
        if state.binding_entered {
            return Err(publication_input_error(
                "content overlay binds once for its declared allocation alternatives",
            ));
        }
        state.binding_entered = true;
        let event =
            session.stream.context().new_event(None).map_err(|error| {
                runtime_error("content overlay consumer retirement event", error)
            })?;
        let layout = SemanticTensorLayout {
            role: 0,
            index: 0,
            scalar_type: 1,
            element_bytes: 1,
            rank: 1,
            dimensions: [owner.span_bytes, 0, 0, 0],
            logical_axis: u64::MAX,
            strides_bytes: [1, 0, 0, 0],
        };
        let backing = pending
            .inputs
            .as_mut()
            .expect("retained overlay producer")
            .pop()
            .ok_or(SemanticTransitionError::ObservationMismatch)?;
        // The canonical importer takes this exact token before its first
        // metadata check; its stream was authenticated above on both sides.
        pending.ready = true;
        let mut tensors = prepare_semantic_tensors(
            &session.provider,
            vec![SemanticTensorInput {
                layout,
                logical_begin: 0,
                logical_end: 0,
                tensor: backing,
                native_allocation: None,
            }],
        )?;
        let tensor = tensors
            .pop()
            .ok_or(SemanticTransitionError::ObservationMismatch)?;
        let view = tensor
            .source
            .clone()
            .ok_or(SemanticTransitionError::ObservationMismatch)?;
        // Retain before validating addresses: a rejected import must keep its
        // original foreign token through stream completion, just like success.
        let retained_owner: Arc<dyn Send + Sync> = owner._reservation.clone();
        state.backing = Some(ContentOverlayBacking {
            tensor,
            consumer_stream,
            device: session.provider.device().ordinal(),
            // Reuse the canonical foreign-view consumer retirement owner.
            // Its original event and managed view survive a failed/unknown
            // completion; this grants no native allocation provenance.
            _retirement: ModelExportOwner(Some(ModelExportUse {
                _generation: None,
                _view: view,
                _publication: None,
                _guard: Some(Arc::clone(&owner.scope)),
                event,
                stream: Arc::clone(&session.stream),
                consumer_stream,
                _retained_owner: Some(retained_owner),
            })),
        });
        let backing = state
            .backing
            .as_ref()
            .expect("retained original overlay backing");
        if !backing.tensor.data.is_multiple_of(owner.alignment) {
            return Err(publication_input_error(
                "content overlay backing does not satisfy its original full allocation alignment",
            ));
        }
        cuda_backing_allocation(backing.tensor.data, owner.span_bytes as usize)?;
        let groups = self.owner.overlay_instruction.groups.lock().map_err(|_| {
            publication_input_error("original instruction overlay roster custody is poisoned")
        })?;
        for other in groups
            .iter()
            .cloned()
            .filter(|other| !Arc::ptr_eq(other, &owner))
        {
            let other_state = other.state.lock().map_err(|_| {
                publication_input_error("original content overlay backing custody is poisoned")
            })?;
            if let Some(other_backing) = &other_state.backing {
                if step_input_overlap(
                    backing.tensor.data,
                    owner.span_bytes as usize,
                    other_backing.tensor.data,
                    other.span_bytes as usize,
                )? {
                    return Err(publication_input_error("content overlay backing is already retained by another operation, member or Update"));
                }
            }
        }
        for original in session
            .publication
            .iter()
            .flat_map(|storage| &storage.allocations)
        {
            if step_input_overlap(
                backing.tensor.data,
                owner.span_bytes as usize,
                original.entry().pointer,
                original.entry().bytes as usize,
            )? {
                return Err(publication_input_error(
                    "content overlay backing aliases original published or model storage",
                ));
            }
        }
        state.bound = true;
        Ok(())
    }
}

impl CurrentContentOverlay {
    fn allocation(&self, index: usize) -> SemanticSourceGeometryAllocation {
        SemanticSourceGeometryAllocation::declared(
            &self.scope,
            SourceGeometryAllocationPort::ContentOverlay(index),
            None,
            self.span_bytes,
        )
    }

    fn placement(
        &self,
        alternative: usize,
        tensor: &SemanticSourceTensorDescriptor,
    ) -> Result<(u64, u64, u64), SemanticTransitionError> {
        let bytes = u64::try_from(tensor_layout_bytes(&tensor.layout)?)
            .map_err(|_| SemanticTransitionError::GenerationExhausted)?;
        self.placements[alternative]
            .iter()
            .copied()
            .find(|&(_, offset, span)| {
                tensor.byte_offset >= offset
                    && tensor
                        .byte_offset
                        .checked_add(bytes)
                        .zip(offset.checked_add(span))
                        .is_some_and(|(end, limit)| end <= limit)
                    && (bytes != 0 || span == 0 || tensor.byte_offset < offset + span)
            })
            .ok_or_else(|| {
                publication_input_error(
                    "content overlay tensor exceeds its original allocation placement",
                )
            })
    }
}

impl SourceContentGeometryOwner {
    pub(super) fn bind_content_overlay_actor_refresh(
        &self,
        proof: &Arc<actor_refresh::PreparedActorRefreshOwner>,
    ) -> Result<(), SemanticTransitionError> {
        let overlays = self
            .overlays
            .lock()
            .map_err(|_| publication_input_error("original content overlay custody is poisoned"))?
            .clone();
        for owner in overlays.iter().flatten() {
            for (bank, geometry) in proof.content_geometry.iter().enumerate() {
                if owner.owners[bank]
                    .upgrade()
                    .is_none_or(|original| !Arc::ptr_eq(&original, &geometry.owner))
                    || proof.kind != self.kind
                {
                    return Err(publication_input_error(
                        "content overlay changed its authentic original ActorRefresh guard owner",
                    ));
                }
            }
            let mut state = owner.state.lock().map_err(|_| {
                publication_input_error("original content overlay actor custody is poisoned")
            })?;
            if state.actor.as_ref().is_some_and(|original| {
                original
                    .upgrade()
                    .is_none_or(|original| !Arc::ptr_eq(&original, proof))
            }) {
                return Err(publication_input_error(
                    "content overlay cannot reuse a different original ActorRefresh",
                ));
            }
            state.actor = Some(Arc::downgrade(proof));
        }
        Ok(())
    }

    pub(in crate::semantic_transition) fn require_content_overlay_stream(
        &self,
        section: (SemanticTransitionKind, usize),
        consumer_stream: u64,
    ) -> Result<(), SemanticTransitionError> {
        let index = overlay_index(section.0)?;
        let overlays = self
            .overlays
            .lock()
            .map_err(|_| publication_input_error("original content overlay custody is poisoned"))?
            .clone();
        let Some(owner) = &overlays[index] else {
            return Ok(());
        };
        let state = owner.state.lock().map_err(|_| {
            publication_input_error("original content overlay backing custody is poisoned")
        })?;
        if section.1 > 1
            || !state.bound
            || state.backing.as_ref().is_none_or(|backing| {
                dlpack_consumer_stream(backing.consumer_stream)
                    .map_or(true, |stream| stream as u64 != consumer_stream)
            })
        {
            return Err(publication_input_error(
                "CURRENT content changed its original overlay consumer stream",
            ));
        }
        Ok(())
    }

    fn content_overlay_owner(
        &self,
        index: usize,
    ) -> Result<Arc<CurrentContentOverlay>, SemanticTransitionError> {
        self.overlays
            .lock()
            .map_err(|_| publication_input_error("original content overlay custody is poisoned"))?
            .get(index)
            .and_then(Clone::clone)
            .ok_or(SemanticTransitionError::NotBound)
    }

    pub(super) fn require_content_overlay_allocation(
        &self,
        index: usize,
        allocation: &SemanticSourceGeometryAllocation,
        span: u64,
    ) -> Result<(), SemanticTransitionError> {
        let owner = self.content_overlay_owner(index)?;
        let bank = self
            .bank
            .ok_or(SemanticTransitionError::ObservationMismatch)?;
        if !owner.allocation(index).same_handle(allocation)
            || span != owner.span_bytes
            || allocation.span_bytes != span
            || bank > 1
            || owner.owners[bank]
                .upgrade()
                .is_none_or(|original| !std::ptr::eq(self, Arc::as_ptr(&original)))
        {
            return Err(publication_input_error(
                "content overlay changed its original allocation owner or full span",
            ));
        }
        Ok(())
    }

    pub(super) fn require_content_overlay_section(
        &self,
        index: usize,
        section: Option<(SemanticTransitionKind, usize)>,
    ) -> Result<(), SemanticTransitionError> {
        let owner = self.content_overlay_owner(index)?;
        if !self.kind.is_current()
            || section.is_none_or(|(kind, bank)| kind != owner.kind || bank > 1)
        {
            return Err(publication_input_error(
                "content overlay belongs to another original prepared operation",
            ));
        }
        Ok(())
    }

    pub(super) fn classify_content_overlay_tensor(
        &self,
        allocation: &SemanticSourceGeometryAllocation,
        descriptor: &SemanticSourceTensorDescriptor,
        section: Option<(SemanticTransitionKind, usize)>,
    ) -> Result<(bool, bool), SemanticTransitionError> {
        self.require_allocation_section(allocation, section)?;
        let SourceGeometryAllocationPort::ContentOverlay(index) = allocation.port else {
            return Err(SemanticTransitionError::ObservationMismatch);
        };
        let owner = self.content_overlay_owner(index)?;
        let alternative = 2 * self
            .bank
            .ok_or(SemanticTransitionError::ObservationMismatch)?
            + section
                .ok_or(SemanticTransitionError::ObservationMismatch)?
                .1;
        owner.placement(alternative, descriptor)?;
        Ok((false, false))
    }

    pub(super) fn require_prepared_content_overlay(
        &self,
        session: &SemanticTransitionSession,
        step: &SemanticPreparedStep,
        section: (SemanticTransitionKind, usize),
        allocation: &SemanticSourceGeometryAllocation,
        descriptor: &SemanticSourceTensorDescriptor,
        tensor: &PreparedSemanticTensor,
    ) -> Result<(), SemanticTransitionError> {
        let SourceGeometryAllocationPort::ContentOverlay(index) = allocation.port else {
            return Err(SemanticTransitionError::ObservationMismatch);
        };
        let owner = self.content_overlay_owner(index)?;
        let state = owner.state.lock().map_err(|_| {
            publication_input_error("original content overlay backing custody is poisoned")
        })?;
        let backing = state
            .backing
            .as_ref()
            .ok_or(SemanticTransitionError::NotBound)?;
        if !state.bound
            || session
                .prepared_segment
                .as_ref()
                .is_none_or(|build| build.submitted)
            || backing.device != session.provider.device().ordinal()
            || tensor.native_allocation.is_some()
            || backing.tensor.data.checked_add(descriptor.byte_offset) != Some(tensor.data)
            || session.actor_refresh.as_ref().is_none_or(|staged| {
                state
                    .actor
                    .as_ref()
                    .and_then(std::sync::Weak::upgrade)
                    .is_none_or(|actor| !Arc::ptr_eq(&actor, &staged.proof.inner))
            })
        {
            return Err(publication_input_error("CURRENT content changed its original foreign overlay backing or typed view address"));
        }
        dlpack_consumer_stream(backing.consumer_stream)?;
        let staged = session
            .actor_refresh
            .as_ref()
            .ok_or(SemanticTransitionError::NotBound)?;
        let inputs = session
            .steps
            .get(&step.token)
            .and_then(|step| step.inputs.as_ref())
            .ok_or(SemanticTransitionError::NotBound)?;
        // Use the canonical complete input-owner check, including private
        // headers, ranges, bindings, digest buffers and backing padding.
        if staged.proof.inner.inputs.owns_tensor(&backing.tensor)?
            || inputs.owns_tensor(&backing.tensor)?
        {
            return Err(publication_input_error(
                "content overlay backing aliases its original native input owner",
            ));
        }
        for original in staged
            .proof
            .inner
            .target
            .allocations
            .iter()
            .chain(&staged.proof.inner.inputs.storage.allocations)
            .chain(
                session
                    .publication
                    .iter()
                    .flat_map(|storage| &storage.allocations),
            )
            .chain(&inputs.storage.allocations)
        {
            if step_input_overlap(
                backing.tensor.data,
                owner.span_bytes as usize,
                original.entry().pointer,
                original.entry().bytes as usize,
            )? {
                return Err(publication_input_error("content overlay backing aliases an original published, model or input allocation"));
            }
        }
        for view in staged
            .proof
            .inner
            .inputs
            .views
            .iter()
            .chain(&inputs.views)
            .flat_map(|views| views.values())
        {
            if step_input_overlap(
                backing.tensor.data,
                owner.span_bytes as usize,
                *view.device_ptr(),
                view.len(),
            )? {
                return Err(publication_input_error(
                    "content overlay backing aliases an original borrowed input root",
                ));
            }
        }
        Ok(())
    }
}
