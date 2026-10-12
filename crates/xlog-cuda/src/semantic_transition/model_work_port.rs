//! Original counter ownership before its count-derived allocation exists.

use super::*;

#[derive(Clone)]
enum ModelWorkBufferOwner {
    Step(SemanticPreparedStep),
    #[cfg(feature = "semantic-policy")]
    Cold(SemanticColdModelWork, Option<SemanticColdModelWorkRegion>),
    #[cfg(feature = "semantic-policy")]
    Evaluation(SemanticModelEvaluation),
}

/// An original counter port, not a tensor, allocation or recording permission.
/// Its event extent comes only from the original complete plan. Hot banks share
/// their step's counter; cold regions share their enclosing report's counter.
#[derive(Clone)]
pub struct SemanticModelWorkBufferPort {
    owner: ModelWorkBufferOwner,
    device: usize,
    stream: u64,
}

impl SemanticModelWorkBufferPort {
    pub(super) fn for_step(step: SemanticPreparedStep, device: usize, stream: u64) -> Self {
        Self {
            owner: ModelWorkBufferOwner::Step(step),
            device,
            stream,
        }
    }

    #[cfg(feature = "semantic-policy")]
    pub(super) fn for_cold(
        work: SemanticColdModelWork,
        region: Option<SemanticColdModelWorkRegion>,
        device: usize,
        stream: u64,
    ) -> Self {
        Self {
            owner: ModelWorkBufferOwner::Cold(work, region),
            device,
            stream,
        }
    }

    #[cfg(feature = "semantic-policy")]
    pub(super) fn for_evaluation(
        work: SemanticModelEvaluation,
        device: usize,
        stream: u64,
    ) -> Self {
        Self {
            owner: ModelWorkBufferOwner::Evaluation(work),
            device,
            stream,
        }
    }

    pub fn same_counter(&self, other: &Self) -> bool {
        self.device == other.device
            && self.stream == other.stream
            && match (&self.owner, &other.owner) {
                (ModelWorkBufferOwner::Step(left), ModelWorkBufferOwner::Step(right)) => {
                    left.same_handle(right)
                }
                #[cfg(feature = "semantic-policy")]
                (ModelWorkBufferOwner::Cold(left, _), ModelWorkBufferOwner::Cold(right, _)) => {
                    left.same_invocation(right)
                }
                #[cfg(feature = "semantic-policy")]
                (
                    ModelWorkBufferOwner::Evaluation(left),
                    ModelWorkBufferOwner::Evaluation(right),
                ) => left.same_invocation(right),
                #[cfg(feature = "semantic-policy")]
                _ => false,
            }
    }

    pub fn device_ordinal(&self) -> usize {
        self.device
    }
    pub fn consumer_stream(&self) -> u64 {
        self.stream
    }
    pub fn dtype(&self) -> (u8, u8) {
        (1, 64)
    }
    pub fn row_width(&self) -> usize {
        3
    }

    #[cfg(feature = "semantic-policy")]
    pub(super) fn cold_binding(
        &self,
    ) -> Result<
        (
            &SemanticColdModelWork,
            Option<&SemanticColdModelWorkRegion>,
            u64,
            usize,
        ),
        SemanticTransitionError,
    > {
        match &self.owner {
            ModelWorkBufferOwner::Cold(work, region) => {
                Ok((work, region.as_ref(), self.stream, self.device))
            }
            _ => Err(publication_input_error(
                "counter port belongs to another original numerical owner",
            )),
        }
    }

    #[cfg(feature = "semantic-policy")]
    pub(super) fn evaluation_binding(
        &self,
    ) -> Result<(&SemanticModelEvaluation, u64, usize), SemanticTransitionError> {
        match &self.owner {
            ModelWorkBufferOwner::Evaluation(work) => Ok((work, self.stream, self.device)),
            _ => Err(publication_input_error(
                "counter port belongs to another original numerical owner",
            )),
        }
    }
}

/// A genuine late export paired by native code with that allocation's layout.
/// It is never constructed from a caller's tensor or an unmaterialized extent.
pub struct SemanticModelWorkBufferExport {
    allocation: DeviceAllocationProvenance,
    layout: SemanticTensorLayout,
    tensor: DlpackManagedTensor,
    device: usize,
    stream: u64,
}

impl SemanticModelWorkBufferExport {
    pub(super) fn new(
        allocation: DeviceAllocationProvenance,
        layout: SemanticTensorLayout,
        tensor: DlpackManagedTensor,
        device: usize,
        stream: u64,
    ) -> Result<Self, SemanticTransitionError> {
        let bytes = layout.dimensions[0]
            .checked_mul(3)
            .and_then(|cells| cells.checked_mul(8))
            .ok_or(SemanticTransitionError::GenerationExhausted)?;
        if layout.rank != 2
            || layout.scalar_type != 3
            || layout.element_bytes != 8
            || layout.dimensions[0] == 0
            || layout.dimensions[1] != 3
            || layout.strides_bytes[..2] != [24, 8]
            || allocation.view_bytes() != bytes
            || allocation.byte_offset() != 0
        {
            return Err(publication_input_error(
                "late counter export differs from its original materialized row layout",
            ));
        }
        Ok(Self {
            allocation,
            layout,
            tensor,
            device,
            stream,
        })
    }

    pub fn into_parts(
        self,
    ) -> (
        DeviceAllocationProvenance,
        SemanticTensorLayout,
        DlpackManagedTensor,
        usize,
        u64,
    ) {
        (
            self.allocation,
            self.layout,
            self.tensor,
            self.device,
            self.stream,
        )
    }
}

impl SemanticTransitionSession {
    /// Inspect the genuine structural step without allocating event scratch or
    /// exporting a capsule. The caller retains this same owner across refusal.
    pub fn prepared_model_work_buffer_port(
        &self,
        step: &SemanticPreparedStep,
        consumer_stream: u64,
    ) -> Result<SemanticModelWorkBufferPort, SemanticTransitionError> {
        self.check_prepared_cold(step)?;
        if consumer_stream != self.stream.cu_stream() as u64 {
            return Err(publication_input_error(
                "counter port changed its original recording stream",
            ));
        }
        Ok(SemanticModelWorkBufferPort::for_step(
            step.clone(),
            self.provider.device().ordinal(),
            consumer_stream,
        ))
    }

    /// Export only the allocation belonging to this original finalized port.
    pub fn prepared_model_work_buffer_port_finalized_extent(
        &self,
        port: &SemanticModelWorkBufferPort,
    ) -> Result<usize, SemanticTransitionError> {
        let ModelWorkBufferOwner::Step(step) = &port.owner else {
            return Err(publication_input_error(
                "counter port does not belong to an original prepared step",
            ));
        };
        let layout = self.prepared_step_layout(step)?;
        if port.device != self.provider.device().ordinal()
            || port.stream != self.stream.cu_stream() as u64
            || self.prepared_segment.as_ref().is_none_or(|build| {
                !build.recording_ready.load(Ordering::Acquire)
                    || build.initialization_pending.load(Ordering::Acquire)
            })
        {
            return Err(publication_input_error(
                "counter extent requires its original known initialized plan and stream",
            ));
        }
        Ok(layout.model_work_capacity)
    }

    /// Export only the allocation belonging to this original finalized port.
    pub fn export_prepared_model_work_buffer_port(
        &mut self,
        port: &SemanticModelWorkBufferPort,
    ) -> Result<SemanticModelWorkBufferExport, SemanticTransitionError> {
        let ModelWorkBufferOwner::Step(step) = &port.owner else {
            return Err(publication_input_error(
                "counter port does not belong to an original prepared step",
            ));
        };
        self.check_prepared_cold(step)?;
        if port.device != self.provider.device().ordinal()
            || port.stream != self.stream.cu_stream() as u64
        {
            return Err(publication_input_error(
                "late counter export changed its original device or stream",
            ));
        }
        let work = self.steps[&step.token]
            .prepared
            .as_ref()
            .expect("prepared owner")
            .model_work
            .as_ref()
            .ok_or_else(|| {
                publication_input_error("counter port requires its original finalized layout")
            })?;
        let rows = u64::try_from(work.actual.len() / 3)
            .map_err(|_| SemanticTransitionError::GenerationExhausted)?;
        let allocation = work
            .actual
            .view()
            .allocation_provenance()
            .ok_or(SemanticTransitionError::ObservationMismatch)?;
        let layout = canonical_tensor_layout(SemanticTensorLayout {
            role: 0,
            index: 0,
            element_bytes: 8,
            scalar_type: 3,
            rank: 2,
            logical_axis: u64::MAX,
            dimensions: [rows, 3, 0, 0],
            strides_bytes: [0; 4],
        })?;
        let tensor = self.model_work_buffer(step, port.stream)?;
        SemanticModelWorkBufferExport::new(allocation, layout, tensor, port.device, port.stream)
    }
}
