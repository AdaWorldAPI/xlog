use super::*;

/// Actual model and native components of their original producer enclosure.
/// Physical peak and report-observation overhead are measured separately.
#[derive(Clone, Copy, Debug)]
pub struct SemanticColdModelWorkResult {
    pub model_work: u64,
    pub operation_count: u64,
    pub work_bound: u64,
    pub model_calls: u64,
    pub native_work: u64,
    pub native_events: [u64; 9],
}

impl SemanticColdModelWorkResult {
    pub(super) fn from_native_report(words: &[u64]) -> Result<Self, SemanticTransitionError> {
        if words.len() != 15
            || words[0] != 0
            || words[1] > words[3]
            || words[6..15]
                .iter()
                .try_fold(0u64, |sum, units| sum.checked_add(*units))
                != Some(words[5])
            || words[1].checked_add(words[5]).is_none()
        {
            return Err(publication_input_error(
                "original native work report is incomplete",
            ));
        }
        Ok(Self {
            model_work: words[1],
            operation_count: words[2],
            work_bound: words[3],
            model_calls: words[4],
            native_work: words[5],
            native_events: words[6..15]
                .try_into()
                .expect("checked native event extent"),
        })
    }

    pub(super) fn include(&mut self, other: Self) -> Result<(), SemanticTransitionError> {
        let add = |left: u64, right: u64| {
            left.checked_add(right)
                .ok_or(SemanticTransitionError::GenerationExhausted)
        };
        self.model_work = add(self.model_work, other.model_work)?;
        self.operation_count = add(self.operation_count, other.operation_count)?;
        self.work_bound = add(self.work_bound, other.work_bound)?;
        self.model_calls = add(self.model_calls, other.model_calls)?;
        self.native_work = add(self.native_work, other.native_work)?;
        for (total, actual) in self.native_events.iter_mut().zip(other.native_events) {
            *total = add(*total, actual)?;
        }
        Ok(())
    }
}

pub(super) fn add_native_work_ceiling(
    destination: &mut [u64; 9],
    addition: [u64; 9],
) -> Result<(), SemanticTransitionError> {
    let mut combined = *destination;
    for (total, value) in combined.iter_mut().zip(addition) {
        *total = total
            .checked_add(value)
            .ok_or_else(native_work_ceiling_overflow)?;
    }
    *destination = combined;
    Ok(())
}

pub(super) fn native_work_ceiling_units(ceiling: [u64; 9]) -> Result<u64, SemanticTransitionError> {
    ceiling.into_iter().try_fold(0u64, |total, value| {
        total
            .checked_add(value)
            .ok_or_else(native_work_ceiling_overflow)
    })
}

pub(super) fn native_work_ceiling_overflow() -> SemanticTransitionError {
    publication_input_error("native producer work ceiling overflowed")
}

pub(super) fn ceiling_sum(
    values: impl IntoIterator<Item = u64>,
) -> Result<u64, SemanticTransitionError> {
    values.into_iter().try_fold(0u64, |sum, value| {
        sum.checked_add(value)
            .ok_or_else(native_work_ceiling_overflow)
    })
}

pub(super) fn ceiling_product(a: u64, b: u64) -> Result<u64, SemanticTransitionError> {
    a.checked_mul(b).ok_or_else(native_work_ceiling_overflow)
}

pub(super) fn publication_hash_ceiling(bytes: u64) -> Result<[u64; 9], SemanticTransitionError> {
    if bytes > u64::MAX / 8 {
        return Err(native_work_ceiling_overflow());
    }
    let blocks = ceiling_sum([bytes, 9 + 63])? / 64;
    Ok([0, 0, 0, 0, 0, 0, 0, blocks, 32])
}

pub(super) fn selected_model_origin_native_work_ceiling(
    storage: &PublicationStorage,
    directory: &[PublicationRange],
) -> Result<[u64; 9], SemanticTransitionError> {
    selected_model_origin_geometry_native_work_ceiling(&storage.contract_value, directory)
}

pub(super) fn selected_model_origin_geometry_native_work_ceiling(
    contract: &PublicationContract,
    directory: &[PublicationRange],
) -> Result<[u64; 9], SemanticTransitionError> {
    if !directory
        .iter()
        .any(|range| range.role == 44 && range.index == 0)
    {
        return Err(SemanticTransitionError::ObservationMismatch);
    }
    let mut result = [
        0,
        ceiling_sum([directory.len() as u64, 4 + 72])?,
        0,
        0,
        0,
        0,
        0,
        0,
        112,
    ];
    add_native_work_ceiling(
        &mut result,
        publication_hash_ceiling(contract.model_contract_layout.schema_bytes)?,
    )?;
    add_native_work_ceiling(
        &mut result,
        publication_hash_ceiling(b"xlog.semantic.model-identity.v2\0".len() as u64 + 32 + 8 + 32)?,
    )?;
    native_work_ceiling_units(result)?;
    Ok(result)
}

pub(super) struct OriginalNativeWorkAccounting {
    pub(super) work: Option<DeviceMemoryView<u64>>,
    pub(super) initialized: bool,
    native_ceiling: [u64; 9],
    dma_ceiling: [u64; 9],
    dma_actual: [u64; 9],
}

impl OriginalNativeWorkAccounting {
    pub(super) fn new(native_ceiling: [u64; 9], dma_ceiling: [u64; 9]) -> Self {
        Self {
            work: None,
            initialized: false,
            native_ceiling,
            dma_ceiling,
            dma_actual: [0; 9],
        }
    }

    pub(super) fn record_dma(
        &mut self,
        bytes: usize,
        recorded: &mut bool,
    ) -> Result<(), SemanticTransitionError> {
        record_submitted_native_dma(&mut self.dma_actual, self.dma_ceiling, bytes, recorded)
    }

    pub(super) fn complete(
        &self,
        mut result: SemanticColdModelWorkResult,
    ) -> Result<SemanticColdModelWorkResult, SemanticTransitionError> {
        if result.model_work != 0
            || result.operation_count != 0
            || result.work_bound != 0
            || result.model_calls != 0
            || result
                .native_events
                .iter()
                .zip(self.native_ceiling)
                .any(|(actual, upper)| *actual > upper)
            || self
                .dma_actual
                .iter()
                .zip(self.dma_ceiling)
                .any(|(actual, upper)| *actual > upper)
        {
            return Err(publication_input_error(
                "training materialization exceeded its original native producer bound",
            ));
        }
        add_native_work_ceiling(&mut result.native_events, self.dma_actual)?;
        result.native_work = native_work_ceiling_units(result.native_events)?;
        Ok(result)
    }
}

pub(super) fn record_submitted_native_dma(
    actual: &mut [u64; 9],
    ceiling: [u64; 9],
    bytes: usize,
    recorded: &mut bool,
) -> Result<(), SemanticTransitionError> {
    if *recorded {
        return Ok(());
    }
    add_native_work_ceiling(
        actual,
        [
            1,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            u64::try_from(bytes).map_err(|_| native_work_ceiling_overflow())?,
        ],
    )?;
    *recorded = true;
    if actual
        .iter()
        .zip(ceiling)
        .any(|(actual, upper)| *actual > upper)
    {
        return Err(publication_input_error(
            "submitted native DMA exceeded its original finite entitlement",
        ));
    }
    Ok(())
}

pub(super) fn run_native_work_report(
    provider: &CudaKernelProvider,
    domain: &ResidentExecutionDomain,
    command: &mut OriginalNativeCommand,
    poisoned: &mut bool,
    input: ModelWorkInput,
    native: &DeviceMemoryView<u64>,
    report: &DeviceMemoryView<u64>,
    mut recorder: LaunchRecorder,
) -> Result<(), SemanticTransitionError> {
    let kernel = provider
        .device()
        .inner()
        .get_func(
            "xlog_semantic_transition",
            "semantic_cold_model_work_result",
        )
        .ok_or_else(|| runtime_error("kernel lookup", "cold model work result unavailable"))?;
    recorder.read(native);
    recorder.write(report);
    let arguments = (
        input.events,
        input.count,
        input.bound,
        *native.device_ptr(),
        *report.device_ptr(),
    );
    let mut failure = None;
    let result = command.run(domain, poisoned, recorder, |enqueue, entered, submitted| {
        let (events, count, bound, native, report) = arguments;
        let events = events.into_kernel_param_storage();
        let count = count.into_kernel_param_storage();
        let bound = bound.into_kernel_param_storage();
        let native = native.into_kernel_param_storage();
        let report = report.into_kernel_param_storage();
        let mut parameters = [
            events.as_kernel_param(),
            count.as_kernel_param(),
            bound.as_kernel_param(),
            native.as_kernel_param(),
            report.as_kernel_param(),
        ];
        // SAFETY: the original report owner retains every argument allocation.
        unsafe {
            kernel.launch_raw_in_original(
                enqueue,
                LaunchConfig {
                    grid_dim: (1, 1, 1),
                    block_dim: (1, 1, 1),
                    shared_mem_bytes: 0,
                },
                &mut parameters,
                false,
                entered,
                submitted,
            )
        }
        .map_err(|error| {
            failure = Some(error.0);
            XlogError::Kernel(error.to_string())
        })
    });
    if let Some(status) = failure {
        command.submission_failure = Some(status);
        return Err(native_runtime_error(
            "original native work report",
            cudarc::driver::DriverError(status),
        ));
    }
    result
}
