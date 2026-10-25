//! Producer-site recording for complete native model forwards with reused scratch.

use super::*;
use xlog_cuda::{
    SemanticModelCanaryStream, SemanticModelCanaryWitness, SemanticModelForwardStream,
};

#[pyclass(name = "SemanticModelForwardStream", module = "pyxlog._native", frozen)]
pub(crate) struct PySemanticModelForwardStream {
    pub(super) inner: SemanticModelForwardStream,
    source: ModelForwardSource,
}

enum ModelForwardSource {
    Update(Py<PySemanticPreparedStep>),
    Evaluation(Py<model_evaluation::PySemanticModelEvaluation>),
}

#[pyclass(name = "SemanticModelCanaryStream", module = "pyxlog._native", frozen)]
pub(crate) struct PySemanticModelCanaryStream {
    inner: SemanticModelCanaryStream,
    step: Py<PySemanticPreparedStep>,
    baseline: Py<PySemanticModelForwardStream>,
    candidate: Py<PySemanticModelForwardStream>,
}

#[pyclass(name = "SemanticModelCanaryWitness", module = "pyxlog._native", frozen)]
pub(crate) struct PySemanticModelCanaryWitness {
    pub(super) inner: SemanticModelCanaryWitness,
    _session: Py<PySemanticTransitionSession>,
}

fn unsigned(value: &Bound<'_, PyAny>) -> PyResult<u64> {
    ColdValue::read(value, &mut 128, 0)?.unsigned()
}

fn bank(value: &Bound<'_, PyAny>) -> PyResult<usize> {
    match unsigned(value)? {
        0 => Ok(0),
        1 => Ok(1),
        _ => Err(invalid("model stream bank must be zero or one")),
    }
}

fn role(value: &str) -> PyResult<usize> {
    match value {
        "baseline" => Ok(0),
        "candidate" => Ok(1),
        _ => Err(invalid("model stream role must be baseline or candidate")),
    }
}

/// Check the original task and original active capture again around every
/// producer callback. A DLPack/device accessor is external Python execution.
fn original_recording_binding(
    py: Python<'_>,
    step: &PySemanticPreparedStep,
    stream: u64,
) -> PyResult<(Option<String>, Vec<u8>)> {
    let session = step.session.borrow(py);
    session.require_creator()?;
    let owner = session.owner()?;
    step.require_recording_stream(&owner, stream)?;
    step.content_binding_with_owner(py, &owner)
}

fn original_tensor(
    py: Python<'_>,
    step: &PySemanticPreparedStep,
    logits: &Bound<'_, PyAny>,
    layout: SemanticTensorLayout,
    role: usize,
    stream: u64,
    expected: &(Option<String>, Vec<u8>),
) -> PyResult<SemanticTensorInput> {
    let check = || -> PyResult<()> {
        if &original_recording_binding(py, step, stream)? != expected {
            return Err(invalid(
                "model stream authority changed during original producer handoff",
            ));
        }
        Ok(())
    };
    let device = step.session.borrow(py).device_ordinal;
    original_tensor_guarded(logits, layout, role, stream, device, &check)
}

fn original_tensor_guarded(
    logits: &Bound<'_, PyAny>,
    mut layout: SemanticTensorLayout,
    role: usize,
    stream: u64,
    device: usize,
    check: &impl Fn() -> PyResult<()>,
) -> PyResult<SemanticTensorInput> {
    validate_producer_device_guarded(logits, device, check)?;
    layout.role = 0;
    layout.index = role as u64;
    let tensor = crate::dlpack_from_py_for_stream_guarded(
        logits,
        i64::try_from(stream)
            .map_err(|_| invalid("consumer stream exceeds DLPack address space"))?,
        check,
    )?;
    check()?;
    Ok(SemanticTensorInput {
        tensor,
        layout,
        logical_begin: 0,
        logical_end: 0,
        native_allocation: None,
    })
}

#[pymethods]
impl PySemanticPreparedStep {
    #[pyo3(signature = (bank, role, *, vocabulary, scalar_type=5, block_positions=64))]
    fn prepare_model_forward_stream(
        slf: Py<Self>,
        py: Python<'_>,
        bank: &Bound<'_, PyAny>,
        role: &str,
        vocabulary: &Bound<'_, PyAny>,
        scalar_type: u64,
        block_positions: u64,
    ) -> PyResult<PySemanticModelForwardStream> {
        let bank = self::bank(bank)?;
        let role = self::role(role)?;
        let vocabulary = unsigned(vocabulary)?;
        let step = slf.borrow(py);
        let session = step.session.borrow(py);
        session.require_creator()?;
        let mut owner = session.owner()?;
        step.content_binding_with_owner(py, &owner)?;
        let inner = owner
            .prepare_model_forward_stream(
                &step.inner,
                bank,
                role,
                vocabulary,
                scalar_type,
                block_positions,
            )
            .map_err(xlog_err)?;
        Ok(PySemanticModelForwardStream {
            inner,
            source: ModelForwardSource::Update(slf.clone_ref(py)),
        })
    }

    #[pyo3(signature = (bank, *, baseline_stream, candidate_stream))]
    fn prepare_model_canary_stream(
        slf: Py<Self>,
        py: Python<'_>,
        bank: &Bound<'_, PyAny>,
        baseline_stream: Py<PySemanticModelForwardStream>,
        candidate_stream: Py<PySemanticModelForwardStream>,
    ) -> PyResult<PySemanticModelCanaryStream> {
        let bank = self::bank(bank)?;
        let baseline = baseline_stream.borrow(py);
        let candidate = candidate_stream.borrow(py);
        if !matches!(&baseline.source, ModelForwardSource::Update(original) if original.as_ptr() == slf.as_ptr())
            || !matches!(&candidate.source, ModelForwardSource::Update(original) if original.as_ptr() == slf.as_ptr())
        {
            return Err(invalid(
                "paired canaries require both original forwards of this exact prepared step",
            ));
        }
        let step = slf.borrow(py);
        let session = step.session.borrow(py);
        session.require_creator()?;
        let mut owner = session.owner()?;
        step.content_binding_with_owner(py, &owner)?;
        let inner = owner
            .prepare_model_canary_stream(&step.inner, bank, &baseline.inner, &candidate.inner)
            .map_err(xlog_err)?;
        Ok(PySemanticModelCanaryStream {
            inner,
            step: slf.clone_ref(py),
            baseline: baseline_stream.clone_ref(py),
            candidate: candidate_stream.clone_ref(py),
        })
    }
}

impl ModelForwardSource {
    fn session(&self, py: Python<'_>) -> Py<PySemanticTransitionSession> {
        match self {
            Self::Update(step) => step.borrow(py).session.clone_ref(py),
            Self::Evaluation(evaluation) => evaluation
                .borrow(py)
                .parent
                .borrow(py)
                .session
                .clone_ref(py),
        }
    }

    fn recording_binding(
        &self,
        py: Python<'_>,
        inner: &SemanticModelForwardStream,
        stream: u64,
    ) -> PyResult<(Option<String>, Vec<u8>)> {
        match self {
            Self::Update(step) => original_recording_binding(py, &step.borrow(py), stream),
            Self::Evaluation(evaluation) => {
                let evaluation = evaluation.borrow(py);
                let parent = evaluation.parent.borrow(py);
                let session = parent.session.borrow(py);
                session.require_creator()?;
                let owner = session.owner()?;
                evaluation.check(py, &owner)?;
                owner
                    .check_evaluation_model_forward_stream(
                        &*parent.lease()?,
                        &evaluation.inner,
                        inner,
                        stream,
                    )
                    .map_err(xlog_err)?;
                Ok(evaluation.binding.clone())
            }
        }
    }
}

impl PySemanticModelForwardStream {
    pub(super) fn from_evaluation(
        inner: SemanticModelForwardStream,
        evaluation: Py<model_evaluation::PySemanticModelEvaluation>,
    ) -> Self {
        Self {
            inner,
            source: ModelForwardSource::Evaluation(evaluation),
        }
    }
}

#[pymethods]
impl PySemanticModelForwardStream {
    /// Original issued source owner, not a reconstructed Python handle.
    #[getter]
    fn source_owner(&self, py: Python<'_>) -> Py<PyAny> {
        match &self.source {
            ModelForwardSource::Update(step) => step.clone_ref(py).into_any(),
            ModelForwardSource::Evaluation(evaluation) => evaluation.clone_ref(py).into_any(),
        }
    }

    #[getter]
    fn geometry(&self) -> (u64, u64, u64, u64, u64) {
        self.inner.geometry()
    }

    #[getter]
    fn role(&self) -> &'static str {
        match self.inner.role() {
            0 => "baseline",
            1 => "candidate",
            2 => "evaluation",
            _ => unreachable!("native model forward role"),
        }
    }

    #[getter]
    fn bank(&self) -> Option<usize> {
        self.inner.bank()
    }

    #[pyo3(signature = (*, consumer_stream))]
    fn begin(&self, py: Python<'_>, consumer_stream: &Bound<'_, PyAny>) -> PyResult<()> {
        let stream = parse_witness_consumer_stream(consumer_stream, &mut 128)?;
        self.source.recording_binding(py, &self.inner, stream)?;
        let session_ref = self.source.session(py);
        let session = session_ref.borrow(py);
        let mut owner = session.owner()?;
        match &self.source {
            ModelForwardSource::Update(step) => {
                owner.begin_model_forward_stream(&step.borrow(py).inner, &self.inner, stream)
            }
            ModelForwardSource::Evaluation(evaluation) => {
                let evaluation = evaluation.borrow(py);
                let parent = evaluation.parent.borrow(py);
                let lease = parent.lease()?;
                owner.begin_evaluation_model_forward_stream(
                    &lease,
                    &evaluation.inner,
                    &self.inner,
                    stream,
                )
            }
        }
        .map_err(xlog_err)
    }

    #[pyo3(signature = (*, row, position_begin, count, logits, logits_layout, consumer_stream))]
    fn record_block(
        &self,
        py: Python<'_>,
        row: &Bound<'_, PyAny>,
        position_begin: &Bound<'_, PyAny>,
        count: &Bound<'_, PyAny>,
        logits: &Bound<'_, PyAny>,
        logits_layout: &Bound<'_, PyAny>,
        consumer_stream: &Bound<'_, PyAny>,
    ) -> PyResult<()> {
        let row = unsigned(row)?;
        let position_begin = unsigned(position_begin)?;
        let count = unsigned(count)?;
        let mut budget = 16 * 1024;
        let stream = parse_witness_consumer_stream(consumer_stream, &mut budget)?;
        let layout = parse_tensor_layout(&ColdValue::read(logits_layout, &mut budget, 0)?)?;
        let expected = self.source.recording_binding(py, &self.inner, stream)?;
        let check = || -> PyResult<()> {
            if self.source.recording_binding(py, &self.inner, stream)? != expected {
                return Err(invalid(
                    "model forward authority changed during native block handoff",
                ));
            }
            Ok(())
        };
        let session_ref = self.source.session(py);
        let device = session_ref.borrow(py).device_ordinal;
        let tensor =
            original_tensor_guarded(logits, layout, self.inner.role(), stream, device, &check)?;
        check()?;
        let session = session_ref.borrow(py);
        let mut owner = session.owner()?;
        match &self.source {
            ModelForwardSource::Update(step) => owner.record_model_forward_stream_block(
                &step.borrow(py).inner,
                &self.inner,
                row,
                position_begin,
                count,
                tensor,
                stream,
            ),
            ModelForwardSource::Evaluation(evaluation) => {
                let evaluation = evaluation.borrow(py);
                let parent = evaluation.parent.borrow(py);
                let lease = parent.lease()?;
                owner.record_evaluation_model_forward_stream_block(
                    &lease,
                    &evaluation.inner,
                    &self.inner,
                    row,
                    position_begin,
                    count,
                    tensor,
                    stream,
                )
            }
        }
        .map_err(xlog_err)
    }

    #[pyo3(signature = (*, allocation_owner=None, consumer_stream))]
    fn finish(
        &self,
        py: Python<'_>,
        allocation_owner: Option<&PySemanticTensorOwner>,
        consumer_stream: &Bound<'_, PyAny>,
    ) -> PyResult<PySemanticModelForwardWitness> {
        let stream = parse_witness_consumer_stream(consumer_stream, &mut 128)?;
        self.source.recording_binding(py, &self.inner, stream)?;
        let session_ref = self.source.session(py);
        let session = session_ref.borrow(py);
        let mut owner = session.owner()?;
        let inner = match &self.source {
            ModelForwardSource::Update(step) => owner
                .finish_model_forward_stream(
                    &step.borrow(py).inner,
                    &self.inner,
                    allocation_owner.map(|owner| &owner.inner),
                    stream,
                )
                .map_err(xlog_err)?,
            ModelForwardSource::Evaluation(evaluation) => {
                if allocation_owner.is_some() {
                    return Err(invalid(
                        "evaluation forward finish does not accept an allocation owner",
                    ));
                }
                let evaluation = evaluation.borrow(py);
                let parent = evaluation.parent.borrow(py);
                let lease = parent.lease()?;
                owner
                    .finish_evaluation_model_forward_stream(
                        &lease,
                        &evaluation.inner,
                        &self.inner,
                        stream,
                    )
                    .map_err(xlog_err)?
            }
        };
        Ok(PySemanticModelForwardWitness {
            inner,
            _session: session_ref.clone_ref(py),
        })
    }
}

#[pymethods]
impl PySemanticModelCanaryStream {
    #[getter]
    fn source_owner(&self, py: Python<'_>) -> Py<PySemanticPreparedStep> {
        self.step.clone_ref(py)
    }

    #[getter]
    fn baseline_stream(&self, py: Python<'_>) -> Py<PySemanticModelForwardStream> {
        self.baseline.clone_ref(py)
    }

    #[getter]
    fn candidate_stream(&self, py: Python<'_>) -> Py<PySemanticModelForwardStream> {
        self.candidate.clone_ref(py)
    }

    #[getter]
    fn canary_rows(&self) -> (u64, u64, u64, u64, u64) {
        let [a, b, c, d, e] = self.inner.canary_rows();
        (a, b, c, d, e)
    }

    #[pyo3(signature = (*, consumer_stream))]
    fn begin(&self, py: Python<'_>, consumer_stream: &Bound<'_, PyAny>) -> PyResult<()> {
        let stream = parse_witness_consumer_stream(consumer_stream, &mut 128)?;
        let step = self.step.borrow(py);
        original_recording_binding(py, &step, stream)?;
        let session = step.session.borrow(py);
        let mut owner = session.owner()?;
        owner
            .begin_model_canary_stream(&step.inner, &self.inner, stream)
            .map_err(xlog_err)
    }

    #[pyo3(signature = (*, canary_index, position_begin, count, baseline_logits, candidate_logits, logits_layout, consumer_stream))]
    fn record_block(
        &self,
        py: Python<'_>,
        canary_index: &Bound<'_, PyAny>,
        position_begin: &Bound<'_, PyAny>,
        count: &Bound<'_, PyAny>,
        baseline_logits: &Bound<'_, PyAny>,
        candidate_logits: &Bound<'_, PyAny>,
        logits_layout: &Bound<'_, PyAny>,
        consumer_stream: &Bound<'_, PyAny>,
    ) -> PyResult<()> {
        let canary_index = unsigned(canary_index)?;
        let position_begin = unsigned(position_begin)?;
        let count = unsigned(count)?;
        let mut budget = 16 * 1024;
        let stream = parse_witness_consumer_stream(consumer_stream, &mut budget)?;
        let layout = parse_tensor_layout(&ColdValue::read(logits_layout, &mut budget, 0)?)?;
        let step = self.step.borrow(py);
        let expected = original_recording_binding(py, &step, stream)?;
        let baseline = original_tensor(py, &step, baseline_logits, layout, 0, stream, &expected)?;
        let candidate = original_tensor(py, &step, candidate_logits, layout, 1, stream, &expected)?;
        if original_recording_binding(py, &step, stream)? != expected {
            return Err(invalid(
                "paired canary authority changed before native block recording",
            ));
        }
        let session = step.session.borrow(py);
        let mut owner = session.owner()?;
        owner
            .record_model_canary_stream_block(
                &step.inner,
                &self.inner,
                canary_index,
                position_begin,
                count,
                baseline,
                candidate,
                stream,
            )
            .map_err(xlog_err)
    }

    #[pyo3(signature = (*, consumer_stream))]
    fn finish(
        &self,
        py: Python<'_>,
        consumer_stream: &Bound<'_, PyAny>,
    ) -> PyResult<PySemanticModelCanaryWitness> {
        let stream = parse_witness_consumer_stream(consumer_stream, &mut 128)?;
        let step = self.step.borrow(py);
        original_recording_binding(py, &step, stream)?;
        let session = step.session.borrow(py);
        let inner = session
            .owner()?
            .finish_model_canary_stream(&step.inner, &self.inner, stream)
            .map_err(xlog_err)?;
        Ok(PySemanticModelCanaryWitness {
            inner,
            _session: step.session.clone_ref(py),
        })
    }
}
