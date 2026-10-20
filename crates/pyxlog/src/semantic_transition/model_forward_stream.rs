//! Producer-site recording for complete native model forwards with reused scratch.

use super::*;
use xlog_cuda::{
    SemanticModelCanaryStream, SemanticModelCanaryWitness, SemanticModelForwardStream,
};

#[pyclass(name = "SemanticModelForwardStream", module = "pyxlog", frozen)]
pub(crate) struct PySemanticModelForwardStream {
    inner: SemanticModelForwardStream,
    step: Py<PySemanticPreparedStep>,
}

#[pyclass(name = "SemanticModelCanaryStream", module = "pyxlog", frozen)]
pub(crate) struct PySemanticModelCanaryStream {
    inner: SemanticModelCanaryStream,
    step: Py<PySemanticPreparedStep>,
}

#[pyclass(name = "SemanticModelCanaryWitness", module = "pyxlog", frozen)]
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
    mut layout: SemanticTensorLayout,
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
    validate_producer_device_guarded(logits, device, &check)?;
    layout.role = 0;
    layout.index = role as u64;
    let tensor = crate::dlpack_from_py_for_stream_guarded(
        logits,
        i64::try_from(stream)
            .map_err(|_| invalid("consumer stream exceeds DLPack address space"))?,
        &check,
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
            step: slf.clone_ref(py),
        })
    }

    #[pyo3(signature = (bank, *, baseline_stream, candidate_stream))]
    fn prepare_model_canary_stream(
        slf: Py<Self>,
        py: Python<'_>,
        bank: &Bound<'_, PyAny>,
        baseline_stream: &PySemanticModelForwardStream,
        candidate_stream: &PySemanticModelForwardStream,
    ) -> PyResult<PySemanticModelCanaryStream> {
        let bank = self::bank(bank)?;
        if baseline_stream.step.as_ptr() != slf.as_ptr()
            || candidate_stream.step.as_ptr() != slf.as_ptr()
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
            .prepare_model_canary_stream(
                &step.inner,
                bank,
                &baseline_stream.inner,
                &candidate_stream.inner,
            )
            .map_err(xlog_err)?;
        Ok(PySemanticModelCanaryStream {
            inner,
            step: slf.clone_ref(py),
        })
    }
}

#[pymethods]
impl PySemanticModelForwardStream {
    #[pyo3(signature = (*, consumer_stream))]
    fn begin(&self, py: Python<'_>, consumer_stream: &Bound<'_, PyAny>) -> PyResult<()> {
        let stream = parse_witness_consumer_stream(consumer_stream, &mut 128)?;
        let step = self.step.borrow(py);
        original_recording_binding(py, &step, stream)?;
        let session = step.session.borrow(py);
        session
            .owner()?
            .begin_model_forward_stream(&step.inner, &self.inner, stream)
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
        let step = self.step.borrow(py);
        let expected = original_recording_binding(py, &step, stream)?;
        let role = self.inner.role();
        let tensor = original_tensor(py, &step, logits, layout, role, stream, &expected)?;
        if original_recording_binding(py, &step, stream)? != expected {
            return Err(invalid(
                "model forward authority changed before native block recording",
            ));
        }
        let session = step.session.borrow(py);
        session
            .owner()?
            .record_model_forward_stream_block(
                &step.inner,
                &self.inner,
                row,
                position_begin,
                count,
                tensor,
                stream,
            )
            .map_err(xlog_err)
    }

    #[pyo3(signature = (*, allocation_witness=None, consumer_stream))]
    fn finish(
        &self,
        py: Python<'_>,
        allocation_witness: Option<&PySemanticTensorContentWitness>,
        consumer_stream: &Bound<'_, PyAny>,
    ) -> PyResult<PySemanticModelForwardWitness> {
        let stream = parse_witness_consumer_stream(consumer_stream, &mut 128)?;
        let step = self.step.borrow(py);
        original_recording_binding(py, &step, stream)?;
        let session = step.session.borrow(py);
        let inner = session
            .owner()?
            .finish_model_forward_stream(
                &step.inner,
                &self.inner,
                allocation_witness.map(|witness| &witness.inner),
                stream,
            )
            .map_err(xlog_err)?;
        Ok(PySemanticModelForwardWitness {
            inner,
            _session: step.session.clone_ref(py),
        })
    }
}

#[pymethods]
impl PySemanticModelCanaryStream {
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
        session
            .owner()?
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
        session
            .owner()?
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
