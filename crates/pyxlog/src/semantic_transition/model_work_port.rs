//! One original numerical counter port and its genuine late tensor export.

use super::*;
use xlog_cuda::{SemanticModelWorkBufferExport, SemanticModelWorkBufferPort};

enum OriginalCounterOwner {
    Step(Py<PySemanticPreparedStep>),
    #[cfg(feature = "semantic-policy")]
    Cold(Py<learning_phase::cold_model_work::PySemanticColdModelWork>),
    #[cfg(feature = "semantic-policy")]
    Evaluation(Py<model_evaluation::PySemanticModelEvaluation>),
}

struct OriginalCounterExport {
    provenance: DeviceAllocationProvenance,
    layout: SemanticTensorLayout,
    tensor: Option<DlpackManagedTensor>,
    capsule: Option<Py<PyAny>>,
    allocation: Option<Py<PyNativeTensorAllocation>>,
    device: usize,
    stream: u64,
    session: Py<PySemanticTransitionSession>,
    result: Option<Py<PyTuple>>,
}

/// Carries an original numerical owner, never an invented tensor extent.
/// The caller retains the same port through plan freezing and late binding.
#[pyclass(
    name = "SemanticModelWorkBufferPort",
    module = "pyxlog._native",
    frozen
)]
pub(crate) struct PySemanticModelWorkBufferPort {
    owner: OriginalCounterOwner,
    inner: SemanticModelWorkBufferPort,
    exported: Mutex<Option<OriginalCounterExport>>,
}

impl PySemanticModelWorkBufferPort {
    pub(in crate::semantic_transition) fn new_step(
        step: Py<PySemanticPreparedStep>,
        inner: SemanticModelWorkBufferPort,
    ) -> Self {
        Self {
            owner: OriginalCounterOwner::Step(step),
            inner,
            exported: Mutex::new(None),
        }
    }

    #[cfg(feature = "semantic-policy")]
    pub(in crate::semantic_transition) fn new_cold(
        work: Py<learning_phase::cold_model_work::PySemanticColdModelWork>,
        inner: SemanticModelWorkBufferPort,
    ) -> Self {
        Self {
            owner: OriginalCounterOwner::Cold(work),
            inner,
            exported: Mutex::new(None),
        }
    }

    #[cfg(feature = "semantic-policy")]
    pub(in crate::semantic_transition) fn new_evaluation(
        work: Py<model_evaluation::PySemanticModelEvaluation>,
        inner: SemanticModelWorkBufferPort,
    ) -> Self {
        Self {
            owner: OriginalCounterOwner::Evaluation(work),
            inner,
            exported: Mutex::new(None),
        }
    }

    fn native_export(
        &self,
        py: Python<'_>,
    ) -> PyResult<(
        SemanticModelWorkBufferExport,
        Py<PySemanticTransitionSession>,
    )> {
        match &self.owner {
            OriginalCounterOwner::Step(original) => {
                let step = original.borrow(py);
                let session = step.session.borrow(py);
                session.require_creator()?;
                let mut owner = session.owner()?;
                step.content_binding_with_owner(py, &owner)?;
                let exported = owner
                    .export_prepared_model_work_buffer_port(&self.inner)
                    .map_err(xlog_err)?;
                Ok((exported, step.session.clone_ref(py)))
            }
            #[cfg(feature = "semantic-policy")]
            OriginalCounterOwner::Cold(original) => original
                .borrow(py)
                .export_model_work_buffer_port(py, &self.inner),
            #[cfg(feature = "semantic-policy")]
            OriginalCounterOwner::Evaluation(original) => original
                .borrow(py)
                .export_model_work_buffer_port(py, &self.inner),
        }
    }
}

#[pymethods]
impl PySemanticModelWorkBufferPort {
    /// Compare genuine original counter ownership, not Python tensor shape.
    fn same_counter(&self, other: &Self) -> bool {
        self.inner.same_counter(&other.inner)
    }

    #[getter]
    fn original_owner(&self, py: Python<'_>) -> Py<PyAny> {
        match &self.owner {
            OriginalCounterOwner::Step(owner) => owner.clone_ref(py).into_any(),
            #[cfg(feature = "semantic-policy")]
            OriginalCounterOwner::Cold(owner) => owner.clone_ref(py).into_any(),
            #[cfg(feature = "semantic-policy")]
            OriginalCounterOwner::Evaluation(owner) => owner.clone_ref(py).into_any(),
        }
    }

    #[getter]
    fn device_ordinal(&self) -> usize {
        self.inner.device_ordinal()
    }

    #[getter]
    fn consumer_stream(&self) -> u64 {
        self.inner.consumer_stream()
    }

    #[getter]
    fn dtype(&self) -> (u8, u8) {
        self.inner.dtype()
    }

    #[getter]
    fn row_width(&self) -> usize {
        self.inner.row_width()
    }

    /// Exact original extent after known initialization; never allocates or resumes.
    #[getter]
    fn finalized_extent(&self, py: Python<'_>) -> PyResult<usize> {
        match &self.owner {
            OriginalCounterOwner::Step(original) => {
                let step = original.borrow(py);
                let session = step.session.borrow(py);
                session.require_creator()?;
                let owner = session.owner()?;
                step.content_binding_with_owner(py, &owner)?;
                owner
                    .prepared_model_work_buffer_port_finalized_extent(&self.inner)
                    .map_err(xlog_err)
            }
            #[cfg(feature = "semantic-policy")]
            OriginalCounterOwner::Cold(original) => original
                .borrow(py)
                .finalized_model_work_extent(py, &self.inner),
            #[cfg(feature = "semantic-policy")]
            OriginalCounterOwner::Evaluation(original) => original
                .borrow(py)
                .finalized_model_work_extent(py, &self.inner),
        }
    }

    /// Return the same native allocation/layout and capsule after finalization.
    /// No numerical reader is granted, and a consumed capsule is not reissued.
    fn export(&self, py: Python<'_>) -> PyResult<Py<PyTuple>> {
        let mut stored = self
            .exported
            .lock()
            .map_err(|_| invalid("original counter export custody is poisoned"))?;
        if stored.is_none() {
            let (exported, session_owner) = self.native_export(py)?;
            let (provenance, layout, tensor, device, stream) = exported.into_parts();
            let session = session_owner.borrow(py);
            let tensor =
                retain_export_owner(tensor, session_owner.clone_ref(py), session.owner_thread)?;
            drop(session);
            // Retain the native pair before any fallible Python conversion.
            *stored = Some(OriginalCounterExport {
                provenance,
                layout,
                tensor: Some(tensor),
                capsule: None,
                allocation: None,
                device,
                stream,
                session: session_owner,
                result: None,
            });
        }
        let original = stored.as_mut().expect("retained original counter export");
        original.session.borrow(py).require_creator()?;
        if let Some(result) = &original.result {
            return Ok(result.clone_ref(py));
        }
        if original.capsule.is_none() {
            original.capsule = Some(crate::dlpack_capsule_from_retained_tensor(
                py,
                &mut original.tensor,
            )?);
        }
        if original.allocation.is_none() {
            original.allocation = Some(Py::new(
                py,
                PyNativeTensorAllocation {
                    provenance: original.provenance.clone(),
                    policy_layout: Some(original.layout),
                },
            )?);
        }
        let layout = original.layout;
        let shape = PyTuple::new(py, [layout.dimensions[0], layout.dimensions[1]])?;
        let strides = PyTuple::new(
            py,
            [layout.strides_bytes[0] / 8, layout.strides_bytes[1] / 8],
        )?;
        let result = (
            original
                .allocation
                .as_ref()
                .expect("original allocation projection")
                .clone_ref(py),
            shape,
            strides,
            (1u8, 64u8),
            original.device,
            original.stream,
            original
                .capsule
                .as_ref()
                .expect("original capsule")
                .clone_ref(py),
        )
            .into_pyobject(py)?
            .unbind();
        original.result = Some(result.clone_ref(py));
        Ok(result)
    }
}
