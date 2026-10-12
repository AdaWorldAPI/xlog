//! Original compiler diagnostic custody across the Python error boundary.

use std::fmt;
use std::sync::OnceLock;

use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use pyo3::types::{PyInt, PyModule, PyString, PyTuple};
use xlog_core::XlogError;
use xlog_cuda::{SemanticReplayMaterialBudget, SemanticReplayMaterialReservation};

use crate::semantic_transition::material_bytes::{python_instance_extent, python_integer_extent};
use crate::semantic_transition::material_owner::{
    PySemanticMaterialReadWorkspace, PySemanticMaterialSource,
};
use crate::semantic_transition::python_metadata_tuple_extent;

#[pyclass(
    name = "_OwnedCompilerDiagnostic",
    module = "pyxlog._native",
    frozen,
    immutable_type
)]
struct PyOwnedCompilerDiagnostic {
    diagnostic: OnceLock<XlogError>,
    projection: OnceLock<PyResult<DiagnosticProjection>>,
    budget: SemanticReplayMaterialBudget,
    // Original error and Python object retire before their original receipt.
    _allocation: SemanticReplayMaterialReservation,
}

struct DiagnosticProjection {
    text_bytes: Py<PyInt>,
    workspace_bytes: Py<PyInt>,
    workspace_extent: u64,
}

impl PyOwnedCompilerDiagnostic {
    fn original(&self) -> &XlogError {
        self.diagnostic
            .get()
            .expect("original compiler diagnostic is bound before publication")
    }

    fn projection_extent(py: Python<'_>, text: u64) -> PyResult<u64> {
        let header = u64::try_from(python_instance_extent::<PyString>(py)?).map_err(|_| {
            super::xlog_err(XlogError::allocation_extent_overflow("diagnostic Unicode"))
        })?;
        // CPython's UTF-8 decoder can retain the old Unicode buffer while
        // widening or shrinking it. Both buffers may use four-byte code points.
        // Retain the native formatter and possible UTF-8 cache as well.
        let terminated = text.checked_add(1);
        terminated
            .and_then(|n| n.checked_mul(std::mem::size_of::<u32>() as u64))
            .and_then(|n| n.checked_add(header))
            .and_then(|n| n.checked_mul(2))
            .and_then(|n| n.checked_add(text))
            .and_then(|n| n.checked_add(terminated?))
            .ok_or_else(|| {
                super::xlog_err(XlogError::allocation_extent_overflow(
                    "diagnostic projection",
                ))
            })
    }

    fn projection(&self, py: Python<'_>) -> PyResult<&DiagnosticProjection> {
        self.projection
            .get_or_init(|| {
                let text =
                    xlog_core::error::formatted_text_extent(format_args!("{}", self.original()))
                        .map_err(super::xlog_err)?;
                let workspace = Self::projection_extent(py, text)?;
                // Both scalar allocations were reserved with the original family,
                // before compiler entry; repeated getters only share these objects.
                Ok(DiagnosticProjection {
                    text_bytes: text.into_pyobject(py)?.unbind(),
                    workspace_bytes: workspace.into_pyobject(py)?.unbind(),
                    workspace_extent: workspace,
                })
            })
            .as_ref()
            .map_err(|error| error.clone_ref(py))
    }
}

#[pymethods]
impl PyOwnedCompilerDiagnostic {
    fn __str__(&self) -> PyResult<String> {
        // Derived rendering belongs to the actual consumer workspace acquired
        // before str(error), not another original diagnostic allocation.
        let extent = xlog_core::error::formatted_text_extent(format_args!("{}", self.original()))
            .map_err(super::xlog_err)?;
        let extent = usize::try_from(extent).map_err(|_| {
            super::xlog_err(XlogError::allocation_extent_overflow("diagnostic text"))
        })?;
        let mut rendered = String::with_capacity(extent);
        fmt::write(&mut rendered, format_args!("{}", self.original())).map_err(|_| {
            super::xlog_err(XlogError::allocation_extent_overflow("diagnostic text"))
        })?;
        Ok(rendered)
    }

    fn text_utf8_bytes(&self, py: Python<'_>) -> PyResult<Py<PyInt>> {
        Ok(self.projection(py)?.text_bytes.clone_ref(py))
    }

    fn format_workspace_bytes(&self, py: Python<'_>) -> PyResult<Py<PyInt>> {
        Ok(self.projection(py)?.workspace_bytes.clone_ref(py))
    }

    fn reserve_projection_workspace(
        slf: PyRef<'_, Self>,
        py: Python<'_>,
        material_source: &Bound<'_, PyAny>,
        allocation_bytes: u64,
    ) -> PyResult<Py<PySemanticMaterialReadWorkspace>> {
        if !material_source.is_exact_instance_of::<PySemanticMaterialSource>() {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "diagnostic projection requires its original native material source",
            ));
        }
        if allocation_bytes < slf.projection(py)?.workspace_extent {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "diagnostic projection omits its original formatting workspace",
            ));
        }
        let budget = slf.budget.clone();
        let original_payload = slf.into_pyobject(py)?.into_any().unbind();
        material_source
            .cast::<PySemanticMaterialSource>()?
            .borrow()
            .reserve_diagnostic_projection(py, &budget, allocation_bytes, original_payload)
    }
}

#[pyclass(name = "_OwnedCompilerError", module = "pyxlog._native",
          extends = PyRuntimeError, frozen, immutable_type)]
struct PyOwnedCompilerError {
    // Retain outside mutable BaseException args; cause/traceback remain standard.
    _original_payload: Py<PyOwnedCompilerDiagnostic>,
}

/// Acquired before the compiler's first allocation request, then bound once.
pub(crate) struct PreparedDiagnosticHandoff {
    payload: Py<PyOwnedCompilerDiagnostic>,
    exception: Py<PyOwnedCompilerError>,
}

pub(crate) enum DiagnosticFamilyAcquisitionError {
    Allocation(xlog_cuda::SemanticTransitionError),
    Python(PyErr),
}

impl DiagnosticFamilyAcquisitionError {
    /// Only initial process admission has no original blocking outcome yet.
    pub(crate) fn into_bootstrap_error(self) -> PyErr {
        match self {
            Self::Allocation(error) => super::xlog_err(error),
            Self::Python(error) => error,
        }
    }
}

impl From<PyErr> for DiagnosticFamilyAcquisitionError {
    fn from(error: PyErr) -> Self {
        Self::Python(error)
    }
}

impl PreparedDiagnosticHandoff {
    pub(crate) fn is_unbound_for(
        &self,
        py: Python<'_>,
        budget: &SemanticReplayMaterialBudget,
    ) -> bool {
        let payload = self.payload.borrow(py);
        payload.diagnostic.get().is_none() && payload.budget.same_pool(budget)
    }

    pub(crate) fn bind_failure(self, py: Python<'_>, error: XlogError) -> PyErr {
        assert!(
            self.payload.borrow(py).diagnostic.set(error).is_ok(),
            "original compiler diagnostic family is bound once"
        );
        PyErr::from_value(self.exception.into_bound(py).into_any())
    }
}

pub(crate) fn initialize_owned_diagnostic_types(
    py: Python<'_>,
    module: &Bound<'_, PyModule>,
) -> PyResult<()> {
    module.add_class::<PyOwnedCompilerDiagnostic>()?;
    module.add_class::<PyOwnedCompilerError>()?;
    let _ = pyo3::intern!(py, "args");
    Ok(())
}

pub(crate) fn prepare_owned_diagnostic_handoff(
    py: Python<'_>,
    budget: &SemanticReplayMaterialBudget,
) -> Result<PreparedDiagnosticHandoff, DiagnosticFamilyAcquisitionError> {
    let payload = python_instance_extent::<PyOwnedCompilerDiagnostic>(py)?;
    let exception = python_instance_extent::<PyOwnedCompilerError>(py)?;
    let args = python_metadata_tuple_extent(py, 1)?;
    let integers = python_integer_extent(py)?.checked_mul(2).ok_or_else(|| {
        super::xlog_err(XlogError::allocation_extent_overflow(
            "diagnostic scalar metadata",
        ))
    })?;
    let extent = payload
        .checked_add(exception)
        .and_then(|n| n.checked_add(args))
        .and_then(|n| n.checked_add(integers))
        .and_then(|n| u64::try_from(n).ok())
        .ok_or_else(|| {
            super::xlog_err(XlogError::allocation_extent_overflow("diagnostic family"))
        })?;
    let allocation = budget
        .reserve_auxiliary(extent)
        .map_err(DiagnosticFamilyAcquisitionError::Allocation)?;
    let payload = Py::new(
        py,
        PyOwnedCompilerDiagnostic {
            diagnostic: OnceLock::new(),
            projection: OnceLock::new(),
            budget: budget.clone(),
            _allocation: allocation,
        },
    )?;
    let args = PyTuple::new(py, [payload.clone_ref(py).into_any()])?;
    let exception = Py::new(
        py,
        PyOwnedCompilerError {
            _original_payload: payload.clone_ref(py),
        },
    )?;
    exception
        .bind(py)
        .setattr(pyo3::intern!(py, "args"), args)?;
    Ok(PreparedDiagnosticHandoff { payload, exception })
}
