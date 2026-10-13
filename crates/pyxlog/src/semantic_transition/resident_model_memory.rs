//! One structural resident envelope and its original admitted activations.

use super::*;
use xlog_cuda::{SemanticPreparedResidentModelMemory, SemanticResidentModelMemory};

#[cfg(feature = "semantic-policy")]
use super::learning_phase::cold_model_work::PySemanticColdModelWork;
#[cfg(feature = "semantic-policy")]
use xlog_cuda::{SemanticPreparedResidentModelRestoration, SemanticResidentModelRead};

/// Opaque metadata for one original read; the actual carrier owns its device use.
#[cfg(feature = "semantic-policy")]
#[pyclass(name = "SemanticResidentModelRead", module = "pyxlog._native", frozen)]
pub(crate) struct PySemanticResidentModelRead {
    inner: SemanticResidentModelRead,
    _metadata: xlog_cuda::SemanticReplayMaterialReservation,
}

#[cfg(feature = "semantic-policy")]
impl PySemanticResidentModelRead {
    pub(super) fn new(py: Python<'_>, inner: SemanticResidentModelRead) -> PyResult<Py<Self>> {
        let metadata =
            material_bytes::reserve_python_material::<Self>(py, &inner.material_budget())?;
        Py::new(
            py,
            Self {
                inner,
                _metadata: metadata,
            },
        )
    }

    pub(super) fn inner(&self) -> &SemanticResidentModelRead {
        &self.inner
    }
}

#[cfg(feature = "semantic-policy")]
#[pyclass(
    name = "SemanticPreparedResidentModelRestoration",
    module = "pyxlog._native",
    frozen
)]
pub(crate) struct PySemanticPreparedResidentModelRestoration {
    pub(super) inner: SemanticPreparedResidentModelRestoration,
    restore_read: Py<PySemanticResidentModelRead>,
    snapshot_read: Py<PySemanticResidentModelRead>,
    training_domain: Py<PyAny>,
    objective_law: Py<PyTuple>,
    feedback_intervention: Py<PyTuple>,
    _metadata: xlog_cuda::SemanticReplayMaterialReservation,
}

#[cfg(feature = "semantic-policy")]
impl PySemanticPreparedResidentModelRestoration {
    pub(super) fn new(
        py: Python<'_>,
        inner: SemanticPreparedResidentModelRestoration,
        work: &xlog_cuda::SemanticColdModelWork,
        projection_extent: u64,
        project: impl FnOnce() -> PyResult<(Py<PyAny>, Py<PyTuple>, Py<PyTuple>)>,
    ) -> PyResult<Py<Self>> {
        let original_read = inner.prepare_read(work, 2).map_err(xlog_err)?;
        let budget = original_read.material_budget();
        let class_extent = u64::try_from(material_bytes::python_instance_extent::<Self>(py)?)
            .map_err(|_| invalid("restoration metadata extent exceeds u64"))?;
        let metadata = budget
            .reserve_auxiliary(
                projection_extent
                    .checked_add(class_extent)
                    .ok_or_else(|| invalid("restoration metadata extent overflow"))?,
            )
            .map_err(xlog_err)?;
        let restore_read = PySemanticResidentModelRead::new(py, original_read)?;
        let snapshot_read =
            PySemanticResidentModelRead::new(py, inner.prepare_read(work, 3).map_err(xlog_err)?)?;
        let (training_domain, objective_law, feedback_intervention) = project()?;
        Py::new(
            py,
            Self {
                inner,
                restore_read,
                snapshot_read,
                training_domain,
                objective_law,
                feedback_intervention,
                _metadata: metadata,
            },
        )
    }

    pub(super) fn require_metadata(
        &self,
        py: Python<'_>,
        training_domain: &Bound<'_, PyAny>,
        objective_law: &Bound<'_, PyTuple>,
        feedback_intervention: &Bound<'_, PyTuple>,
    ) -> PyResult<()> {
        if !self.training_domain.bind(py).eq(training_domain)?
            || !self.objective_law.bind(py).eq(objective_law)?
            || !self
                .feedback_intervention
                .bind(py)
                .eq(feedback_intervention)?
        {
            return Err(invalid(
                "restored model metadata changed its original planned task or intervention",
            ));
        }
        Ok(())
    }

    pub(super) fn material_budget(
        &self,
        py: Python<'_>,
    ) -> xlog_cuda::SemanticReplayMaterialBudget {
        self.restore_read.borrow(py).inner().material_budget()
    }
}

#[cfg(feature = "semantic-policy")]
#[pymethods]
impl PySemanticPreparedResidentModelRestoration {
    #[getter]
    fn training_domain(&self, py: Python<'_>) -> Py<PyAny> {
        self.training_domain.clone_ref(py)
    }

    #[getter]
    fn objective_law(&self, py: Python<'_>) -> Py<PyTuple> {
        self.objective_law.clone_ref(py)
    }

    #[getter]
    fn feedback_intervention(&self, py: Python<'_>) -> Py<PyTuple> {
        self.feedback_intervention.clone_ref(py)
    }

    #[getter]
    fn restore_read(&self, py: Python<'_>) -> Py<PySemanticResidentModelRead> {
        self.restore_read.clone_ref(py)
    }

    #[getter]
    fn snapshot_read(&self, py: Python<'_>) -> Py<PySemanticResidentModelRead> {
        self.snapshot_read.clone_ref(py)
    }
}

#[pyclass(
    name = "SemanticPreparedResidentModelMemory",
    module = "pyxlog._native",
    frozen
)]
pub(crate) struct PySemanticPreparedResidentModelMemory {
    parent: Py<PySemanticPublishedParent>,
    inner: SemanticPreparedResidentModelMemory,
    consumer_stream: u64,
    memory: Py<PyTuple>,
    model_contract_metadata: Py<PyTuple>,
}

impl PySemanticPreparedResidentModelMemory {
    #[cfg(feature = "semantic-policy")]
    pub(super) fn prepare_original_restoration(
        &self,
        py: Python<'_>,
        work: &xlog_cuda::SemanticColdModelWork,
    ) -> PyResult<SemanticPreparedResidentModelRestoration> {
        let parent = self.parent.borrow(py);
        let session = parent.session.borrow(py);
        session.require_creator()?;
        let owner = session.owner()?;
        parent
            .task_use
            .borrow(py)
            .require_original_identity(&owner)?;
        let lease = parent.lease()?;
        self.inner
            .prepare_resident_restoration(&owner, &*lease, work)
            .map_err(xlog_err)
    }

    pub(super) fn original_binding(
        &self,
    ) -> (
        &Py<PySemanticPublishedParent>,
        &SemanticPreparedResidentModelMemory,
        u64,
    ) {
        (&self.parent, &self.inner, self.consumer_stream)
    }

    fn require_original(&self, py: Python<'_>) -> PyResult<bool> {
        let parent = self.parent.borrow(py);
        let session = parent.session.borrow(py);
        session.require_creator()?;
        let owner = session.owner()?;
        parent
            .task_use
            .borrow(py)
            .require_original_identity(&owner)?;
        let result = owner
            .resident_model_memory_activation_pending(
                &*parent.lease()?,
                &self.inner,
                self.consumer_stream,
            )
            .map_err(xlog_err);
        result
    }
}

#[pymethods]
impl PySemanticPreparedResidentModelMemory {
    /// Associate one actual numerical occurrence before the sole report admission.
    #[cfg(feature = "semantic-policy")]
    fn prepare_resident_read(
        &self,
        py: Python<'_>,
        work: &PySemanticColdModelWork,
    ) -> PyResult<Py<PySemanticResidentModelRead>> {
        let custody = work.custody(py)?;
        if custody.parent()?.as_ptr() != self.parent.as_ptr() {
            return Err(invalid(
                "resident read changed its original callback parent",
            ));
        }
        let parent = self.parent.borrow(py);
        let session = parent.session.borrow(py);
        session.require_creator()?;
        let owner = session.owner()?;
        parent
            .task_use
            .borrow(py)
            .require_original_identity(&owner)?;
        let ordinal = custody
            .region
            .as_ref()
            .map_or(0, xlog_cuda::SemanticColdModelWorkRegion::ordinal);
        let inner = self
            .inner
            .prepare_resident_read(&owner, &*parent.lease()?, &custody.inner, ordinal)
            .map_err(xlog_err)?;
        PySemanticResidentModelRead::new(py, inner)
    }

    /// The sole original six-field envelope; no capsules are re-exported.
    #[getter]
    fn memory(&self, py: Python<'_>) -> PyResult<Py<PyTuple>> {
        self.require_original(py)?;
        Ok(self.memory.clone_ref(py))
    }

    /// Original selected record bytes and layout, without numerical access.
    #[getter]
    fn model_contract_metadata(&self, py: Python<'_>) -> PyResult<Py<PyTuple>> {
        self.require_original(py)?;
        Ok(self.model_contract_metadata.clone_ref(py))
    }

    /// Guard every genuine numerical occurrence, or join this exact pending one.
    #[pyo3(signature = (*, consumer_stream))]
    fn activate(&self, py: Python<'_>, consumer_stream: &Bound<'_, PyAny>) -> PyResult<()> {
        let stream = ColdValue::read(consumer_stream, &mut 128, 0)?.unsigned()?;
        if stream != self.consumer_stream {
            return Err(invalid(
                "resident model activation changed its original consumer stream",
            ));
        }
        let parent = self.parent.borrow(py);
        let session = parent.session.borrow(py);
        session.require_creator()?;
        #[cfg(not(feature = "semantic-policy"))]
        return Err(invalid(
            "staged resident activation requires its original admitted model recorder",
        ));
        #[cfg(feature = "semantic-policy")]
        {
            // Resolve only already-issued callback custody, before taking either
            // native owner lock. This never lazily adopts or issues another work.
            let task = parent.task_use.borrow(py);
            let original = task.already_issued_cold_model_work(py)?;
            let custody = original.borrow(py).custody(py)?;
            if custody.parent()?.as_ptr() != self.parent.as_ptr() {
                return Err(invalid(
                    "resident activation changed its original callback parent",
                ));
            }
            let reader = custody.reader.borrow(py);
            let source = reader.session.borrow(py);
            source.require_creator()?;
            let mut owner = session.owner()?;
            task.require_original_identity(&owner)?;
            let lease = parent.lease()?;
            owner
                .resident_model_memory_activation_pending(&lease, &self.inner, stream)
                .map_err(xlog_err)?;
            if reader.session.as_ptr() == parent.session.as_ptr() {
                owner
                    .activate_resident_model_memory(
                        &lease,
                        &self.inner,
                        stream,
                        &custody.inner,
                        custody.region.as_ref(),
                    )
                    .map_err(xlog_err)
            } else {
                // The original report cannot leave Recording while its owner
                // lock and borrowed permit are held through this activation.
                let source_owner = source.owner()?;
                let permit = source_owner
                    .cold_model_work_activation_permit(&custody.inner, custody.region.as_ref())
                    .map_err(xlog_err)?;
                owner
                    .activate_resident_model_memory_with_permit(
                        &lease,
                        &self.inner,
                        stream,
                        &permit,
                    )
                    .map_err(xlog_err)
            }
        }
    }
}

#[pymethods]
impl PySemanticPublishedParent {
    /// Prepare genuine allocation aliases, without guards, copies or stream work.
    #[pyo3(signature = (*, consumer_stream))]
    fn prepare_resident_model_memory(
        slf: Py<Self>,
        py: Python<'_>,
        consumer_stream: &Bound<'_, PyAny>,
    ) -> PyResult<Py<PySemanticPreparedResidentModelMemory>> {
        let stream = ColdValue::read(consumer_stream, &mut 128, 0)?.unsigned()?;
        let parent = slf.borrow(py);
        let session = parent.session.borrow(py);
        session.require_creator()?;
        let owner = session.owner()?;
        parent
            .task_use
            .borrow(py)
            .require_original_identity(&owner)?;
        let mut inner = owner
            .prepare_resident_model_memory(&*parent.lease()?, stream)
            .map_err(xlog_err)?;
        drop(owner);
        let memory =
            resident_memory_tuple(py, &parent.session, inner.take_memory().map_err(xlog_err)?)?;
        let (bytes, layout) = inner.model_contract_metadata();
        let model_contract_metadata = (
            PyBytes::new(py, bytes),
            (
                layout.schema_begin,
                layout.schema_bytes,
                layout.schema_digest_offset,
                layout.generation_offset,
                layout.numerical_digest_offset,
                layout.identity_offset,
            ),
        )
            .into_pyobject(py)?
            .unbind();
        Py::new(
            py,
            PySemanticPreparedResidentModelMemory {
                parent: slf.clone_ref(py),
                inner,
                consumer_stream: stream,
                memory,
                model_contract_metadata,
            },
        )
    }
}

/// Shared conversion for staged and ordinary exports of the same native owner.
pub(super) fn resident_memory_tuple(
    py: Python<'_>,
    session: &Py<PySemanticTransitionSession>,
    memory: SemanticResidentModelMemory,
) -> PyResult<Py<PyTuple>> {
    let owner_thread = session.borrow(py).owner_thread;
    let parent = memory.parent;
    let mut allocations = Vec::with_capacity(memory.allocations.len());
    for (provenance, tensor) in memory.allocations {
        let tensor = retain_export_owner(tensor, session.clone_ref(py), owner_thread)?;
        allocations.push(
            (
                Py::new(
                    py,
                    PyNativeTensorAllocation {
                        provenance,
                        policy_layout: None,
                    },
                )?,
                crate::dlpack_capsule_from_tensor(py, tensor)?,
            )
                .into_pyobject(py)?
                .unbind(),
        );
    }
    let storages = PyTuple::new(
        py,
        memory
            .storages
            .into_iter()
            .map(|storage| (storage.allocation, storage.byte_offset, storage.span_bytes)),
    )?;
    let views = PyTuple::new(
        py,
        memory.views.into_iter().map(|(view, layout)| {
            (
                view.role,
                view.index,
                view.storage,
                view.byte_offset,
                (
                    layout.element_bytes,
                    layout.scalar_type,
                    layout.rank,
                    layout.logical_axis,
                    (
                        layout.dimensions[0],
                        layout.dimensions[1],
                        layout.dimensions[2],
                        layout.dimensions[3],
                    ),
                    (
                        layout.strides_bytes[0],
                        layout.strides_bytes[1],
                        layout.strides_bytes[2],
                        layout.strides_bytes[3],
                    ),
                ),
            )
        }),
    )?;
    Ok((
        (
            PyBytes::new(py, parent.instance.as_bytes()),
            parent.word,
            PyBytes::new(py, parent.logical_digest.as_bytes()),
            PyBytes::new(py, parent.state_digest.as_bytes()),
        ),
        memory.generations,
        (
            PyBytes::new(py, memory.numerical_state.0.as_bytes()),
            PyBytes::new(py, memory.numerical_state.1.as_bytes()),
        ),
        PyTuple::new(py, allocations)?,
        storages,
        views,
    )
        .into_pyobject(py)?
        .unbind())
}
