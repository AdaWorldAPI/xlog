//! The original process material source, admitted before file or task effects.

use super::*;

#[pyclass(
    name = "SemanticMaterialReadWorkspace",
    module = "pyxlog._native",
    frozen
)]
pub(crate) struct PySemanticMaterialReadWorkspace {
    _owner: OriginalMaterialReadOwner,
    _allocation: xlog_cuda::SemanticReplayMaterialReservation,
}

enum OriginalMaterialReadOwner {
    Material {
        _material: xlog_cuda::SemanticReplayLoadedMaterial,
    },
    Source {
        _source: Arc<material_source::MaterialSourceRegistry>,
    },
}

pub(super) fn prepare_material_read_workspace(
    py: Python<'_>,
    material: xlog_cuda::SemanticReplayLoadedMaterial,
    budget: &xlog_cuda::SemanticReplayMaterialBudget,
    allocation_bytes: u64,
) -> PyResult<Py<PySemanticMaterialReadWorkspace>> {
    prepare_read_workspace(
        py,
        OriginalMaterialReadOwner::Material {
            _material: material,
        },
        budget,
        allocation_bytes,
    )
}

fn prepare_read_workspace(
    py: Python<'_>,
    owner: OriginalMaterialReadOwner,
    budget: &xlog_cuda::SemanticReplayMaterialBudget,
    allocation_bytes: u64,
) -> PyResult<Py<PySemanticMaterialReadWorkspace>> {
    if allocation_bytes == 0 {
        return Err(invalid(
            "material read workspace requires a positive finite allocation bound",
        ));
    }
    let instance = material_bytes::python_instance_extent::<PySemanticMaterialReadWorkspace>(py)?;
    let extent = allocation_bytes
        .checked_add(
            u64::try_from(instance)
                .map_err(|_| invalid("material read workspace metadata exceeds u64"))?,
        )
        .ok_or_else(|| invalid("material read workspace allocation extent overflow"))?;
    let allocation = budget.reserve_auxiliary(extent).map_err(xlog_err)?;
    Py::new(
        py,
        PySemanticMaterialReadWorkspace {
            _owner: owner,
            _allocation: allocation,
        },
    )
}

#[pyclass(name = "SemanticMaterialSource", module = "pyxlog._native", frozen)]
pub(crate) struct PySemanticMaterialSource {
    owner_thread: std::thread::ThreadId,
    registry: Arc<material_source::MaterialSourceRegistry>,
    store_identity: Py<PyTuple>,
    _metadata: xlog_cuda::SemanticReplayMaterialReservation,
}

impl PySemanticMaterialSource {
    fn require_creator(&self) -> PyResult<()> {
        if self.owner_thread != std::thread::current().id() {
            return Err(invalid(
                "material source requires its original process owner thread",
            ));
        }
        Ok(())
    }

    pub(super) fn registry(&self) -> PyResult<Arc<material_source::MaterialSourceRegistry>> {
        self.require_creator()?;
        Ok(Arc::clone(&self.registry))
    }
}

pub(super) fn original_registry(
    source: &Bound<'_, PyAny>,
) -> PyResult<Arc<material_source::MaterialSourceRegistry>> {
    if !source.is_exact_instance_of::<PySemanticMaterialSource>() {
        return Err(invalid(
            "material source requires its original native process owner",
        ));
    }
    source
        .cast::<PySemanticMaterialSource>()?
        .borrow()
        .registry()
}

#[pyfunction]
#[pyo3(signature = (material_source, max_material_bytes, max_material_store_bytes, max_resolved_material_bytes))]
pub(crate) fn prepare_material_source(
    py: Python<'_>,
    material_source: &Bound<'_, PyAny>,
    max_material_bytes: u64,
    max_material_store_bytes: u64,
    max_resolved_material_bytes: u64,
) -> PyResult<Py<PySemanticMaterialSource>> {
    let registry = material_source::MaterialSourceRegistry::default();
    registry.bind(
        py,
        material_source,
        max_material_bytes,
        max_material_store_bytes,
        max_resolved_material_bytes,
    )?;
    let budget = registry.native_budget()?;
    let instance = material_bytes::python_instance_extent::<PySemanticMaterialSource>(py)?;
    let identity_extent = python_metadata_tuple_extent(py, 2)?;
    let integer = material_bytes::python_instance_extent::<PyInt>(py)?
        + std::mem::size_of::<u64>() * std::mem::size_of::<u32>();
    let registry_extent = std::mem::size_of::<material_source::MaterialSourceRegistry>()
        .checked_add(2 * std::mem::size_of::<usize>())
        .ok_or_else(|| invalid("material source registry allocation extent overflow"))?;
    let registry_metadata = budget
        .reserve_auxiliary(
            u64::try_from(registry_extent)
                .map_err(|_| invalid("material source registry allocation exceeds u64"))?,
        )
        .map_err(xlog_err)?;
    registry.retain_process_owner_metadata(registry_metadata)?;
    let extent = instance
        .checked_add(identity_extent)
        .and_then(|bytes| bytes.checked_add(2 * integer))
        .ok_or_else(|| invalid("material source owner metadata extent overflow"))?;
    let metadata = budget
        .reserve_auxiliary(
            u64::try_from(extent)
                .map_err(|_| invalid("material source owner metadata exceeds u64"))?,
        )
        .map_err(xlog_err)?;
    let store = registry.opened_store_identity()?;
    let store_identity = (store[0], store[1]).into_pyobject(py)?.unbind();
    Py::new(
        py,
        PySemanticMaterialSource {
            owner_thread: std::thread::current().id(),
            registry: Arc::new(registry),
            store_identity,
            _metadata: metadata,
        },
    )
}

#[pymethods]
impl PySemanticMaterialSource {
    #[getter]
    fn store_identity(&self, py: Python<'_>) -> PyResult<Py<PyTuple>> {
        self.require_creator()?;
        Ok(self.store_identity.clone_ref(py))
    }

    fn extend(&self, py: Python<'_>, material_source: &Bound<'_, PyAny>) -> PyResult<()> {
        self.require_creator()?;
        let (item_cap, store_cap, resolved_cap) = self
            .registry
            .saved_caps()?
            .ok_or_else(|| invalid("material source has no original immutable caps"))?;
        self.registry
            .bind(py, material_source, item_cap, store_cap, resolved_cap)
    }

    fn reserve_read_workspace(
        &self,
        py: Python<'_>,
        allocation_bytes: u64,
    ) -> PyResult<Py<PySemanticMaterialReadWorkspace>> {
        self.require_creator()?;
        prepare_read_workspace(
            py,
            OriginalMaterialReadOwner::Source {
                _source: Arc::clone(&self.registry),
            },
            &self.registry.native_budget()?,
            allocation_bytes,
        )
    }

    fn resolve_stored(
        &self,
        py: Python<'_>,
        sha256_hex: &Bound<'_, PyAny>,
        size: u64,
    ) -> PyResult<Py<material_bytes::PySemanticMaterialBytes>> {
        self.require_creator()?;
        if !sha256_hex.is_exact_instance_of::<PyString>() || size == 0 {
            return Err(invalid(
                "stored material requires its exact sealed SHA256 and positive extent",
            ));
        }
        let digest = sha256_hex.cast::<PyString>()?.to_str()?;
        material_source::validate_digest(digest)?;
        let budget = self.registry.native_budget()?;
        let _digest_metadata = budget
            .reserve_auxiliary(
                u64::try_from(digest.len())
                    .map_err(|_| invalid("stored digest extent exceeds u64"))?,
            )
            .map_err(xlog_err)?;
        let reference = material_source::StoredMaterial {
            store: material_source::MaterialStoreIdentity(self.registry.opened_store_identity()?),
            digest: digest.to_owned(),
            size,
            source: Arc::downgrade(&self.registry),
        };
        self.registry.resolve_carrier(py, &reference)
    }

    fn prepare_checkpoint_writer(
        &self,
        py: Python<'_>,
        capacity: usize,
        working_bytes: u64,
    ) -> PyResult<Py<material_bytes::PySemanticMaterialWriter>> {
        self.require_creator()?;
        material_bytes::PySemanticMaterialWriter::prepare(
            py,
            &self.registry.native_budget()?,
            capacity,
            working_bytes,
        )
    }
}
