//! Original publication projections and their retained cold allocation owners.

use super::material_bytes::PySemanticMaterialBytes;
use super::*;
use pyo3::exceptions::PyKeyError;

const FIELD_NAMES: [&str; 14] = [
    "schema",
    "publication",
    "semantic_root",
    "model_geometry_digest",
    "prefix_identity",
    "model_generation",
    "neural_generation",
    "cache_generation",
    "prefix_extent",
    "ring_head",
    "prefix",
    "source",
    "token_provenance",
    "cache",
];

const ORIGIN_KEYS: [&str; 12] = [
    "source_slot",
    "logical_position",
    "token",
    "base_word",
    "proposal",
    "ordinal",
    "authority_generation",
    "action_receipt_digest",
    "origin_logical_digest",
    "successor_logical_digest",
    "authority_closure_digest",
    "record_digest",
];

const CACHE_KEYS: [&str; 8] = [
    "role",
    "index",
    "capacity_bytes",
    "logical_begin",
    "logical_end",
    "layout",
    "identity",
    "logical_bytes",
];

#[pyclass(
    name = "SemanticPublicationMaterialProjection",
    module = "pyxlog._native",
    frozen
)]
pub(crate) struct PySemanticPublicationMaterialProjection {
    fields: [Py<PyAny>; 14],
    _original: xlog_cuda::SemanticLoadedPublicationInputProjection,
    _metadata: Arc<xlog_cuda::SemanticReplayMaterialReservation>,
}

#[pymethods]
impl PySemanticPublicationMaterialProjection {
    #[pyo3(signature = (key, default=None))]
    fn get(&self, py: Python<'_>, key: &str, default: Option<Py<PyAny>>) -> Py<PyAny> {
        match FIELD_NAMES.iter().position(|name| *name == key) {
            Some(index) => self.fields[index].clone_ref(py),
            None => default.unwrap_or_else(|| py.None()),
        }
    }

    fn __getitem__(&self, py: Python<'_>, key: &str) -> PyResult<Py<PyAny>> {
        let index = FIELD_NAMES
            .iter()
            .position(|name| *name == key)
            .ok_or_else(|| PyKeyError::new_err(key.to_owned()))?;
        Ok(self.fields[index].clone_ref(py))
    }
}

fn add_extent(total: &mut usize, extent: usize) -> PyResult<()> {
    *total = total
        .checked_add(extent)
        .ok_or_else(|| invalid("publication projection metadata extent overflow"))?;
    Ok(())
}

fn repeated_extent(extent: usize, count: usize) -> PyResult<usize> {
    extent
        .checked_mul(count)
        .ok_or_else(|| invalid("publication projection metadata extent overflow"))
}

fn dictionary_extent(py: Python<'_>, keys: &[&str]) -> PyResult<usize> {
    let word = std::mem::size_of::<usize>();
    let mut extent = material_bytes::python_instance_extent::<PyDict>(py)?;
    let mut slots = 8usize;
    loop {
        let entries = slots
            .checked_mul(2)
            .ok_or_else(|| invalid("publication dictionary capacity overflow"))?
            / 3;
        // Combined CPython dictionaries start with eight indices and grow at
        // two-thirds occupancy. The older keys header occupies five words;
        // general entries have hash/key/value words and bound Unicode entries.
        // Retain every reached resize table, including simultaneous old/new
        // tables, with the widest possible index width.
        let table = 5usize
            .checked_add(slots)
            .and_then(|count| count.checked_add(entries.checked_mul(3)?))
            .and_then(|count| count.checked_mul(word))
            .ok_or_else(|| invalid("publication dictionary capacity overflow"))?;
        add_extent(&mut extent, table)?;
        if entries >= keys.len() {
            break;
        }
        slots = slots
            .checked_mul(2)
            .ok_or_else(|| invalid("publication dictionary capacity overflow"))?;
    }
    for key in keys {
        add_extent(&mut extent, python_metadata_text_extent(py, key)?)?;
    }
    Ok(extent)
}

fn projection_metadata_extent(
    py: Python<'_>,
    original: &xlog_cuda::SemanticPublicationInputProjection<
        xlog_cuda::SemanticReplayLoadedMaterial,
    >,
) -> PyResult<usize> {
    let mut extent =
        material_bytes::python_instance_extent::<PySemanticPublicationMaterialProjection>(py)?;
    add_extent(
        &mut extent,
        std::mem::size_of::<xlog_cuda::SemanticReplayMaterialReservation>()
            .checked_add(2 * std::mem::size_of::<usize>())
            .ok_or_else(|| invalid("publication metadata owner extent overflow"))?,
    )?;
    let integer = material_bytes::python_instance_extent::<PyInt>(py)?
        .checked_add(std::mem::size_of::<u64>() * std::mem::size_of::<u32>())
        .ok_or_else(|| invalid("publication integer extent overflow"))?;
    let identity = material_bytes::python_instance_extent::<PyBytes>(py)?
        .checked_add(33)
        .ok_or_else(|| invalid("publication identity extent overflow"))?;
    add_extent(&mut extent, repeated_extent(integer, 7)?)?;
    add_extent(&mut extent, repeated_extent(identity, 6)?)?;
    add_extent(&mut extent, python_metadata_tuple_extent(py, 4)?)?;
    for count in [original.prefix.len(), original.source.len()] {
        add_extent(&mut extent, python_metadata_tuple_extent(py, count)?)?;
        let row = python_metadata_tuple_extent(py, 8)?
            .checked_add(repeated_extent(integer, 8)?)
            .ok_or_else(|| invalid("publication text row extent overflow"))?;
        add_extent(&mut extent, repeated_extent(row, count)?)?;
    }
    let count = original.token_provenance.len();
    add_extent(&mut extent, python_metadata_tuple_extent(py, count)?)?;
    let origin = dictionary_extent(py, &ORIGIN_KEYS)?
        .checked_add(repeated_extent(integer, 7)?)
        .and_then(|bytes| bytes.checked_add(identity.checked_mul(5)?))
        .and_then(|bytes| bytes.checked_add(std::mem::size_of::<Bound<'_, PyDict>>()))
        .ok_or_else(|| invalid("publication provenance extent overflow"))?;
    add_extent(&mut extent, repeated_extent(origin, count)?)?;
    let count = original.cache.len();
    add_extent(&mut extent, python_metadata_tuple_extent(py, count)?)?;
    let mut cache = dictionary_extent(py, &CACHE_KEYS)?;
    for count in [8, 4, 4] {
        add_extent(&mut cache, python_metadata_tuple_extent(py, count)?)?;
    }
    add_extent(&mut cache, repeated_extent(integer, 19)?)?;
    add_extent(&mut cache, identity)?;
    add_extent(&mut cache, std::mem::size_of::<Bound<'_, PyDict>>())?;
    add_extent(&mut extent, repeated_extent(cache, count)?)?;
    Ok(extent)
}

pub(super) fn inspect(
    py: Python<'_>,
    full_p: &Bound<'_, PyAny>,
) -> PyResult<Py<PySemanticPublicationMaterialProjection>> {
    if !full_p.is_exact_instance_of::<PySemanticMaterialBytes>() {
        return Err(invalid(
            "publication inspection requires original bounded material bytes",
        ));
    }
    let bytes = full_p.extract::<PyRef<'_, PySemanticMaterialBytes>>()?;
    let budget = bytes.material_budget().clone();
    let mut loaded = bytes.original_loaded(&budget)?;
    let original = SemanticTransitionSession::state_material_loaded_input_projection(&mut loaded)
        .map_err(xlog_err)?;
    let projection = original.projection();
    let metadata = Arc::new(material_bytes::reserve_material_metadata(
        &budget,
        projection_metadata_extent(py, projection)?,
    )?);

    let publication = completed_publication_identity(py, projection.publication)?.into_any();
    let prefix = PyTuple::new(py, projection.prefix.iter().map(text_slot_fields))?
        .into_any()
        .unbind();
    let source = PyTuple::new(py, projection.source.iter().map(text_slot_fields))?
        .into_any()
        .unbind();
    let mut origins = Vec::with_capacity(projection.token_provenance.len());
    for record in &projection.token_provenance {
        let origin = PyDict::new(py);
        for (name, value) in [
            ("source_slot", record.source_slot),
            ("logical_position", record.logical_position),
            ("token", record.token),
            ("base_word", record.base_word),
            ("proposal", record.proposal),
            ("ordinal", record.ordinal),
            ("authority_generation", record.authority_generation),
        ] {
            origin.set_item(name, value)?;
        }
        for (name, identity) in [
            ("action_receipt_digest", record.action_receipt_digest),
            ("origin_logical_digest", record.origin_logical_digest),
            ("successor_logical_digest", record.successor_logical_digest),
            ("authority_closure_digest", record.authority_closure_digest),
            ("record_digest", record.record_digest),
        ] {
            origin.set_item(name, PyBytes::new(py, identity.as_bytes()))?;
        }
        origins.push(origin);
    }
    let token_provenance = PyTuple::new(py, origins)?.into_any().unbind();
    let mut caches = Vec::with_capacity(projection.cache.len());
    for item in &projection.cache {
        let cache = PyDict::new(py);
        let layout = item.layout;
        for (name, value) in [
            ("role", layout.role),
            ("index", layout.index),
            ("capacity_bytes", item.capacity_bytes),
            ("logical_begin", item.logical_begin),
            ("logical_end", item.logical_end),
        ] {
            cache.set_item(name, value)?;
        }
        cache.set_item(
            "layout",
            (
                layout.role,
                layout.index,
                layout.element_bytes,
                layout.scalar_type,
                layout.rank,
                layout.logical_axis,
                PyTuple::new(py, layout.dimensions)?,
                PyTuple::new(py, layout.strides_bytes)?,
            ),
        )?;
        cache.set_item("identity", PyBytes::new(py, item.identity.as_bytes()))?;
        let payload = item
            .logical_bytes
            .slice_payload(0..item.logical_bytes.as_bytes().len())
            .and_then(|bytes| bytes.retain_shared_metadata(Arc::clone(&metadata)))
            .map_err(xlog_err)?;
        cache.set_item(
            "logical_bytes",
            Py::new(
                py,
                PySemanticMaterialBytes::from_native(py, payload, &budget)?,
            )?,
        )?;
        caches.push(cache);
    }
    let cache = PyTuple::new(py, caches)?.into_any().unbind();
    let integer =
        |value: u64| -> PyResult<Py<PyAny>> { Ok(value.into_pyobject(py)?.into_any().unbind()) };
    let fields = [
        integer(1)?,
        publication,
        PyBytes::new(py, projection.semantic_root.as_bytes())
            .into_any()
            .unbind(),
        PyBytes::new(py, projection.model_geometry_digest.as_bytes())
            .into_any()
            .unbind(),
        PyBytes::new(py, projection.prefix_identity.as_bytes())
            .into_any()
            .unbind(),
        integer(projection.model_generation)?,
        integer(projection.neural_generation)?,
        integer(projection.cache_generation)?,
        integer(projection.prefix_extent)?,
        integer(projection.ring_head)?,
        prefix,
        source,
        token_provenance,
        cache,
    ];
    Py::new(
        py,
        PySemanticPublicationMaterialProjection {
            fields,
            _original: original,
            _metadata: metadata,
        },
    )
}
