//! Typed source geometry rows for original allocation and restore ownership.

use super::*;
use xlog_cuda::{
    SemanticSourceAllocationDescriptor, SemanticSourceContentAllocation,
    SemanticSourceTensorDescriptor,
};

pub(super) fn prepared_section_arguments(
    kind: Option<&Bound<'_, PyAny>>,
    bank: Option<&Bound<'_, PyAny>>,
) -> PyResult<Option<(SemanticTransitionKind, usize)>> {
    match (kind, bank) {
        (None, None) => Ok(None),
        (Some(kind), Some(bank)) => {
            if !kind.is_exact_instance_of::<PyString>() || !bank.is_exact_instance_of::<PyInt>() {
                return Err(invalid(
                    "prepared content section requires an exact string and integer bank",
                ));
            }
            let same = |spelling: &std::ffi::CStr| unsafe {
                pyo3::ffi::PyUnicode_CompareWithASCIIString(kind.as_ptr(), spelling.as_ptr()) == 0
            };
            let kind = if same(c"recompute") {
                SemanticTransitionKind::Recompute
            } else if same(c"proposal") {
                SemanticTransitionKind::Proposal
            } else {
                return Err(invalid(
                    "prepared content section requires the original Recompute or Proposal",
                ));
            };
            let bank = bank.extract::<usize>()?;
            if bank > 1 {
                return Err(invalid(
                    "prepared content section requires bank zero or one",
                ));
            }
            Ok(Some((kind, bank)))
        }
        _ => Err(invalid(
            "prepared content section requires both kind and bank",
        )),
    }
}

pub(super) fn exact_tuple<'py>(
    value: &Bound<'py, PyAny>,
    arity: Option<usize>,
    budget: &mut usize,
) -> PyResult<Bound<'py, PyTuple>> {
    if !value.is_exact_instance_of::<PyTuple>() {
        return Err(invalid(
            "source content metadata requires exact immutable tuples",
        ));
    }
    let tuple = value.cast::<PyTuple>()?;
    if arity.is_some_and(|expected| tuple.len() != expected) {
        return Err(invalid(
            "source content metadata has an incorrect tuple length",
        ));
    }
    *budget = budget
        .checked_sub(tuple.len().saturating_add(1))
        .ok_or_else(|| invalid("source content metadata exceeds its transport bound"))?;
    Ok(tuple.clone())
}

fn unsigned(value: &Bound<'_, PyAny>) -> PyResult<u64> {
    if !value.is_exact_instance_of::<PyInt>() {
        return Err(invalid(
            "source content coordinates require exact unsigned integers",
        ));
    }
    value.extract()
}

fn optional_unsigned(value: &Bound<'_, PyAny>) -> PyResult<Option<u64>> {
    if value.is_none() {
        Ok(None)
    } else {
        unsigned(value).map(Some)
    }
}

fn unsigned_array_from_tuple(value: &Bound<'_, PyAny>, budget: &mut usize) -> PyResult<[u64; 4]> {
    let tuple = exact_tuple(value, Some(4), budget)?;
    let mut values = [0; 4];
    for (index, value) in values.iter_mut().enumerate() {
        *value = unsigned(&tuple.get_item(index)?)?;
    }
    Ok(values)
}

pub(super) fn parse_tensor_descriptors(
    value: &Bound<'_, PyAny>,
    budget: &mut usize,
) -> PyResult<Vec<SemanticSourceTensorDescriptor>> {
    let rows = exact_tuple(value, None, budget)?;
    let mut tensors = Vec::with_capacity(rows.len());
    for tensor in rows.iter() {
        let tensor = exact_tuple(&tensor, Some(7), budget)?;
        let layout = exact_tuple(&tensor.get_item(0)?, Some(8), budget)?;
        let mut fields = [0; 6];
        for (index, value) in fields.iter_mut().enumerate() {
            *value = unsigned(&layout.get_item(index)?)?;
        }
        tensors.push(SemanticSourceTensorDescriptor {
            layout: tensor_layout_from_values(
                fields,
                unsigned_array_from_tuple(&layout.get_item(6)?, budget)?,
                unsigned_array_from_tuple(&layout.get_item(7)?, budget)?,
            ),
            logical_begin: unsigned(&tensor.get_item(1)?)?,
            logical_end: unsigned(&tensor.get_item(2)?)?,
            storage_slot: unsigned(&tensor.get_item(3)?)?,
            version_slot: optional_unsigned(&tensor.get_item(4)?)?,
            allocation_slot: unsigned(&tensor.get_item(5)?)?,
            byte_offset: unsigned(&tensor.get_item(6)?)?,
        });
    }
    Ok(tensors)
}

pub(super) fn parse_allocation_descriptors(
    value: &Bound<'_, PyAny>,
    budget: &mut usize,
) -> PyResult<Vec<SemanticSourceAllocationDescriptor<SemanticSourceContentAllocation>>> {
    let rows = exact_tuple(value, None, budget)?;
    let mut allocations = Vec::with_capacity(rows.len());
    for allocation in rows.iter() {
        let allocation = exact_tuple(&allocation, Some(3), budget)?;
        let provenance = allocation.get_item(2)?;
        allocations.push(SemanticSourceAllocationDescriptor {
            slot: unsigned(&allocation.get_item(0)?)?,
            span_bytes: unsigned(&allocation.get_item(1)?)?,
            native_allocation: if provenance.is_none() {
                None
            } else if provenance
                .is_instance_of::<source_content_geometry::PySemanticSourceGeometryAllocation>()
            {
                Some(SemanticSourceContentAllocation::Geometry(
                    provenance
                        .extract::<PyRef<
                            '_,
                            source_content_geometry::PySemanticSourceGeometryAllocation,
                        >>()?
                        .inner
                        .clone(),
                ))
            } else {
                Some(SemanticSourceContentAllocation::Native(
                    provenance
                        .extract::<PyRef<'_, PyNativeTensorAllocation>>()?
                        .provenance
                        .clone(),
                ))
            },
        });
    }
    Ok(allocations)
}
