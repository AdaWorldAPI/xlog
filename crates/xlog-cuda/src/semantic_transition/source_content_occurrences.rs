use super::*;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SemanticSourceTensorDescriptor {
    pub layout: SemanticTensorLayout,
    pub logical_begin: u64,
    pub logical_end: u64,
    pub storage_slot: u64,
    // An optional registered producer version group identifies this view's owner.
    pub version_slot: Option<u64>,
    pub allocation_slot: u64,
    pub byte_offset: u64,
}

#[derive(Clone)]
pub struct SemanticSourceAllocationDescriptor<A = DeviceAllocationProvenance> {
    pub slot: u64,
    pub span_bytes: u64,
    pub native_allocation: Option<A>,
}

#[derive(Clone)]
pub enum SemanticSourceContentAllocation {
    Native(DeviceAllocationProvenance),
    Geometry(prepared_source_content::SemanticSourceGeometryAllocation),
}

mod allocation_seal {
    pub trait Sealed {}
    impl Sealed for super::DeviceAllocationProvenance {}
    impl Sealed for super::SemanticSourceContentAllocation {}
    impl Sealed for super::prepared_source_content::SemanticSourceGeometryAllocation {}
}

pub trait SourceContentAllocation: allocation_seal::Sealed + Clone {
    fn native_provenance(&self) -> Option<&DeviceAllocationProvenance>;
    fn original_geometry(
        &self,
    ) -> Option<&prepared_source_content::SemanticSourceGeometryAllocation>;
}

impl SourceContentAllocation for DeviceAllocationProvenance {
    fn native_provenance(&self) -> Option<&DeviceAllocationProvenance> {
        Some(self)
    }
    fn original_geometry(
        &self,
    ) -> Option<&prepared_source_content::SemanticSourceGeometryAllocation> {
        None
    }
}
impl SourceContentAllocation for prepared_source_content::SemanticSourceGeometryAllocation {
    fn native_provenance(&self) -> Option<&DeviceAllocationProvenance> {
        None
    }
    fn original_geometry(
        &self,
    ) -> Option<&prepared_source_content::SemanticSourceGeometryAllocation> {
        Some(self)
    }
}
impl SourceContentAllocation for SemanticSourceContentAllocation {
    fn native_provenance(&self) -> Option<&DeviceAllocationProvenance> {
        match self {
            Self::Native(value) => Some(value),
            Self::Geometry(_) => None,
        }
    }
    fn original_geometry(
        &self,
    ) -> Option<&prepared_source_content::SemanticSourceGeometryAllocation> {
        match self {
            Self::Native(_) => None,
            Self::Geometry(value) => Some(value),
        }
    }
}
