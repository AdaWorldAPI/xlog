//! Allocation custody for a diagnostic produced by an owned operation.

use std::fmt;
use std::sync::Arc;

use super::XlogError;

/// An immutable original diagnostic and the receipts for its actual data.
///
/// The diagnostic drops before its allocation receipts. Cloning its `Arc`
/// keeps both together; a language binding must retain that same owner rather
/// than formatting the message and discarding it.
pub struct OwnedDiagnostic {
    error: XlogError,
    _allocations: Arc<dyn Send + Sync>,
    _metadata: Arc<dyn Send + Sync>,
}

impl OwnedDiagnostic {
    pub(super) fn new(
        error: XlogError,
        allocations: Arc<dyn Send + Sync>,
        metadata: Arc<dyn Send + Sync>,
    ) -> Self {
        Self {
            error,
            _allocations: allocations,
            _metadata: metadata,
        }
    }

    /// Metadata extent of this carrier's original `Arc` allocation.
    ///
    /// The returned extent is reserved before the operation's data is built;
    /// it is not a bound for that data or permission to format another copy.
    pub const fn allocation_bytes() -> usize {
        std::mem::size_of::<Self>() + 2 * std::mem::size_of::<usize>()
    }

    /// Borrow the original error without detaching allocation custody.
    pub fn error(&self) -> &XlogError {
        &self.error
    }
}

impl fmt::Debug for OwnedDiagnostic {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Debug::fmt(&self.error, formatter)
    }
}

impl fmt::Display for OwnedDiagnostic {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(&self.error, formatter)
    }
}

impl std::error::Error for OwnedDiagnostic {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        Some(&self.error)
    }
}
