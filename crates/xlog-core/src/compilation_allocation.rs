//! Shared allocation custody for canonical compilation and metadata.

use std::sync::Arc;

use crate::{Result, XlogError};

/// Original allocation receipts for one canonical compilation. The caller's
/// callback is the capacity authority; this scope neither chooses a budget nor
/// grants parsing, admission, or execution rights.
pub struct CompilationAllocationScope<'a> {
    reserve: &'a mut dyn FnMut(u64) -> Result<Arc<dyn Send + Sync>>,
    receipts: Vec<Arc<dyn Send + Sync>>,
    planner_threshold: Option<f64>,
    receipt_storage: Option<Arc<dyn Send + Sync>>,
    owner_metadata: Option<Arc<dyn Send + Sync>>,
}

struct CompilationAllocationReceipts {
    _receipts: Vec<Arc<dyn Send + Sync>>,
    _receipt_storage: Option<Arc<dyn Send + Sync>>,
    _metadata: Arc<dyn Send + Sync>,
}

/// Prepared custody for one original diagnostic payload, independent of the
/// compiler stage's temporary allocations. Preparation ends the callback borrow
/// before the producer formats its original borrowed context.
pub struct CompilationDiagnosticProducer {
    custody: Option<(Arc<dyn Send + Sync>, Arc<dyn Send + Sync>)>,
}

impl CompilationDiagnosticProducer {
    /// Construct the unchanged classified diagnostic only after its payload and
    /// carrier have been acquired. This creates no compiler-stage owner.
    pub fn finish(self, build: impl FnOnce() -> XlogError) -> XlogError {
        let error = build();
        match self.custody {
            Some((payload, metadata)) => error.with_diagnostic_owner(payload, metadata),
            None => error,
        }
    }
}

/// Checked typed hash-table storage for the original requested entry count.
pub fn hash_allocation_bytes<K, V>(count: usize) -> Result<u64> {
    if count == 0 {
        return Ok(0);
    }
    let overflow = || XlogError::allocation_extent_overflow("compiler table");
    // Swiss-table storage has one control byte per bucket, a trailing control
    // group, and a seven-eighths maximum load factor. Eight buckets also cover
    // the original small-table special cases.
    let buckets = count
        .checked_mul(8)
        .and_then(|n| n.checked_add(6))
        .map(|n| n / 7)
        .and_then(|n| n.max(8).checked_next_power_of_two())
        .ok_or_else(overflow)?;
    buckets
        .checked_mul(
            std::mem::size_of::<(K, V)>()
                .checked_add(1)
                .ok_or_else(overflow)?,
        )
        .and_then(|n| n.checked_add(16))
        .and_then(|n| u64::try_from(n).ok())
        .ok_or_else(overflow)
}

/// Checked geometric storage of the original push-built vector and its growth.
pub fn push_vector_allocation_bytes<T>(count: usize) -> Result<u64> {
    let width = std::mem::size_of::<T>();
    if count == 0 || width == 0 {
        return Ok(0);
    }
    let minimum = if width == 1 {
        8
    } else if width <= 1024 {
        4
    } else {
        1
    };
    // Capacity of the actual push-built vector, including its old buffer at a
    // geometric growth operation. This does not allocate a counting vector.
    count
        .checked_mul(2)
        .map(|n| n.max(minimum))
        .and_then(|n| n.checked_mul(3))
        .and_then(|n| n.checked_mul(width))
        .and_then(|n| u64::try_from(n).ok())
        .ok_or_else(|| XlogError::allocation_extent_overflow("compiler vector"))
}

/// Checked typed tree-node storage at one original insertion boundary.
pub fn btree_insert_allocation_bytes<K, V>(length: usize) -> Result<u64> {
    let overflow = || XlogError::allocation_extent_overflow("compiler tree");
    let mut height = 1usize;
    let mut entries = length;
    while entries >= 6 {
        entries /= 6;
        height = height.checked_add(1).ok_or_else(overflow)?;
    }
    let node = (2 * std::mem::size_of::<usize>())
        .checked_add(2 * std::mem::size_of::<u16>())
        .and_then(|n| n.checked_add(11usize.checked_mul(std::mem::size_of::<(K, V)>())?))
        .and_then(|n| n.checked_add(12 * std::mem::size_of::<usize>()))
        .ok_or_else(overflow)?;
    let alignment = std::mem::align_of::<(K, V)>().max(std::mem::align_of::<usize>());
    let node = node
        .checked_add(alignment - 1)
        .map(|bytes| bytes / alignment * alignment)
        .ok_or_else(overflow)?;
    node.checked_mul(height.checked_add(1).ok_or_else(overflow)?)
        .and_then(|n| u64::try_from(n).ok())
        .ok_or_else(overflow)
}

impl<'a> CompilationAllocationScope<'a> {
    /// Acquire only one diagnostic's original payload and carrier. Ordinary
    /// producers keep the same constructor; bounded producers never format
    /// before the same capacity callback has accepted both actual allocations.
    pub fn prepare_diagnostic(
        bytes: Result<u64>,
        reserve: Option<&mut dyn FnMut(u64) -> Result<Arc<dyn Send + Sync>>>,
    ) -> Result<CompilationDiagnosticProducer> {
        match reserve {
            Some(reserve) => {
                let payload = reserve(bytes?)?;
                let metadata_bytes = u64::try_from(
                    crate::error::OwnedDiagnostic::allocation_bytes(),
                )
                .map_err(|_| XlogError::allocation_extent_overflow("compiler diagnostic owner"))?;
                let metadata = reserve(metadata_bytes)?;
                Ok(CompilationDiagnosticProducer {
                    custody: Some((payload, metadata)),
                })
            }
            None => Ok(CompilationDiagnosticProducer { custody: None }),
        }
    }
    /// Start an empty scope without allocating receipt storage.
    pub fn new(reserve: &'a mut dyn FnMut(u64) -> Result<Arc<dyn Send + Sync>>) -> Self {
        Self {
            reserve,
            receipts: Vec::new(),
            planner_threshold: None,
            receipt_storage: None,
            owner_metadata: None,
        }
    }

    /// A stage-local receipt scope under the exact same reservation authority
    /// and captured planner configuration. It creates no pool or receipt until
    /// an actual stage allocation is acquired.
    pub fn child_scope(&mut self) -> CompilationAllocationScope<'_> {
        CompilationAllocationScope {
            reserve: &mut *self.reserve,
            receipts: Vec::new(),
            planner_threshold: self.planner_threshold,
            receipt_storage: None,
            owner_metadata: None,
        }
    }

    /// Attach the caller's original parsed planner configuration. Owned
    /// planning never reads or substitutes an environment value.
    pub fn with_planner_threshold(mut self, threshold: f64) -> Result<Self> {
        if !threshold.is_finite() || threshold <= 0.0 {
            let producer = Self::prepare_diagnostic(
                Ok("original planner threshold must be finite and positive".len() as u64),
                Some(&mut *self.reserve),
            )?;
            return Err(producer.finish(|| {
                XlogError::Compilation(
                    "original planner threshold must be finite and positive".into(),
                )
            }));
        }
        self.planner_threshold = Some(threshold);
        Ok(self)
    }

    /// Return the exact original captured planner threshold, if one was supplied.
    pub fn planner_threshold(&self) -> Option<f64> {
        self.planner_threshold
    }

    /// Receipt boundary for a genuine attempt. The caller may retire the
    /// suffix only after all of that failed attempt's actual data has dropped.
    pub fn receipt_checkpoint(&self) -> usize {
        self.receipts.len()
    }

    /// Retire a known-discarded attempt; retained receipt-vector capacity keeps
    /// its separate original owner and cannot be released by this operation.
    pub fn retire_since(&mut self, checkpoint: usize) -> Result<()> {
        self.require_receipt_checkpoint(checkpoint)?;
        self.receipts.truncate(checkpoint);
        Ok(())
    }

    fn require_receipt_checkpoint(&mut self, checkpoint: usize) -> Result<()> {
        if checkpoint > self.receipts.len() {
            let producer = Self::prepare_diagnostic(
                Ok("compiler receipt boundary is not original".len() as u64),
                Some(&mut *self.reserve),
            )?;
            return Err(producer.finish(|| {
                XlogError::Compilation("compiler receipt boundary is not original".into())
            }));
        }
        Ok(())
    }

    /// Prepare the original survivor's receipt storage before retiring data.
    /// The closure drops replaced allocations while their old receipts remain
    /// held; replacing the suffix afterward has no fallible allocation step.
    pub fn replace_since_with_retained_allocation(
        &mut self,
        checkpoint: usize,
        receipt: Arc<dyn Send + Sync>,
        retire: impl FnOnce(),
    ) -> Result<()> {
        self.require_receipt_checkpoint(checkpoint)?;
        if checkpoint == self.receipts.len() {
            self.reserve_receipt_storage()?;
        }
        retire();
        self.receipts.truncate(checkpoint);
        self.receipts.push(receipt);
        Ok(())
    }

    fn reserve_receipt_storage(&mut self) -> Result<()> {
        if self.receipts.len() == self.receipts.capacity() {
            let capacity = self
                .receipts
                .len()
                .checked_add(1)
                .and_then(|n| n.max(4).checked_next_power_of_two())
                .ok_or_else(|| XlogError::allocation_extent_overflow("compiler receipt"))?;
            let bytes = capacity
                .checked_mul(std::mem::size_of::<Arc<dyn Send + Sync>>())
                .and_then(|n| u64::try_from(n).ok())
                .ok_or_else(|| XlogError::allocation_extent_overflow("compiler receipt"))?;
            let owner = (self.reserve)(bytes)?;
            self.receipts.reserve_exact(capacity - self.receipts.len());
            self.receipt_storage = Some(owner);
        }
        Ok(())
    }

    /// Acquire custody for an actual temporary owner. Unlike `acquire`, the
    /// caller retains this receipt in that owner and retires it with its data.
    pub fn reserve_allocation(&mut self, bytes: u64) -> Result<Arc<dyn Send + Sync>> {
        (self.reserve)(bytes)
    }

    /// Acquire the returned owner's own metadata before the result's data is
    /// constructed. Final transfer then cannot refuse capacity after that data
    /// has already outlived this scope's receipts.
    pub fn prepare_owner(&mut self) -> Result<()> {
        if self.owner_metadata.is_none() {
            let bytes = std::mem::size_of::<CompilationAllocationReceipts>()
                .checked_add(2 * std::mem::size_of::<usize>())
                .and_then(|n| n.checked_add(crate::error::OwnedDiagnostic::allocation_bytes()))
                .and_then(|n| u64::try_from(n).ok())
                .ok_or_else(|| XlogError::allocation_extent_overflow("compiler owner"))?;
            self.owner_metadata = Some((self.reserve)(bytes)?);
        }
        Ok(())
    }

    /// Retain the already-acquired receipt of an actual returned allocation.
    pub fn retain(&mut self, receipt: Arc<dyn Send + Sync>) -> Result<()> {
        self.reserve_receipt_storage()?;
        self.receipts.push(receipt);
        Ok(())
    }

    /// Acquire a receipt before its actual allocation. The receipt collection
    /// itself is charged before growth and retained by the same original scope.
    pub fn acquire(&mut self, bytes: u64) -> Result<Arc<dyn Send + Sync>> {
        self.reserve_receipt_storage()?;
        let receipt = (self.reserve)(bytes)?;
        self.receipts.push(receipt.clone());
        Ok(receipt)
    }

    /// Transfer all receipts into the original returned compiler owner. Its
    /// metadata is acquired before the owning Arc is allocated.
    pub fn into_owner(mut self) -> Result<Arc<dyn Send + Sync>> {
        self.prepare_owner()?;
        let metadata = self
            .owner_metadata
            .take()
            .expect("prepared compiler owner metadata");
        Ok(Arc::new(CompilationAllocationReceipts {
            _receipts: self.receipts,
            _receipt_storage: self.receipt_storage,
            _metadata: metadata,
        }))
    }

    /// Preserve a prepared failed attempt's diagnostic and original receipts.
    /// The two carrier allocations were acquired by `prepare_owner` before
    /// the attempt. No new reservation can replace the original failure here.
    pub fn into_failure(mut self, error: XlogError) -> XlogError {
        let metadata = self
            .owner_metadata
            .take()
            .expect("failed compiler attempt requires its prepared original owner");
        let allocations = Arc::new(CompilationAllocationReceipts {
            _receipts: self.receipts,
            _receipt_storage: self.receipt_storage,
            _metadata: Arc::clone(&metadata),
        });
        error.with_diagnostic_owner(allocations, metadata)
    }
}
