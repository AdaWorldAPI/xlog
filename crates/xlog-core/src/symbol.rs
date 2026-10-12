//! Global symbol interning for reversible string-to-ID mapping.

use arrow::array::{Array, DictionaryArray, StringArray, UInt32Array};
use arrow::datatypes::UInt32Type;
use std::collections::HashMap;
use std::sync::{Arc, OnceLock, RwLock};

static REGISTRY: OnceLock<RwLock<SymbolRegistry>> = OnceLock::new();

struct SymbolRegistry {
    to_id: HashMap<String, u32>,
    to_string: Vec<String>,
    string_owners: Vec<Option<Arc<dyn Send + Sync>>>,
    table_owner: Option<Arc<dyn Send + Sync>>,
    strings_array_owner: Option<Arc<dyn Send + Sync>>,
    owners_array_owner: Option<Arc<dyn Send + Sync>>,
}

impl SymbolRegistry {
    fn new() -> Self {
        Self {
            to_id: HashMap::new(),
            to_string: Vec::new(),
            string_owners: Vec::new(),
            table_owner: None,
            strings_array_owner: None,
            owners_array_owner: None,
        }
    }
}

fn registry() -> &'static RwLock<SymbolRegistry> {
    REGISTRY.get_or_init(|| RwLock::new(SymbolRegistry::new()))
}

/// Intern a string, returning its unique ID.
/// Thread-safe. Returns existing ID if already interned.
pub fn intern(s: &str) -> u32 {
    intern_owned(s, None).expect("symbol registry allocation")
}

/// Actual UTF-8 clone extent of an original symbol, without making a copy.
pub fn resolved_allocation_bytes(id: u32) -> crate::Result<usize> {
    let reg = registry().read().unwrap();
    reg.to_string
        .get(id as usize)
        .map(String::len)
        .ok_or_else(|| crate::XlogError::Compilation("unknown original symbol identity".into()))
}

/// Intern in the original registry, retaining native allocation custody.
/// The callback reserves capacity only; it must not invoke a language runtime.
pub fn intern_with_allocation_owner(
    s: &str,
    reserve: &mut dyn FnMut(usize) -> crate::Result<Arc<dyn Send + Sync>>,
) -> crate::Result<u32> {
    intern_owned(s, Some(reserve))
}

fn intern_owned(
    s: &str,
    mut reserve: Option<&mut dyn FnMut(usize) -> crate::Result<Arc<dyn Send + Sync>>>,
) -> crate::Result<u32> {
    // Fast path: check if already interned (read lock)
    {
        let reg = registry().read().unwrap();
        if let Some(&id) = reg.to_id.get(s) {
            return Ok(id);
        }
    }
    // Slow path: insert new (write lock)
    let mut reg = registry().write().unwrap();
    // Double-check after acquiring write lock
    if let Some(&id) = reg.to_id.get(s) {
        return Ok(id);
    }
    let overflow =
        || crate::XlogError::Compilation("symbol registry extent exceeds this host".into());
    let id = u32::try_from(reg.to_string.len()).map_err(|_| overflow())?;
    let next = reg.to_string.len().checked_add(1).ok_or_else(overflow)?;
    let retain_string_owner = reserve.is_some() || !reg.string_owners.is_empty();
    let mut retain = |bytes: usize| -> crate::Result<Option<Arc<dyn Send + Sync>>> {
        reserve.as_mut().map(|reserve| reserve(bytes)).transpose()
    };
    let strings_owner = retain(s.len().checked_mul(2).ok_or_else(overflow)?)?;
    if next > reg.to_id.capacity() {
        // Swiss-table growth keeps one eighth of its power-of-two buckets
        // empty. The control group is at most sixteen bytes on supported hosts.
        let buckets = if next < 4 {
            4
        } else if next < 8 {
            8
        } else if next < 15 {
            16
        } else {
            next.checked_mul(8)
                .ok_or_else(overflow)?
                .checked_div(7)
                .and_then(usize::checked_next_power_of_two)
                .ok_or_else(overflow)?
        };
        let entries = buckets
            .checked_mul(std::mem::size_of::<(String, u32)>())
            .ok_or_else(overflow)?;
        let aligned = entries
            .checked_add(15)
            .map(|bytes| bytes & !15)
            .ok_or_else(overflow)?;
        let bytes = aligned
            .checked_add(buckets)
            .and_then(|bytes| bytes.checked_add(16))
            .ok_or_else(overflow)?;
        let owner = retain(bytes)?;
        reg.to_id.try_reserve(1).map_err(|_| overflow())?;
        reg.table_owner = owner;
    }
    if next > reg.to_string.capacity() {
        let capacity = reg
            .to_string
            .capacity()
            .checked_mul(2)
            .ok_or_else(overflow)?
            .max(next)
            .max(4);
        let bytes = capacity
            .checked_mul(std::mem::size_of::<String>())
            .ok_or_else(overflow)?;
        let owner = retain(bytes)?;
        let additional = capacity
            .checked_sub(reg.to_string.len())
            .ok_or_else(overflow)?;
        reg.to_string
            .try_reserve_exact(additional)
            .map_err(|_| overflow())?;
        reg.strings_array_owner = owner;
    }
    if retain_string_owner {
        let owner_count = reg
            .string_owners
            .len()
            .checked_add(1)
            .ok_or_else(overflow)?;
        if owner_count > reg.string_owners.capacity() {
            let capacity = reg
                .string_owners
                .capacity()
                .checked_mul(2)
                .ok_or_else(overflow)?
                .max(owner_count)
                .max(4);
            let bytes = capacity
                .checked_mul(std::mem::size_of::<Option<Arc<dyn Send + Sync>>>())
                .ok_or_else(overflow)?;
            let owner = retain(bytes)?;
            let additional = capacity
                .checked_sub(reg.string_owners.len())
                .ok_or_else(overflow)?;
            reg.string_owners
                .try_reserve_exact(additional)
                .map_err(|_| overflow())?;
            reg.owners_array_owner = owner;
        }
        reg.string_owners.push(strings_owner);
    }
    let owned = s.to_string();
    reg.to_id.insert(owned.clone(), id);
    reg.to_string.push(owned);
    Ok(id)
}

/// Resolve an ID to its string. Panics if ID is invalid.
pub fn resolve(id: u32) -> String {
    resolve_checked(id).expect("invalid symbol ID: this is a bug")
}

/// Resolve an ID to its string if present.
pub fn resolve_checked(id: u32) -> Option<String> {
    resolve_owned(id, None)
        .expect("ordinary symbol read has no allocation callback")
        .map(|(text, _)| text)
}

/// Copy the original symbol text only after its real UTF-8 extent is reserved
/// under the same registry read lock. The returned owner follows that copy's
/// actual consumer lifetime; it does not grant symbol or program admission.
pub fn resolve_with_allocation_owner(
    id: u32,
    reserve: &mut dyn FnMut(usize) -> crate::Result<Arc<dyn Send + Sync>>,
) -> crate::Result<(String, Arc<dyn Send + Sync>)> {
    let (text, owner) = resolve_owned(id, Some(reserve))?.ok_or_else(|| {
        crate::XlogError::Compilation("invalid symbol ID in original owned read".into())
    })?;
    Ok((text, owner.expect("original symbol allocation owner")))
}

fn resolve_owned(
    id: u32,
    reserve: Option<&mut dyn FnMut(usize) -> crate::Result<Arc<dyn Send + Sync>>>,
) -> crate::Result<Option<(String, Option<Arc<dyn Send + Sync>>)>> {
    let reg = registry().read().unwrap();
    let Some(text) = reg.to_string.get(id as usize) else {
        return Ok(None);
    };
    let owner = reserve.map(|reserve| reserve(text.len())).transpose()?;
    Ok(Some((text.clone(), owner)))
}

/// Owned symbol meanings copied together from one registry state.
///
/// Entries retain request order and duplicates. Their text does not change when
/// the registry is cleared or an ID is reused. This authenticates the meaning at
/// snapshot time, not the provenance of an ID supplied before that time.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SymbolSnapshot {
    entries: Box<[(u32, Box<str>)]>,
}

impl SymbolSnapshot {
    /// Returns immutable `(registry ID, accepted UTF-8 text)` entries.
    pub fn entries(&self) -> &[(u32, Box<str>)] {
        &self.entries
    }
}

/// A complete symbol snapshot could not be admitted within its caller's bounds.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum SymbolSnapshotError {
    /// A requested ID is absent from the locked registry state.
    MissingSymbol {
        /// Missing registry ID.
        id: u32,
    },
    /// The request contains too many entries, including duplicates.
    EntryLimit {
        /// Number of requested entries.
        requested: usize,
        /// Maximum admitted entries.
        limit: usize,
    },
    /// Accepted text would exceed the UTF-8 byte budget.
    ByteLimit {
        /// Maximum admitted UTF-8 bytes, counting duplicates.
        limit: usize,
    },
    /// A writer previously panicked while holding the registry lock.
    RegistryPoisoned,
    /// The original allocation owner refused snapshot storage.
    AllocationRejected {
        /// Original allocation diagnostic.
        detail: String,
    },
}

impl std::fmt::Display for SymbolSnapshotError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::MissingSymbol { id } => write!(f, "symbol ID {id} is not registered"),
            Self::EntryLimit { requested, limit } => {
                write!(
                    f,
                    "symbol snapshot has {requested} entries, exceeding {limit}"
                )
            }
            Self::ByteLimit { limit } => {
                write!(f, "symbol snapshot UTF-8 bytes exceed {limit}")
            }
            Self::RegistryPoisoned => write!(f, "symbol registry lock is poisoned"),
            Self::AllocationRejected { detail } => f.write_str(detail),
        }
    }
}

impl std::error::Error for SymbolSnapshotError {}

/// Copies all requested symbols under one read lock, or returns no snapshot.
///
/// Count and UTF-8 byte bounds include duplicates. All IDs and byte lengths are
/// checked before copying any text; no registry entry is added or renumbered.
pub fn snapshot_checked(
    ids: &[u32],
    max_entries: usize,
    max_utf8_bytes: usize,
) -> Result<SymbolSnapshot, SymbolSnapshotError> {
    snapshot_owned(ids, max_entries, max_utf8_bytes, None).map(|(snapshot, _)| snapshot)
}

/// Reserve actual snapshot storage under the same registry lock before copying.
/// The returned owner must follow the snapshot's real consumer lifetime.
pub fn snapshot_checked_with_allocation_owner(
    ids: &[u32],
    max_entries: usize,
    max_utf8_bytes: usize,
    reserve: &mut dyn FnMut(usize, usize) -> Result<Arc<dyn Send + Sync>, SymbolSnapshotError>,
) -> Result<(SymbolSnapshot, Arc<dyn Send + Sync>), SymbolSnapshotError> {
    let (snapshot, owner) = snapshot_owned(ids, max_entries, max_utf8_bytes, Some(reserve))?;
    Ok((snapshot, owner.expect("original snapshot allocation owner")))
}

fn snapshot_owned(
    ids: &[u32],
    max_entries: usize,
    max_utf8_bytes: usize,
    reserve: Option<
        &mut dyn FnMut(usize, usize) -> Result<Arc<dyn Send + Sync>, SymbolSnapshotError>,
    >,
) -> Result<(SymbolSnapshot, Option<Arc<dyn Send + Sync>>), SymbolSnapshotError> {
    if ids.len() > max_entries {
        return Err(SymbolSnapshotError::EntryLimit {
            requested: ids.len(),
            limit: max_entries,
        });
    }
    let reg = registry()
        .read()
        .map_err(|_| SymbolSnapshotError::RegistryPoisoned)?;
    let mut remaining = max_utf8_bytes;
    for &id in ids {
        let text = reg
            .to_string
            .get(id as usize)
            .ok_or(SymbolSnapshotError::MissingSymbol { id })?;
        remaining = remaining
            .checked_sub(text.len())
            .ok_or(SymbolSnapshotError::ByteLimit {
                limit: max_utf8_bytes,
            })?;
    }
    let owner = reserve
        .map(|reserve| reserve(ids.len(), max_utf8_bytes - remaining))
        .transpose()?;
    let entries = ids
        .iter()
        .map(|&id| (id, reg.to_string[id as usize].clone().into_boxed_str()))
        .collect();
    Ok((SymbolSnapshot { entries }, owner))
}

/// Clear all symbols. For testing/REPL only.
/// WARNING: Invalidates all existing symbol IDs.
pub fn clear() {
    let mut reg = registry().write().unwrap();
    reg.to_id.clear();
    reg.to_string.clear();
    reg.string_owners.clear();
}

/// Number of interned symbols.
pub fn count() -> usize {
    registry().read().unwrap().to_string.len()
}

/// Estimated memory usage in bytes.
pub fn memory_usage() -> usize {
    let reg = registry().read().unwrap();
    let string_bytes: usize = reg.to_string.iter().map(|s| s.len()).sum();
    let map_overhead =
        reg.to_id.len() * (std::mem::size_of::<String>() + std::mem::size_of::<u32>());
    string_bytes + map_overhead
}

/// Convert a column of symbol IDs to Arrow DictionaryArray.
pub fn to_arrow(ids: &[u32]) -> DictionaryArray<UInt32Type> {
    use std::collections::HashSet;

    // Collect unique IDs preserving order
    let mut seen = HashSet::new();
    let unique_ids: Vec<u32> = ids.iter().filter(|id| seen.insert(**id)).copied().collect();

    // Build string dictionary
    let dict_strings: Vec<String> = unique_ids.iter().map(|&id| resolve(id)).collect();
    let dictionary = StringArray::from(dict_strings);

    // Map original IDs to dictionary indices
    let id_to_index: HashMap<u32, u32> = unique_ids
        .iter()
        .enumerate()
        .map(|(i, &id)| (id, i as u32))
        .collect();

    let keys: Vec<u32> = ids.iter().map(|id| *id_to_index.get(id).unwrap()).collect();
    let keys_array = UInt32Array::from(keys);

    DictionaryArray::try_new(keys_array, Arc::new(dictionary)).unwrap()
}

/// Convert Arrow DictionaryArray back to symbol IDs.
pub fn from_arrow(arr: &DictionaryArray<UInt32Type>) -> Vec<u32> {
    let dict = arr
        .values()
        .as_any()
        .downcast_ref::<StringArray>()
        .expect("dictionary values must be StringArray");

    // Intern all dictionary values
    let dict_to_symbol: Vec<u32> = dict
        .iter()
        .map(|s| intern(s.expect("null not supported in symbols")))
        .collect();

    // Map keys through dictionary
    arr.keys()
        .iter()
        .map(|k| {
            let idx = k.expect("null keys not supported") as usize;
            dict_to_symbol[idx]
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use serial_test::serial;

    // Each test must call setup() to get clean state
    fn setup() {
        clear();
    }

    #[test]
    #[serial]
    fn snapshot_preserves_order_duplicates_and_utf8_limits() {
        setup();
        let empty = intern("");
        let unicode = intern("é🎉");
        let ids = [unicode, empty, unicode];
        let snapshot = snapshot_checked(&ids, 3, 12).unwrap();
        let entries: Vec<_> = snapshot
            .entries()
            .iter()
            .map(|(id, text)| (*id, text.as_ref()))
            .collect();
        assert_eq!(entries, [(unicode, "é🎉"), (empty, ""), (unicode, "é🎉")]);
        assert_eq!(
            snapshot_checked(&ids, 2, 12),
            Err(SymbolSnapshotError::EntryLimit {
                requested: 3,
                limit: 2
            })
        );
        assert_eq!(
            snapshot_checked(&ids, 3, 11),
            Err(SymbolSnapshotError::ByteLimit { limit: 11 })
        );
        assert!(snapshot_checked(&[], 0, 0).unwrap().entries().is_empty());
    }

    #[test]
    #[serial]
    fn snapshot_missing_symbol_rejects_whole_request_without_mutation() {
        setup();
        let valid = intern("present");
        assert_eq!(
            snapshot_checked(&[valid, 9, valid], 3, 100),
            Err(SymbolSnapshotError::MissingSymbol { id: 9 })
        );
        assert_eq!(count(), 1);
        assert_eq!(resolve(valid), "present");
    }

    #[test]
    #[serial]
    fn snapshot_survives_clear_and_reintern_without_reinterpreting_ids() {
        setup();
        let id = intern("old");
        let accepted = snapshot_checked(&[id], 1, 3).unwrap();
        clear();
        assert_eq!(intern("new"), id);
        let current = snapshot_checked(&[id], 1, 3).unwrap();
        assert_eq!(accepted.entries()[0].1.as_ref(), "old");
        assert_eq!(current.entries()[0].1.as_ref(), "new");
        assert_ne!(accepted, current);
    }

    #[test]
    #[serial]
    fn snapshot_never_tears_across_concurrent_registry_replacement() {
        setup();
        intern("old-left");
        intern("old-right");
        let start = std::sync::Barrier::new(2);
        std::thread::scope(|scope| {
            let writer = scope.spawn(|| {
                start.wait();
                for index in 0..1000 {
                    clear();
                    let (left, right) = if index % 2 == 0 {
                        ("new-left", "new-right")
                    } else {
                        ("old-left", "old-right")
                    };
                    intern(left);
                    intern(right);
                }
            });
            start.wait();
            for _ in 0..1000 {
                match snapshot_checked(&[0, 1], 2, 18) {
                    Ok(snapshot) => {
                        let entries = snapshot.entries();
                        let pair = (entries[0].1.as_ref(), entries[1].1.as_ref());
                        assert!(
                            pair == ("old-left", "old-right") || pair == ("new-left", "new-right"),
                            "torn snapshot: {pair:?}"
                        );
                    }
                    Err(SymbolSnapshotError::MissingSymbol { .. }) => {}
                    other => panic!("unexpected snapshot result: {other:?}"),
                }
            }
            writer.join().unwrap();
            let final_snapshot = snapshot_checked(&[0, 1], 2, 18).unwrap();
            assert_eq!(final_snapshot.entries()[0].1.as_ref(), "old-left");
            assert_eq!(final_snapshot.entries()[1].1.as_ref(), "old-right");
        });
    }

    #[test]
    #[serial]
    fn test_intern_sequential() {
        setup();
        assert_eq!(intern("foo"), 0);
        assert_eq!(intern("bar"), 1);
        assert_eq!(intern("baz"), 2);
    }

    #[test]
    #[serial]
    fn test_intern_idempotent() {
        setup();
        let id1 = intern("hello");
        let id2 = intern("hello");
        assert_eq!(id1, id2);
        assert_eq!(count(), 1); // only one entry
    }

    #[test]
    #[serial]
    fn test_resolve_roundtrip() {
        setup();
        let id = intern("world");
        assert_eq!(resolve(id), "world");
    }

    #[test]
    #[serial]
    #[should_panic(expected = "invalid symbol ID")]
    fn test_resolve_invalid() {
        setup();
        resolve(9999); // should panic
    }

    #[test]
    #[serial]
    fn test_clear() {
        setup();
        intern("a");
        intern("b");
        assert_eq!(count(), 2);
        clear();
        assert_eq!(count(), 0);
        assert_eq!(intern("a"), 0); // IDs restart from 0
    }

    #[test]
    #[serial]
    fn test_empty_string() {
        setup();
        let id = intern("");
        assert_eq!(resolve(id), "");
    }

    #[test]
    #[serial]
    fn test_unicode() {
        setup();
        let id = intern("日本語");
        assert_eq!(resolve(id), "日本語");

        let id2 = intern("émoji🎉");
        assert_eq!(resolve(id2), "émoji🎉");
    }

    #[test]
    #[serial]
    fn test_concurrent_intern() {
        setup();
        use std::collections::HashSet;
        use std::thread;

        let handles: Vec<_> = (0..10)
            .map(|i| {
                thread::spawn(move || {
                    let mut ids = Vec::new();
                    for j in 0..100 {
                        let s = format!("thread{}_{}", i, j);
                        let id = intern(&s);
                        ids.push((s, id));
                    }
                    ids
                })
            })
            .collect();

        let mut all_results = Vec::new();
        for h in handles {
            all_results.extend(h.join().unwrap());
        }

        // Verify all symbols resolve correctly
        for (s, id) in &all_results {
            assert_eq!(&resolve(*id), s);
        }

        // Verify we have 1000 unique symbols
        assert_eq!(count(), 1000);

        // Verify no duplicate IDs for different strings
        let unique_ids: HashSet<u32> = all_results.iter().map(|(_, id)| *id).collect();
        assert_eq!(unique_ids.len(), 1000);
    }

    #[test]
    #[serial]
    fn test_large_scale() {
        setup();
        use std::time::Instant;

        let start = Instant::now();

        // Intern 100K unique symbols
        for i in 0..100_000 {
            let s = format!("symbol_{:06}", i);
            let id = intern(&s);
            assert_eq!(id, i as u32);
        }

        let intern_time = start.elapsed();

        // Verify all resolve correctly
        let start = Instant::now();
        for i in 0..100_000 {
            let expected = format!("symbol_{:06}", i);
            assert_eq!(resolve(i as u32), expected);
        }
        let resolve_time = start.elapsed();

        // Verify count
        assert_eq!(count(), 100_000);

        // Log performance (not assertions, just info)
        println!(
            "100K intern: {:?}, 100K resolve: {:?}",
            intern_time, resolve_time
        );

        // Memory should be reasonable (rough check: < 10MB for 100K symbols)
        let mem = memory_usage();
        assert!(mem < 10_000_000, "memory usage {} exceeds 10MB", mem);
    }

    #[test]
    #[serial]
    fn test_arrow_roundtrip() {
        setup();
        let ids = vec![
            intern("apple"),
            intern("banana"),
            intern("apple"),
            intern("cherry"),
            intern("banana"),
        ];

        let arrow = to_arrow(&ids);
        let back = from_arrow(&arrow);

        assert_eq!(ids, back);
    }
}
