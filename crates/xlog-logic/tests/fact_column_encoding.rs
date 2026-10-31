//! Fact rows are encoded into byte columns and deduplicated in place.
//!
//! The previous loader copied each row's bytes into an owned key and kept every
//! distinct row a second time in a set. `encode_fact_columns` compares rows
//! where they lie in the columns, so its allocation count depends on the number
//! of columns, not on the number of rows. These tests pin both the output
//! (against the previous algorithm) and that allocation shape.

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;
use std::collections::HashSet;

use xlog_core::ScalarType;
use xlog_logic::ground_term_encoding::{append_ground_term_bytes, encode_fact_columns};
use xlog_logic::Term;

struct CountingAlloc;

thread_local! {
    static ALLOCATIONS: Cell<usize> = const { Cell::new(0) };
}

unsafe impl GlobalAlloc for CountingAlloc {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        ALLOCATIONS.with(|count| count.set(count.get() + 1));
        unsafe { System.alloc(layout) }
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { System.dealloc(ptr, layout) }
    }
}

#[global_allocator]
static GLOBAL: CountingAlloc = CountingAlloc;

fn allocations_during<T>(run: impl FnOnce() -> T) -> (T, usize) {
    let before = ALLOCATIONS.with(Cell::get);
    let result = run();
    (result, ALLOCATIONS.with(Cell::get) - before)
}

/// The previous loader's per-row algorithm: encode into the columns, copy the
/// row's bytes into an owned key, and keep the key in a set.
fn previous_encoding(
    rows: &[&[Term]],
    column_types: &[ScalarType],
    dedup_rows: bool,
) -> (Vec<Vec<u8>>, usize) {
    let row_size: usize = column_types.iter().map(ScalarType::size_bytes).sum();
    let mut seen = dedup_rows.then(|| HashSet::with_capacity(rows.len()));
    let mut columns: Vec<Vec<u8>> = vec![Vec::new(); column_types.len()];
    let mut starts = Vec::with_capacity(column_types.len());
    let mut written = 0;
    for row in rows {
        starts.clear();
        let mut key = Vec::with_capacity(row_size);
        for (index, term) in row.iter().enumerate() {
            starts.push(columns[index].len());
            append_ground_term_bytes(&mut columns[index], term, column_types[index]).unwrap();
            if seen.is_some() {
                key.extend_from_slice(&columns[index][starts[index]..]);
            }
        }
        if seen.as_mut().is_some_and(|set| !set.insert(key)) {
            for (column, start) in columns.iter_mut().zip(&starts) {
                column.truncate(*start);
            }
        } else {
            written += 1;
        }
    }
    (columns, written)
}

/// `n` rows over (u32, i64, f64): one row in ten repeats an earlier row, and
/// the float column alternates +0.0 and -0.0 so equal-valued rows with
/// different bytes stay distinct.
fn fixture(n: usize) -> Vec<Vec<Term>> {
    (0..n)
        .map(|i| {
            let k = if i % 10 == 9 { i / 2 } else { i };
            let zero = if k % 2 == 0 { 0.0 } else { -0.0 };
            vec![
                Term::Integer((k % 4096) as i64),
                Term::Integer(k as i64 - 50),
                Term::Float(zero),
            ]
        })
        .collect()
}

const TYPES: [ScalarType; 3] = [ScalarType::U32, ScalarType::I64, ScalarType::F64];

#[test]
fn matches_the_previous_encoding_with_and_without_deduplication() {
    let owned = fixture(5_000);
    let rows: Vec<&[Term]> = owned.iter().map(Vec::as_slice).collect();
    for dedup in [true, false] {
        let expected = previous_encoding(&rows, &TYPES, dedup);
        let actual = encode_fact_columns(&rows, &TYPES, dedup).unwrap();
        assert_eq!(actual, expected, "dedup = {dedup}");
    }
    let (_, kept) = encode_fact_columns(&rows, &TYPES, true).unwrap();
    assert!(kept < rows.len(), "the fixture must contain duplicates");
}

#[test]
fn a_bad_term_reports_its_row_and_column() {
    let owned = vec![
        vec![Term::Integer(1), Term::Integer(2), Term::Float(0.5)],
        vec![Term::Integer(-1), Term::Integer(2), Term::Float(0.5)],
    ];
    let rows: Vec<&[Term]> = owned.iter().map(Vec::as_slice).collect();
    let error = encode_fact_columns(&rows, &TYPES, true).unwrap_err();
    assert_eq!((error.row, error.column), (1, 0));
}

#[test]
fn allocations_do_not_grow_with_the_number_of_rows() {
    let small_owned = fixture(1_000);
    let large_owned = fixture(100_000);
    let small: Vec<&[Term]> = small_owned.iter().map(Vec::as_slice).collect();
    let large: Vec<&[Term]> = large_owned.iter().map(Vec::as_slice).collect();

    let ((_, kept), small_allocs) =
        allocations_during(|| encode_fact_columns(&small, &TYPES, true).unwrap());
    assert!(kept < small.len(), "the fixture must contain duplicates");
    let (_, large_allocs) =
        allocations_during(|| encode_fact_columns(&large, &TYPES, true).unwrap());
    assert_eq!(small_allocs, large_allocs);
    // One vector per column, the column list, the column widths and the
    // position table.
    assert_eq!(large_allocs, TYPES.len() + 3, "{large_allocs} allocations");

    // The previous algorithm allocates per row, so this measurement can fail.
    let (_, previous_allocs) = allocations_during(|| previous_encoding(&large, &TYPES, true));
    assert!(
        previous_allocs > large.len() / 2,
        "{previous_allocs} allocations"
    );
}
