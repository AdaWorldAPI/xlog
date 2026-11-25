//! Does demand decouple query work from irrelevant graph size? (EXO-457)
//!
//! One demanded chain `0 -> 1 -> .. -> L-1` stays fixed; `k` disconnected
//! chains of the same length are added. `?- reach(0, Y)` runs with and
//! without magic sets. Reported per run: derived `reach` rows, recursive
//! rounds, cube programs, rows the cube materialized, heap allocations and
//! peak heap (a counting allocator around this process), and wall time in
//! µs (median of `RUNS`). Candidates visited inside the cube's walk are not
//! counted by the cube, so they are not reported.
//!
//! `cargo run --release --example magic_scaling`

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicUsize, Ordering::Relaxed};
use std::time::Instant;

use xlog_cube::CubeExecutor;
use xlog_logic::Compiler;

struct Counting;
static ALLOCS: AtomicUsize = AtomicUsize::new(0);
static LIVE: AtomicUsize = AtomicUsize::new(0);
static PEAK: AtomicUsize = AtomicUsize::new(0);

unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, l: Layout) -> *mut u8 {
        ALLOCS.fetch_add(1, Relaxed);
        let live = LIVE.fetch_add(l.size(), Relaxed) + l.size();
        PEAK.fetch_max(live, Relaxed);
        // SAFETY: forwarded unchanged to the system allocator.
        unsafe { System.alloc(l) }
    }
    unsafe fn dealloc(&self, p: *mut u8, l: Layout) {
        LIVE.fetch_sub(l.size(), Relaxed);
        // SAFETY: `p` came from `alloc` above with the same layout.
        unsafe { System.dealloc(p, l) }
    }
}

#[global_allocator]
static GLOBAL: Counting = Counting;

const L: u32 = 8;
const RUNS: usize = 15;

const BODY: &str = r#"
    pred edge(u32, u32). pred reach(u32, u32).
    reach(X, Y) :- edge(X, Y).
    reach(X, Z) :- reach(X, Y), edge(Y, Z).
    ?- reach(0, Y).
"#;

fn edges(k: u32) -> Vec<(u32, u32)> {
    (0..=k)
        .flat_map(|c| (0..L - 1).map(move |i| (c * L + i, c * L + i + 1)))
        .collect()
}

struct Row {
    reach: usize,
    rounds: usize,
    programs: u64,
    materialized: u64,
    allocs: usize,
    peak_kib: usize,
    us: f64,
}

fn once(src: &str, magic: bool, e: &[(u32, u32)]) -> Row {
    let mut c = Compiler::new();
    let plan = c.compile(src).expect("compile");
    let mut ex = CubeExecutor::new();
    for (name, id) in c.rel_ids() {
        ex.register_relation(*id, name);
    }
    ex.put_relation("edge", e);
    if magic {
        let p = xlog_logic::parse_program(src).expect("parse");
        ex.demand_magic_rewrite(&xlog_logic::rewrite_magic_sets(&p).expect("rewrite"));
    }
    ALLOCS.store(0, Relaxed);
    PEAK.store(LIVE.load(Relaxed), Relaxed);
    let base = LIVE.load(Relaxed);
    let t = Instant::now();
    let rep = ex.execute_plan(&plan).expect("plan");
    let us = t.elapsed().as_secs_f64() * 1e6;
    let (allocs, peak) = (ALLOCS.load(Relaxed), PEAK.load(Relaxed) - base);
    assert!(rep.refused.is_empty(), "{:?}", rep.refused);
    let reach = ex.relation("reach").unwrap_or_default().len();
    let answer = ex.relation("__xlog_query_0").unwrap_or_default();
    assert_eq!(answer.len(), (L - 1) as usize, "query answer");
    Row {
        reach,
        rounds: rep.rounds,
        programs: rep.stats.programs,
        materialized: rep.stats.rows_materialized,
        allocs,
        peak_kib: peak / 1024,
        us,
    }
}

fn median(src: &str, magic: bool, e: &[(u32, u32)]) -> Row {
    let mut rows: Vec<Row> = (0..RUNS).map(|_| once(src, magic, e)).collect();
    rows.sort_by(|a, b| a.us.total_cmp(&b.us));
    rows.swap_remove(RUNS / 2)
}

fn main() {
    let magic_src = format!("#pragma magic_sets = on\n{BODY}");
    println!("k\tmode\treach_rows\trounds\tprograms\trows_materialized\tallocs\tpeak_KiB\tus");
    for k in [1u32, 2, 4, 8, 16] {
        let e = edges(k);
        for (mode, src, magic) in [("plain", BODY, false), ("magic", magic_src.as_str(), true)] {
            let r = median(src, magic, &e);
            println!(
                "{k}\t{mode}\t{}\t{}\t{}\t{}\t{}\t{}\t{:.1}",
                r.reach, r.rounds, r.programs, r.materialized, r.allocs, r.peak_kib, r.us
            );
        }
    }
}
