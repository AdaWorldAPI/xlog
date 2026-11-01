//! C0: three-way closure parity (lance-graph D-STAUNEN-5, P1 acceptance).
//!
//! One edge relation, three executions of `edge⁺`:
//!
//! 1. a scalar breadth-first oracle written here, sharing no code with either
//!    engine;
//! 2. `lance_graph_xlog::cube::Cube::closure`, the resident mask closure;
//! 3. XLOG's compiled recursive rules run by [`CubeExecutor`].
//!
//! All three are compared as **sets** of global-id pairs. The input carries
//! duplicate edges on purpose, and XLOG's output is checked for duplicate
//! rows before it is turned into a set, so bag-versus-set behaviour is tested
//! rather than hidden by the comparison. Nothing here concerns learning: a
//! pass says the three closures agree, and nothing more.

use std::collections::{BTreeMap, BTreeSet};

use lance_graph_dt64::dr::{LocalMap, MapTable, Width};
use lance_graph_xlog::cube::Cube;
use xlog_cube::CubeExecutor;
use xlog_logic::Compiler;

type Pairs = BTreeSet<(u32, u32)>;

const TILE: u32 = 5;
const REACH: &str = "reach(X,Y) :- edge(X,Y). reach(X,Z) :- reach(X,Y), edge(Y,Z).";

/// (1) Breadth-first from every source. Global ids in, global ids out.
fn oracle(edges: &[(u32, u32)]) -> Pairs {
    let mut succ: BTreeMap<u32, Vec<u32>> = BTreeMap::new();
    for &(a, b) in edges {
        succ.entry(a).or_default().push(b);
    }
    let mut out = Pairs::new();
    for &s in succ.keys() {
        let mut stack = succ[&s].clone();
        let mut seen = BTreeSet::new();
        while let Some(v) = stack.pop() {
            if seen.insert(v) {
                out.insert((s, v));
                stack.extend(succ.get(&v).into_iter().flatten());
            }
        }
    }
    out
}

/// (2) The resident closure, over a map of the relation's sorted ids; local
/// positions are mapped back to global ids through that same map.
fn resident(edges: &[(u32, u32)]) -> Pairs {
    let mut ids: Vec<u64> = edges
        .iter()
        .flat_map(|&(a, b)| [u64::from(a), u64::from(b)])
        .collect();
    ids.sort_unstable();
    ids.dedup();
    let mut table = MapTable::new(1, 1);
    table.advance(
        1,
        1,
        vec![(TILE, Some(LocalMap::prepare(Width::W64, &ids).unwrap()))],
    );
    let rels: [&[(u32, u32)]; 1] = [edges];
    let cube = Cube::new(&table, &table.seal(), TILE, &rels).unwrap();
    let (closure, _) = cube.closure(&table, 0).unwrap();
    let map = table.map(&table.seal(), TILE).unwrap();
    let global = |l: usize| map.global(l as u8).unwrap() as u32;
    let mut out = Pairs::new();
    for (x, &row) in closure.rows().iter().enumerate() {
        for y in 0..64 {
            if row >> y & 1 == 1 {
                out.insert((global(x), global(y)));
            }
        }
    }
    out
}

/// (3) XLOG's compiled recursion. Returns the set, after checking that the
/// executor produced no duplicate rows and only binary ones.
fn xlog(edges: &[(u32, u32)]) -> Pairs {
    let mut c = Compiler::new();
    let plan = c.compile(REACH).expect("compile");
    let mut ex = CubeExecutor::new();
    for (name, id) in c.rel_ids() {
        ex.register_relation(*id, name);
    }
    ex.put_relation("edge", edges);
    let report = ex.execute_plan(&plan).expect("plan");
    assert!(report.refused.is_empty(), "{report:?}");
    let rows = ex.relation("reach").unwrap_or_default();
    let set: Pairs = rows
        .iter()
        .map(|r| match r.as_slice() {
            [a, b] => (*a, *b),
            other => panic!("non-binary reach row {other:?}"),
        })
        .collect();
    assert_eq!(set.len(), rows.len(), "XLOG returned duplicate reach rows");
    set
}

/// The comparison every test uses: all three must equal each other.
fn parity(edges: &[(u32, u32)], resident: &Pairs) -> Result<Pairs, String> {
    let (o, x) = (oracle(edges), xlog(edges));
    if &o != resident {
        return Err(format!("resident differs from oracle on {edges:?}"));
    }
    if o != x {
        return Err(format!("xlog differs from oracle on {edges:?}"));
    }
    Ok(o)
}

struct Rng(u64);
impl Rng {
    fn next(&mut self, m: u32) -> u32 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        (self.0 % u64::from(m)) as u32
    }
}

/// The canonical fixture: a 3-cycle with an exit. Exactly 12 pairs; the
/// diagonal is `(0,0)`, `(1,1)`, `(2,2)`, and `(3,3)` is absent. Checked on
/// each of the three separately, not only through their agreement.
#[test]
fn the_canonical_cycle_agrees_and_has_its_three_diagonal_pairs() {
    let edges = [(0, 1), (1, 2), (2, 0), (2, 3)];
    let diagonal: Pairs = [(0, 0), (1, 1), (2, 2)].into();
    for (name, got) in [
        ("oracle", oracle(&edges)),
        ("resident", resident(&edges)),
        ("xlog", xlog(&edges)),
    ] {
        assert_eq!(got.len(), 12, "{name}");
        let diag: Pairs = got.iter().filter(|(a, b)| a == b).copied().collect();
        assert_eq!(diag, diagonal, "{name}");
    }
    parity(&edges, &resident(&edges)).unwrap();
}

/// 400 frozen seeded graphs: ids 0..64, cyclic and acyclic, with duplicate
/// edges in the input. The seed is a constant, so the set is the same on
/// every run.
#[test]
fn four_hundred_seeded_graphs_agree() {
    let mut rng = Rng(0xC0C0_5EED_0000_0001);
    let (mut cyclic, mut acyclic, mut dup, mut top) = (0, 0, 0, 0);
    for case in 0..400u32 {
        let n = if case % 4 == 0 { 64 } else { rng.next(64) + 1 };
        let dag = case % 2 == 0;
        let m = rng.next(3 * n) + 1;
        let mut edges: Vec<(u32, u32)> = (0..m)
            .filter_map(|_| {
                let (a, b) = (rng.next(n), rng.next(n));
                (!dag || a < b).then_some((a, b))
            })
            .collect();
        if edges.is_empty() {
            continue;
        }
        // A repeated edge, so the engines see a bag on input.
        edges.push(edges[0]);
        dup += 1;
        let got = parity(&edges, &resident(&edges)).unwrap_or_else(|e| panic!("case {case}: {e}"));
        let has_cycle = got.iter().any(|(a, b)| a == b);
        cyclic += usize::from(has_cycle);
        acyclic += usize::from(!has_cycle);
        top += usize::from(got.iter().any(|&(a, b)| a == 63 || b == 63));
    }
    // Anti-vacuity: both shapes, the duplicate path and id 63 all occur.
    assert!(
        cyclic > 50 && acyclic > 50,
        "{cyclic} cyclic, {acyclic} acyclic"
    );
    assert!(dup > 300, "{dup}");
    assert!(top > 20, "id 63 reached in only {top} cases");
}

/// The comparison can fail: a closure that drops its diagonal, and one that
/// stops after one step, are both caught on the canonical fixture.
#[test]
fn a_broken_closure_is_caught() {
    let edges = [(0, 1), (1, 2), (2, 0), (2, 3)];
    let good = resident(&edges);
    let no_diagonal: Pairs = good.iter().filter(|(a, b)| a != b).copied().collect();
    assert!(parity(&edges, &no_diagonal).is_err());
    let one_step: Pairs = edges.iter().copied().collect();
    assert!(parity(&edges, &one_step).is_err());
    assert!(parity(&edges, &good).is_ok());
}
