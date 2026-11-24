//! D-XLOG-W5-C: one-column heads routed to cube set expressions. The
//! acceptance target is the G1c witness of `replay_reachable`
//! (lance-graph `crates/xlog-coverage-probe/tests/witness.rs`).

use std::collections::BTreeSet;

use xlog_cube::{CubeExecutor, Report};
use xlog_logic::Compiler;

const REPLAY: &str = r#"
    pred frontier_pred(u32).
    pred widened_pred(u32).
    pred frontier_edge(u32, u32).
    pred blocked_pred(u32).
    pred promoted(u32).
    pred replay_reachable(u32).
    pred rollback_hit(u32).

    promoted(P) :- frontier_pred(P), widened_pred(P).
    replay_reachable(P) :- promoted(P).
    replay_reachable(Q) :- replay_reachable(P), frontier_edge(P, Q), frontier_pred(Q).
    rollback_hit(P) :- replay_reachable(P), blocked_pred(P).
"#;

struct Input<'a> {
    sets: &'a [(&'a str, &'a [u32])],
    pairs: &'a [(&'a str, &'a [(u32, u32)])],
}

fn run(src: &str, input: Input<'_>) -> (CubeExecutor, Report) {
    let mut c = Compiler::new();
    let plan = c.compile(src).expect("compile");
    let mut ex = CubeExecutor::new();
    for (name, id) in c.rel_ids() {
        ex.register_relation(*id, name);
    }
    for (n, ids) in input.sets {
        ex.put_set(n, ids);
    }
    for (n, rows) in input.pairs {
        ex.put_relation(n, rows);
    }
    let report = ex.execute_plan(&plan).expect("plan");
    (ex, report)
}

fn rows(ex: &CubeExecutor, name: &str) -> BTreeSet<Vec<u32>> {
    ex.relation(name).unwrap_or_default().into_iter().collect()
}

fn ones(v: &[u32]) -> BTreeSet<Vec<u32>> {
    v.iter().map(|&x| vec![x]).collect()
}

fn routed(r: &Report) -> Vec<(&str, &str)> {
    r.routed
        .iter()
        .map(|(h, p)| (h.as_str(), p.as_str()))
        .collect()
}

fn replay(blocked: &[u32]) -> (CubeExecutor, Report) {
    run(
        REPLAY,
        Input {
            sets: &[
                ("frontier_pred", &[1, 2, 3]),
                ("widened_pred", &[1]),
                ("blocked_pred", blocked),
            ],
            pairs: &[("frontier_edge", &[(1, 2), (2, 3), (3, 4)])],
        },
    )
}

#[test]
fn replay_reachable_runs_on_the_cube_with_the_g1c_answer() {
    let (ex, report) = replay(&[3]);
    assert_eq!(rows(&ex, "promoted"), ones(&[1]));
    assert_eq!(
        rows(&ex, "replay_reachable"),
        ones(&[1, 2, 3]),
        "4 is not on the frontier"
    );
    assert_eq!(rows(&ex, "rollback_hit"), ones(&[3]));
    assert_eq!(
        routed(&report),
        [
            ("promoted", "set"),
            ("replay_reachable", "reach"),
            ("rollback_hit", "set")
        ]
    );
    assert!(report.refused.is_empty(), "{:?}", report.refused);
    // The answers are set expressions on the cube: no host round, no
    // mask-risc program, no relation added for a head.
    assert_eq!(report.rounds, 0);
    assert_eq!(report.stats.programs, 0);
    assert_eq!(report.stats.relations_added, 0);
}

#[test]
fn blocking_an_unreached_node_leaves_rollback_empty() {
    let (ex, report) = replay(&[4]);
    assert_eq!(rows(&ex, "replay_reachable"), ones(&[1, 2, 3]));
    assert!(rows(&ex, "rollback_hit").is_empty());
    assert!(report.refused.is_empty());
}

fn rng(mut s: u64) -> impl FnMut(u64) -> u64 {
    move |m| {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        s % m
    }
}

/// The routed answer against a breadth-first search over random inputs.
#[test]
fn routed_replay_agrees_with_bfs() {
    let mut grew = 0;
    for seed in 1..=30u64 {
        let mut next = rng(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15));
        let n = 2 + next(40) as u32;
        let pick = |next: &mut dyn FnMut(u64) -> u64, p: u64| -> Vec<u32> {
            (0..n).filter(|_| next(p) == 0).collect()
        };
        let frontier = pick(&mut next, 2);
        let widened = pick(&mut next, 3);
        let blocked = pick(&mut next, 3);
        let m = next(u64::from(3 * n)) as usize;
        let edges: Vec<(u32, u32)> = (0..m)
            .map(|_| (next(u64::from(n)) as u32, next(u64::from(n)) as u32))
            .collect();
        // BFS oracle.
        let promoted: Vec<u32> = frontier
            .iter()
            .copied()
            .filter(|x| widened.contains(x))
            .collect();
        let mut seen: BTreeSet<u32> = promoted.iter().copied().collect();
        let mut stack = promoted.clone();
        while let Some(x) = stack.pop() {
            for &(a, b) in &edges {
                if a == x && frontier.contains(&b) && seen.insert(b) {
                    stack.push(b);
                }
            }
        }
        if seen.len() > promoted.len() {
            grew += 1;
        }
        let want_rb: Vec<u32> = seen
            .iter()
            .copied()
            .filter(|x| blocked.contains(x))
            .collect();
        let want: Vec<u32> = seen.into_iter().collect();

        let (ex, report) = run(
            REPLAY,
            Input {
                sets: &[
                    ("frontier_pred", &frontier),
                    ("widened_pred", &widened),
                    ("blocked_pred", &blocked),
                ],
                pairs: &[("frontier_edge", &edges)],
            },
        );
        assert!(
            report.refused.is_empty(),
            "seed {seed}: {:?}",
            report.refused
        );
        assert_eq!(rows(&ex, "promoted"), ones(&promoted), "seed {seed}");
        assert_eq!(rows(&ex, "replay_reachable"), ones(&want), "seed {seed}");
        assert_eq!(rows(&ex, "rollback_hit"), ones(&want_rb), "seed {seed}");
    }
    assert!(grew >= 10, "only {grew} inputs grew past the seed");
}

/// Facts a host stored for a routed head are part of its answer.
#[test]
fn stored_facts_join_a_routed_head() {
    let src = "pred a(u32). pred p(u32). p(X) :- a(X).";
    let (ex, report) = run(
        src,
        Input {
            sets: &[("a", &[1]), ("p", &[7])],
            pairs: &[],
        },
    );
    assert_eq!(rows(&ex, "p"), ones(&[1, 7]));
    assert_eq!(routed(&report), [("p", "set")]);
}

/// A one-column head the old path produces (here a projection of a binary
/// join) is not a set, so a reader of it is not routed: it is refused, never
/// answered from facts stored for that head alone. (Before EXO-394 this test
/// used `h(X) :- r(X, Y).`, which is now a column leaf and routed; see
/// `a_column_head_keeps_its_stored_facts`.)
#[test]
fn a_reader_of_an_unrouted_head_is_not_routed() {
    let src = r#"
        pred r(u32, u32). pred s(u32). pred h(u32). pred g(u32).
        h(X) :- r(X, Y), r(Y, Z).
        g(X) :- h(X), s(X).
    "#;
    let (ex, report) = run(
        src,
        Input {
            sets: &[("s", &[1, 9]), ("h", &[9])],
            pairs: &[("r", &[(1, 2)])],
        },
    );
    assert!(
        routed(&report).iter().all(|(h, _)| *h != "g"),
        "{:?}",
        report.routed
    );
    assert!(
        report.refused.iter().any(|(h, _)| h == "g"),
        "g read only h's stored facts: {:?} / {:?}",
        rows(&ex, "g"),
        report
    );
}

/// The step must go from the recursive atom's variable to the head's: a
/// reversed edge, or no target filter, is not routed.
#[test]
fn other_recursive_shapes_are_not_routed() {
    let reversed = r#"
        pred f(u32). pred e(u32, u32). pred s(u32). pred r(u32).
        r(P) :- s(P).
        r(Q) :- r(P), e(Q, P), f(Q).
    "#;
    let unfiltered = r#"
        pred e(u32, u32). pred s(u32). pred r(u32).
        r(P) :- s(P).
        r(Q) :- r(P), e(P, Q).
    "#;
    for src in [reversed, unfiltered] {
        let (_, report) = run(
            src,
            Input {
                sets: &[("f", &[1, 2]), ("s", &[1])],
                pairs: &[("e", &[(1, 2)])],
            },
        );
        assert!(
            report.routed.iter().all(|(h, _)| h != "r"),
            "{src}: {:?}",
            report.routed
        );
    }
}

/// EXO-394: `h(X) :- r(X, Y).` is a column leaf, so `h` is routed, and the
/// facts a host stored for `h` stay part of its answer: h = col(r, 0) ∪ {9}.
#[test]
fn a_column_head_keeps_its_stored_facts() {
    let src = r#"
        pred r(u32, u32). pred s(u32). pred h(u32). pred g(u32).
        h(X) :- r(X, Y).
        g(X) :- h(X), s(X).
    "#;
    let (ex, report) = run(
        src,
        Input {
            sets: &[("s", &[1, 9]), ("h", &[9])],
            pairs: &[("r", &[(1, 2)])],
        },
    );
    assert_eq!(rows(&ex, "h"), ones(&[1, 9]));
    assert_eq!(rows(&ex, "g"), ones(&[1, 9]));
    assert!(report.refused.is_empty(), "{:?}", report.refused);
}
