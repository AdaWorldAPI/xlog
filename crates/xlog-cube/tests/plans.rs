//! XLOG programs compiled by XLOG's own compiler, executed on the hypercube,
//! compared with a direct enumeration of each rule over random base
//! relations.

use std::collections::BTreeSet;

use xlog_cube::CubeExecutor;
use xlog_logic::Compiler;

type Rel = Vec<(u32, u32)>;
type Oracle = fn(&Rel, &Rel) -> BTreeSet<Vec<u32>>;

fn run(src: &str, r: &Rel, s: &Rel) -> (CubeExecutor, xlog_cube::Report) {
    let mut c = Compiler::new();
    let plan = c.compile(src).expect("compile");
    let mut ex = CubeExecutor::new();
    for (name, id) in c.rel_ids() {
        ex.register_relation(*id, name);
    }
    ex.put_relation("r", r);
    ex.put_relation("s", s);
    let report = ex.execute_plan(&plan).expect("plan");
    (ex, report)
}

fn rows(ex: &CubeExecutor, name: &str) -> BTreeSet<Vec<u32>> {
    ex.relation(name).unwrap_or_default().into_iter().collect()
}

fn random(seed: &mut u64, n: u32) -> Rel {
    let mut next = |m: u32| {
        *seed ^= *seed << 13;
        *seed ^= *seed >> 7;
        *seed ^= *seed << 17;
        (*seed % u64::from(m)) as u32
    };
    let len = next(n) + 1;
    (0..len).map(|_| (next(10), next(10))).collect()
}

const PROGRAMS: [(&str, &str, Oracle); 6] = [
    ("q(X,Z) :- r(X,Y), s(Y,Z).", "q", |r, s| {
        r.iter()
            .flat_map(|&(x, y)| {
                s.iter()
                    .filter(move |t| t.0 == y)
                    .map(move |&(_, z)| vec![x, z])
            })
            .collect()
    }),
    ("q(Y) :- r(X,Y), s(Y,Z).", "q", |r, s| {
        r.iter()
            .filter(|&&(_, y)| s.iter().any(|t| t.0 == y))
            .map(|&(_, y)| vec![y])
            .collect()
    }),
    ("q(X,Z) :- r(X,Y), s(Y,Z), X = 1.", "q", |r, s| {
        r.iter()
            .filter(|t| t.0 == 1)
            .flat_map(|&(x, y)| {
                s.iter()
                    .filter(move |t| t.0 == y)
                    .map(move |&(_, z)| vec![x, z])
            })
            .collect()
    }),
    ("q(Z,X) :- r(X,Y), s(Y,Z).", "q", |r, s| {
        r.iter()
            .flat_map(|&(x, y)| {
                s.iter()
                    .filter(move |t| t.0 == y)
                    .map(move |&(_, z)| vec![z, x])
            })
            .collect()
    }),
    ("q(X,Z) :- r(Y,X), r(Y,Z).", "q", |r, _| {
        r.iter()
            .flat_map(|&(y, x)| {
                r.iter()
                    .filter(move |t| t.0 == y)
                    .map(move |&(_, z)| vec![x, z])
            })
            .collect()
    }),
    ("q(X,Y) :- r(X,Y). q(X,Y) :- s(X,Y).", "q", |r, s| {
        r.iter().chain(s.iter()).map(|&(a, b)| vec![a, b]).collect()
    }),
];

#[test]
fn compiled_programs_match_enumeration() {
    let mut seed = 0x2545_F491_4F6C_DD1Du64;
    for (src, head, oracle) in PROGRAMS {
        let mut nonempty = 0;
        for _ in 0..100 {
            let (r, s) = (random(&mut seed, 15), random(&mut seed, 15));
            let (ex, report) = run(src, &r, &s);
            assert!(
                report.refused.is_empty(),
                "{src}: refused {:?}",
                report.refused
            );
            let want = oracle(&r, &s);
            nonempty += usize::from(!want.is_empty());
            assert_eq!(rows(&ex, head), want, "{src}\nr={r:?}\ns={s:?}");
        }
        assert!(nonempty > 15, "{src}: only {nonempty} non-empty cases");
    }
}

#[test]
fn two_heads_share_one_cube() {
    let src = "a(X,Z) :- r(X,Y), s(Y,Z). b(Y) :- s(Y,Z), r(X,Y).";
    let mut seed = 7u64;
    let (r, s) = (random(&mut seed, 15), random(&mut seed, 15));
    let (ex, report) = run(src, &r, &s);
    assert!(report.refused.is_empty(), "{:?}", report.refused);
    assert_eq!(rows(&ex, "a"), PROGRAMS[0].2(&r, &s));
    assert_eq!(rows(&ex, "b"), PROGRAMS[1].2(&r, &s));
    // Each join is one support pass inside `materialize_rows`: two programs.
    assert_eq!(report.stats.programs, 4);
}

/// Naive fixpoint: apply every rule to the current relations until nothing
/// changes. Independent of the executor's semi-naive rounds.
fn naive_fixpoint(r: &Rel, s: &Rel, step: Step) -> BTreeSet<Vec<u32>> {
    let mut tc = BTreeSet::new();
    loop {
        let next = step(&tc, r, s);
        if next == tc {
            return tc;
        }
        tc = next;
    }
}

type Step = fn(&BTreeSet<Vec<u32>>, &Rel, &Rel) -> BTreeSet<Vec<u32>>;

const RECURSIVE: [(&str, Step); 4] = [
    // Linear transitive closure.
    ("tc(X,Y) :- r(X,Y). tc(X,Z) :- tc(X,Y), r(Y,Z).", |tc, r, _| {
        let mut n: BTreeSet<Vec<u32>> = r.iter().map(|&(a, b)| vec![a, b]).collect();
        for t in tc {
            n.insert(t.clone());
            for &(y, z) in r {
                if t[1] == y {
                    n.insert(vec![t[0], z]);
                }
            }
        }
        n
    }),
    // Non-linear (doubling) transitive closure: two SCC atoms per body.
    ("tc(X,Y) :- r(X,Y). tc(X,Z) :- tc(X,Y), tc(Y,Z).", |tc, r, _| {
        let mut n: BTreeSet<Vec<u32>> = r.iter().map(|&(a, b)| vec![a, b]).collect();
        for a in tc {
            n.insert(a.clone());
            for b in tc {
                if a[1] == b[0] {
                    n.insert(vec![a[0], b[1]]);
                }
            }
        }
        n
    }),
    // Reachability over the union of two edge relations.
    ("tc(X,Y) :- r(X,Y). tc(X,Y) :- s(X,Y). tc(X,Z) :- tc(X,Y), r(Y,Z). tc(X,Z) :- tc(X,Y), s(Y,Z).", |tc, r, s| {
        let edges: Vec<(u32, u32)> = r.iter().chain(s.iter()).copied().collect();
        let mut n: BTreeSet<Vec<u32>> = edges.iter().map(|&(a, b)| vec![a, b]).collect();
        for t in tc {
            n.insert(t.clone());
            for &(y, z) in &edges {
                if t[1] == y {
                    n.insert(vec![t[0], z]);
                }
            }
        }
        n
    }),
    // Reachable from node 1 only, through a bound start.
    ("tc(X,Y) :- r(X,Y), X = 1. tc(X,Z) :- tc(X,Y), r(Y,Z).", |tc, r, _| {
        let mut n: BTreeSet<Vec<u32>> =
            r.iter().filter(|t| t.0 == 1).map(|&(a, b)| vec![a, b]).collect();
        for t in tc {
            n.insert(t.clone());
            for &(y, z) in r {
                if t[1] == y {
                    n.insert(vec![t[0], z]);
                }
            }
        }
        n
    }),
];

#[test]
fn recursive_programs_match_a_naive_fixpoint() {
    let mut seed = 0x9E37_79B9_7F4A_7C15u64;
    for (src, step) in RECURSIVE {
        let (mut nonempty, mut deep) = (0, 0);
        for _ in 0..100 {
            let (r, s) = (random(&mut seed, 15), random(&mut seed, 15));
            let (ex, report) = run(src, &r, &s);
            assert!(
                report.refused.is_empty(),
                "{src}: refused {:?}",
                report.refused
            );
            let want = naive_fixpoint(&r, &s, step);
            nonempty += usize::from(!want.is_empty());
            deep += usize::from(report.rounds >= 3);
            assert_eq!(rows(&ex, "tc"), want, "{src}\nr={r:?}\ns={s:?}");
        }
        assert!(nonempty > 15, "{src}: only {nonempty} non-empty cases");
        assert!(deep > 10, "{src}: only {deep} cases needed 3+ rounds");
    }
}

#[test]
fn a_derived_relation_feeds_a_later_rule() {
    let r: Rel = vec![(1, 2), (2, 3), (3, 3)];
    let s: Rel = vec![(2, 1), (3, 3)];
    // `p` reads `q`, which the plan derives first.
    let (ex, report) = run(
        "q(X,Z) :- r(X,Y), s(Y,Z). p(X,Z) :- q(X,Y), r(Y,Z).",
        &r,
        &s,
    );
    assert!(report.refused.is_empty(), "{report:?}");
    let q = PROGRAMS[0].2(&r, &s);
    let mut p = BTreeSet::new();
    for t in &q {
        for &(y, z) in &r {
            if t[1] == y {
                p.insert(vec![t[0], z]);
            }
        }
    }
    assert_eq!(rows(&ex, "q"), q);
    assert_eq!(rows(&ex, "p"), p);
}

#[test]
fn an_unsupported_rule_refuses_its_whole_recursive_scc() {
    let r: Rel = vec![(1, 2), (2, 3)];
    // The second recursive rule compares two columns: not admitted.
    let (ex, report) = run(
        "tc(X,Y) :- r(X,Y). tc(X,Z) :- tc(X,Y), r(Y,Z). tc(X,X) :- tc(X,X).",
        &r,
        &vec![],
    );
    assert!(report.refused.iter().any(|(h, _)| h == "tc"), "{report:?}");
    assert_eq!(
        ex.relation("tc"),
        None,
        "a refused head is never handed back"
    );
}

#[test]
fn a_head_with_one_refused_rule_is_not_handed_back_partially() {
    // The first rule is admitted on its own; the second (`s(X, X)`, a filter
    // comparing two columns) is refused. Handing back only the first rule's
    // rows would be a wrong, smaller `q`, so the whole SCC is refused.
    let r: Rel = vec![(1, 2), (3, 4)];
    let s: Rel = vec![(5, 5), (6, 7)];
    let (ex, report) = run("q(X,Y) :- r(X,Y). q(X,X) :- s(X,X).", &r, &s);
    assert!(report.refused.iter().any(|(h, _)| h == "q"), "{report:?}");
    assert_eq!(ex.relation("q"), None);
}

/// Brute force over every assignment of `vars` variables to ids 0..10.
fn brute(
    vars: usize,
    r: &Rel,
    s: &Rel,
    keep: &dyn Fn(&[u32], &Rel, &Rel) -> Option<Vec<u32>>,
) -> BTreeSet<Vec<u32>> {
    let mut out = BTreeSet::new();
    let mut b = vec![0u32; vars];
    loop {
        if let Some(row) = keep(&b, r, s) {
            out.insert(row);
        }
        let mut i = 0;
        loop {
            if i == vars {
                return out;
            }
            b[i] += 1;
            if b[i] < 10 {
                break;
            }
            b[i] = 0;
            i += 1;
        }
    }
}

fn has(rel: &Rel, a: u32, b: u32) -> bool {
    rel.contains(&(a, b))
}

type Keep = fn(&[u32], &Rel, &Rel) -> Option<Vec<u32>>;

const MULTIWAY: [(&str, &str, usize, Keep); 4] = [
    (
        "tri(X,Y,Z) :- r(X,Y), r(Y,Z), r(X,Z).",
        "tri",
        3,
        |b, r, _| {
            (has(r, b[0], b[1]) && has(r, b[1], b[2]) && has(r, b[0], b[2]))
                .then(|| vec![b[0], b[1], b[2]])
        },
    ),
    (
        "c4(X,Y,Z,W) :- r(X,Y), r(Y,Z), r(Z,W), r(W,X).",
        "c4",
        4,
        |b, r, _| {
            (has(r, b[0], b[1]) && has(r, b[1], b[2]) && has(r, b[2], b[3]) && has(r, b[3], b[0]))
                .then(|| vec![b[0], b[1], b[2], b[3]])
        },
    ),
    (
        "k4(A,B,C,D) :- r(A,B), r(A,C), r(A,D), r(B,C), r(B,D), r(C,D).",
        "k4",
        4,
        |b, r, _| {
            let e = |i: usize, j: usize| has(r, b[i], b[j]);
            (e(0, 1) && e(0, 2) && e(0, 3) && e(1, 2) && e(1, 3) && e(2, 3)).then(|| b.to_vec())
        },
    ),
    // Two relations, projected: endpoints of an r-s-r path.
    ("q(X,W) :- r(X,Y), s(Y,Z), r(Z,W).", "q", 4, |b, r, s| {
        (has(r, b[0], b[1]) && has(s, b[1], b[2]) && has(r, b[2], b[3])).then(|| vec![b[0], b[3]])
    }),
];

#[test]
fn multi_way_joins_match_brute_force() {
    let mut seed = 0xD1B5_4A32_D192_ED03u64;
    for (src, head, vars, keep) in MULTIWAY {
        let plan = Compiler::new().compile(src).expect("compile");
        let multiway = plan
            .rules_by_scc
            .iter()
            .flatten()
            .any(|r| r.head == head && matches!(r.body, xlog_ir::RirNode::MultiWayJoin { .. }));
        assert!(multiway, "{src}: XLOG did not compile it to a MultiWayJoin");
        let mut nonempty = 0;
        for _ in 0..60 {
            // Dense relations, so cycles and cliques occur.
            let (r, s) = (random(&mut seed, 40), random(&mut seed, 40));
            let (ex, report) = run(src, &r, &s);
            assert!(
                report.refused.is_empty(),
                "{src}: refused {:?}",
                report.refused
            );
            let want = brute(vars, &r, &s, &keep);
            nonempty += usize::from(!want.is_empty());
            assert_eq!(rows(&ex, head), want, "{src}\nr={r:?}\ns={s:?}");
        }
        assert!(nonempty > 10, "{src}: only {nonempty} non-empty cases");
    }
}
