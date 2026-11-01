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

#[test]
fn recursion_and_derived_inputs_are_refused_not_answered() {
    let r: Rel = vec![(1, 2), (2, 3)];
    let (ex, report) = run(
        "tc(X,Y) :- r(X,Y). tc(X,Z) :- tc(X,Y), r(Y,Z).",
        &r,
        &vec![],
    );
    assert!(
        report
            .refused
            .iter()
            .any(|(h, why)| h == "tc" && why.contains("recursive")),
        "{report:?}"
    );
    assert_eq!(
        ex.relation("tc"),
        None,
        "a refused head is never handed back"
    );

    let s: Rel = vec![(2, 1), (3, 3)];
    let (ex, report) = run("q(X,Z) :- r(X,Y), s(Y,Z). p(X) :- q(X,X).", &r, &s);
    assert!(report.ran.contains(&"q".to_string()));
    assert!(report.refused.iter().any(|(h, _)| h == "p"), "{report:?}");
    assert_eq!(rows(&ex, "q"), PROGRAMS[0].2(&r, &s));
    assert_eq!(ex.relation("p"), None);
}
