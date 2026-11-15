//! Two ways the executor used to report a wrong answer as `ran`, found by
//! the lance-graph coverage probe (`crates/xlog-coverage-probe`) against an
//! independent naive oracle.

use std::collections::BTreeSet;

use xlog_cube::CubeExecutor;
use xlog_logic::Compiler;

fn run(src: &str, base: &[(&str, &[(u32, u32)])]) -> (CubeExecutor, xlog_cube::Report) {
    let mut c = Compiler::new();
    let plan = c.compile(src).expect("compile");
    let mut ex = CubeExecutor::new();
    for (name, id) in c.rel_ids() {
        ex.register_relation(*id, name);
    }
    for (name, rows) in base {
        ex.put_relation(name, rows);
    }
    let report = ex.execute_plan(&plan).expect("plan");
    (ex, report)
}

fn rows(ex: &CubeExecutor, name: &str) -> BTreeSet<Vec<u32>> {
    ex.relation(name).unwrap_or_default().into_iter().collect()
}

/// `keep` is refused (its body has a negation). The query reading it must be
/// refused too, naming `keep`, never run over an empty stand-in for it.
#[test]
fn a_rule_reading_a_refused_relation_is_refused() {
    let src = "pair(1, 10). pair(1, 20). pair(2, 10). blocked(1, 10). \
               keep(X, Y) :- pair(X, Y), not blocked(X, Y). ?- keep(X, Y).";
    let pair: &[(u32, u32)] = &[(1, 10), (1, 20), (2, 10)];
    let blocked: &[(u32, u32)] = &[(1, 10)];
    let (ex, report) = run(src, &[("pair", pair), ("blocked", blocked)]);
    assert!(
        report.refused.iter().any(|(h, _)| h == "keep"),
        "{report:?}"
    );
    assert!(
        !report.ran.iter().any(|h| h == "__xlog_query_0"),
        "{report:?}"
    );
    let why = report
        .refused
        .iter()
        .find(|(h, _)| h == "__xlog_query_0")
        .map(|(_, w)| w.as_str());
    assert_eq!(why, Some("reads refused relation `keep`"), "{report:?}");
    assert!(ex.relation("__xlog_query_0").is_none());
}

/// Control: a query over a relation that ran is still run.
#[test]
fn a_rule_reading_a_relation_that_ran_still_runs() {
    let src = "e(1, 2). e(2, 3). p(X,Z) :- e(X,Y), e(Y,Z). ?- p(X, Y).";
    let e: &[(u32, u32)] = &[(1, 2), (2, 3)];
    let (ex, report) = run(src, &[("e", e)]);
    assert!(report.refused.is_empty(), "{report:?}");
    assert_eq!(rows(&ex, "__xlog_query_0"), BTreeSet::from([vec![1, 3]]));
}

/// Rows a host puts for a derived predicate are part of its answer, and in a
/// recursive SCC they seed the first round (XLOG's own
/// `test_same_head_union_batching.rs` expects `reach = {(1,2), (1,3)}`).
#[test]
fn rows_put_for_a_recursive_derived_predicate_seed_it() {
    let src = "edge(2, 3). reach(1, 2). reach(X, Z) :- reach(X, Y), edge(Y, Z). ?- reach(X, Y).";
    let edge: &[(u32, u32)] = &[(2, 3)];
    let reach: &[(u32, u32)] = &[(1, 2)];
    let (ex, report) = run(src, &[("edge", edge), ("reach", reach)]);
    assert!(report.refused.is_empty(), "{report:?}");
    let want = BTreeSet::from([vec![1, 2], vec![1, 3]]);
    assert_eq!(rows(&ex, "reach"), want);
    assert_eq!(rows(&ex, "__xlog_query_0"), want);
}

/// The same for a non-recursive SCC: the rule's rows and the put rows.
#[test]
fn rows_put_for_a_plain_derived_predicate_are_kept() {
    let src = "r(1, 2). s(2, 3). q(1, 2). q(X,Z) :- r(X,Y), s(Y,Z).";
    let r: &[(u32, u32)] = &[(1, 2)];
    let s: &[(u32, u32)] = &[(2, 3)];
    let q: &[(u32, u32)] = &[(1, 2)];
    let (ex, report) = run(src, &[("r", r), ("s", s), ("q", q)]);
    assert!(report.refused.is_empty(), "{report:?}");
    assert_eq!(rows(&ex, "q"), BTreeSet::from([vec![1, 2], vec![1, 3]]));
}
