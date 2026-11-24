//! EXO-394: stratified negation over one-column sets. `not p(X)` lowers to a
//! semi-join over a `Diff`; a one-column read of a binary relation lowers to
//! `Project(Scan r, [c])`, a cube `SetExpr::Column`. The expected rows are
//! written out by hand from the facts.

use std::collections::BTreeSet;

use xlog_cube::{CubeExecutor, Report};
use xlog_logic::Compiler;

fn run(
    src: &str,
    sets: &[(&str, &[u32])],
    pairs: &[(&str, &[(u32, u32)])],
) -> (CubeExecutor, Report) {
    let mut c = Compiler::new();
    let plan = c.compile(src).expect("compile");
    let mut ex = CubeExecutor::new();
    for (name, id) in c.rel_ids() {
        ex.register_relation(*id, name);
    }
    for (n, ids) in sets {
        ex.put_set(n, ids);
    }
    for (n, rows) in pairs {
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

fn route_of<'a>(r: &'a Report, head: &str) -> Option<&'a str> {
    r.routed
        .iter()
        .find(|(h, _)| h == head)
        .map(|(_, p)| p.as_str())
}

fn refused(r: &Report, head: &str) -> bool {
    r.refused.iter().any(|(h, _)| h == head)
}

const EDGE: &[(u32, u32)] = &[(1, 2), (2, 3), (4, 3)];
const NODES: &[u32] = &[1, 2, 3, 4, 5];

#[test]
fn no_in_is_node_minus_the_targets() {
    let src = r#"
        pred node(u32).
        pred edge(u32, u32).
        pred has_in(u32).
        pred no_in(u32).
        has_in(X) :- edge(_, X).
        no_in(X) :- node(X), not has_in(X).
    "#;
    let (ex, r) = run(src, &[("node", NODES)], &[("edge", EDGE)]);
    assert_eq!(rows(&ex, "has_in"), ones(&[2, 3]));
    assert_eq!(rows(&ex, "no_in"), ones(&[1, 4, 5]));
    assert_eq!(route_of(&r, "has_in"), Some("set"));
    assert_eq!(route_of(&r, "no_in"), Some("set"));
    assert!(r.refused.is_empty(), "{:?}", r.refused);
}

/// Negative twin: negating the sources instead of the targets.
#[test]
fn no_out_is_node_minus_the_sources() {
    let src = r#"
        pred node(u32).
        pred edge(u32, u32).
        pred has_out(u32).
        pred no_out(u32).
        has_out(X) :- edge(X, _).
        no_out(X) :- node(X), not has_out(X).
    "#;
    let (ex, _) = run(src, &[("node", NODES)], &[("edge", EDGE)]);
    assert_eq!(rows(&ex, "no_out"), ones(&[3, 5]));
}

#[test]
fn isolated_negates_both_columns() {
    let src = r#"
        pred node(u32).
        pred edge(u32, u32).
        pred isolated(u32).
        isolated(X) :- node(X), not edge(X, _), not edge(_, X).
    "#;
    let (ex, r) = run(src, &[("node", NODES)], &[("edge", EDGE)]);
    assert_eq!(rows(&ex, "isolated"), ones(&[5]));
    assert_eq!(route_of(&r, "isolated"), Some("set"));
}

#[test]
fn leaf_part_negates_a_derived_head() {
    let src = r#"
        pred contains(u32, u32).
        pred part(u32).
        pred has_children(u32).
        pred leaf_part(u32).
        part(X) :- contains(X, Y).
        part(Y) :- contains(X, Y).
        has_children(X) :- contains(X, Y).
        leaf_part(X) :- part(X), not has_children(X).
    "#;
    let contains: &[(u32, u32)] = &[(10, 11), (10, 12), (12, 13)];
    let (ex, r) = run(src, &[], &[("contains", contains)]);
    assert_eq!(rows(&ex, "part"), ones(&[10, 11, 12, 13]));
    assert_eq!(rows(&ex, "leaf_part"), ones(&[11, 13]));
    assert_eq!(route_of(&r, "leaf_part"), Some("set"));
}

/// An empty negated relation removes nothing; an empty positive one yields
/// nothing.
#[test]
fn empty_relations() {
    let src = r#"
        pred node(u32).
        pred edge(u32, u32).
        pred no_in(u32).
        no_in(X) :- node(X), not edge(_, X).
    "#;
    let (ex, _) = run(src, &[("node", NODES)], &[("edge", &[])]);
    assert_eq!(rows(&ex, "no_in"), ones(NODES));
    let (ex, _) = run(src, &[("node", &[])], &[("edge", EDGE)]);
    assert!(rows(&ex, "no_in").is_empty());
}

/// A binary difference is not a one-column term: it is not routed and keeps
/// its typed refusal.
#[test]
fn a_binary_difference_is_not_routed() {
    let src = r#"
        pred pair(u32, u32).
        pred blocked(u32, u32).
        pred keep(u32, u32).
        keep(X, Y) :- pair(X, Y), not blocked(X, Y).
    "#;
    let (_, r) = run(
        src,
        &[],
        &[("pair", &[(1, 2), (3, 4)]), ("blocked", &[(1, 2)])],
    );
    assert_eq!(route_of(&r, "keep"), None);
    assert!(refused(&r, "keep"));
}
