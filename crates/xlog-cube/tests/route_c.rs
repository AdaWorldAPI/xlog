//! Route C (EXO-396): a non-recursive derived binary head is a recipe. A
//! reader whose body names it gets the head's rules substituted in place of
//! the atom, and the head is not re-added to the cube as a relation. When
//! the substitution is refused (variable limit, expansion budget, a shape the
//! cube's walk does not take) the head is re-added and the rule reruns; the
//! report names it, with the reason. Every expected answer is written out by
//! hand from the facts.

use std::collections::BTreeSet;

use xlog_cube::{CubeExecutor, Report};
use xlog_logic::Compiler;

fn run(src: &str, pairs: &[(&str, &[(u32, u32)])]) -> (CubeExecutor, Report) {
    let mut c = Compiler::new();
    let plan = c.compile(src).expect("compile");
    let mut ex = CubeExecutor::new();
    for (name, id) in c.rel_ids() {
        ex.register_relation(*id, name);
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

fn set<const N: usize>(v: &[[u32; N]]) -> BTreeSet<Vec<u32>> {
    v.iter().map(|r| r.to_vec()).collect()
}

fn reingested(r: &Report, head: &str) -> bool {
    r.reingested.iter().any(|(h, _)| h == head)
}

fn kept(r: &Report, head: &str) -> bool {
    r.recipes.iter().any(|h| h == head)
}

fn refused(r: &Report, head: &str) -> bool {
    r.refused.iter().any(|(h, _)| h == head)
}

/// `q` is two hops, `r` one more: only `r` is wanted, and `q` is read by `r`
/// as its recipe, never as a cube relation. `q`'s rows are still returned
/// (the executor downloads every head).
#[test]
fn a_reader_composes_a_derived_head_without_reingesting_it() {
    let src = r#"
        pred e(u32, u32).
        pred q(u32, u32).
        pred r(u32, u32).
        q(X, Z) :- e(X, Y), e(Y, Z).
        r(X, W) :- q(X, Z), e(Z, W).
    "#;
    // e: 1->2, 2->3, 3->4, 4->5.
    let e: &[(u32, u32)] = &[(1, 2), (2, 3), (3, 4), (4, 5)];
    let (ex, rep) = run(src, &[("e", e)]);
    // q = two hops: (1,3) (2,4) (3,5). r = three hops: (1,4) (2,5).
    assert_eq!(rows(&ex, "q"), set(&[[1, 3], [2, 4], [3, 5]]));
    assert_eq!(rows(&ex, "r"), set(&[[1, 4], [2, 5]]));
    assert!(!reingested(&rep, "q"), "reingested: {:?}", rep.reingested);
    assert!(kept(&rep, "q"), "recipes: {:?}", rep.recipes);
    assert!(!refused(&rep, "r"), "{:?}", rep.refused);
}

/// Two rules for `q`: the reader's atom is their disjunction, ORed on the
/// planes. Neither disjunct may be dropped.
#[test]
fn every_rule_of_a_derived_head_reaches_the_reader() {
    let src = r#"
        pred e(u32, u32).
        pred f(u32, u32).
        pred q(u32, u32).
        pred r(u32, u32).
        q(X, Z) :- e(X, Z).
        q(X, Z) :- e(X, Y), e(Y, Z).
        r(X, W) :- q(X, Z), f(Z, W).
    "#;
    let e: &[(u32, u32)] = &[(1, 2), (2, 3)];
    let f: &[(u32, u32)] = &[(2, 7), (3, 8)];
    let (ex, rep) = run(src, &[("e", e), ("f", f)]);
    // q = e ∪ e∘e = (1,2) (2,3) (1,3).
    // r: (1,2)->f(2,7) = (1,7); (2,3)->f(3,8) = (2,8); (1,3)->f(3,8) = (1,8).
    assert_eq!(rows(&ex, "r"), set(&[[1, 7], [2, 8], [1, 8]]));
    // Each disjunct alone would miss one of these.
    assert!(rows(&ex, "r").contains(&vec![1, 7]));
    assert!(rows(&ex, "r").contains(&vec![1, 8]));
    assert!(!reingested(&rep, "q"), "reingested: {:?}", rep.reingested);
    assert!(kept(&rep, "q"), "recipes: {:?}", rep.recipes);
}

/// A distinctness pair inside the derived head is renamed into the reader.
#[test]
fn a_derived_heads_distinctness_survives_substitution() {
    let src = r#"
        pred e(u32, u32).
        pred q(u32, u32).
        pred r(u32, u32).
        q(X, Z) :- e(X, Y), e(Y, Z), X != Z.
        r(X, W) :- q(X, Z), e(Z, W).
    "#;
    // e: 1->2, 2->1, 1->3.
    let e: &[(u32, u32)] = &[(1, 2), (2, 1), (1, 3)];
    let (ex, rep) = run(src, &[("e", e)]);
    // e∘e: 1->2->1 (1,1); 2->1->2 (2,2); 2->1->3 (2,3). X != Z keeps (2,3).
    assert_eq!(rows(&ex, "q"), set(&[[2, 3]]));
    // r: q(2,3) then e(3,_) is empty. So r is empty; without the pair,
    // q(1,1)->e(1,2),(1,3) and q(2,2)->e(2,1) would give (1,2) (1,3) (2,1).
    assert!(rows(&ex, "r").is_empty(), "{:?}", rows(&ex, "r"));
    assert!(!reingested(&rep, "q"), "reingested: {:?}", rep.reingested);
    assert!(kept(&rep, "q"), "recipes: {:?}", rep.recipes);
}

/// A substitution past the eight-variable limit is refused before any body
/// is built; the head is re-added and the rule still answers correctly.
#[test]
fn past_the_variable_limit_the_head_is_reingested_with_its_reason() {
    let src = r#"
        pred e(u32, u32).
        pred q(u32, u32).
        pred r(u32, u32).
        q(A, F) :- e(A, B), e(B, C), e(C, D), e(D, E2), e(E2, F).
        r(X, W) :- q(X, Y), q(Y, Z), e(Z, W).
    "#;
    // A path 1->2->...->12.
    let e: Vec<(u32, u32)> = (1..12).map(|i| (i, i + 1)).collect();
    let (ex, rep) = run(src, &[("e", &e)]);
    // q = five hops: (i, i+5) for i in 1..=7. r = 11 hops: (1, 12).
    assert_eq!(rows(&ex, "r"), set(&[[1, 12]]));
    let why = rep
        .reingested
        .iter()
        .find(|(h, _)| h == "q")
        .map(|(_, w)| w.as_str());
    assert!(
        why.is_some_and(|w| w.contains("variable limit")),
        "reingested: {:?}",
        rep.reingested
    );
    assert!(!kept(&rep, "q"));
}

/// A recursive SCC reading a derived head gets it as a relation: recursion
/// is not composed in this slice.
#[test]
fn a_recursive_reader_reads_the_derived_head_as_a_relation() {
    let src = r#"
        pred e(u32, u32).
        pred q(u32, u32).
        pred t(u32, u32).
        q(X, Z) :- e(X, Y), e(Y, Z).
        t(X, Y) :- q(X, Y).
        t(X, Z) :- t(X, Y), q(Y, Z).
    "#;
    let e: &[(u32, u32)] = &[(1, 2), (2, 3), (3, 4), (4, 5)];
    let (ex, rep) = run(src, &[("e", e)]);
    // q = (1,3) (2,4) (3,5); t = q⁺ = (1,3) (2,4) (3,5) (1,5).
    assert_eq!(rows(&ex, "t"), set(&[[1, 3], [2, 4], [3, 5], [1, 5]]));
    let why = rep
        .reingested
        .iter()
        .find(|(h, _)| h == "q")
        .map(|(_, w)| w.as_str());
    assert_eq!(why, Some("read by a recursive SCC"), "{:?}", rep.reingested);
}

/// What the demand fixture builds, measured from the cube's own counters.
/// `r` reads `q` through its recipe, so no derived relation is added to the
/// cube. `q`'s own three rows are built, by the join path, because every head
/// is downloaded; `r` adds none: the program without `r` builds the same.
#[test]
fn the_composed_reader_adds_no_relation_and_builds_no_rows() {
    let src = r#"
        pred e(u32, u32).
        pred q(u32, u32).
        pred r(u32, u32).
        q(X, Z) :- e(X, Y), e(Y, Z).
        r(X, W) :- q(X, Z), e(Z, W).
    "#;
    let e: &[(u32, u32)] = &[(1, 2), (2, 3), (3, 4), (4, 5)];
    let (_, rep) = run(src, &[("e", e)]);
    assert_eq!(rep.stats.relations_added, 0, "{:?}", rep.stats);
    assert_eq!(rep.stats.rows_materialized, 3, "q's rows: {:?}", rep.stats);
    assert_eq!(rep.recipes, vec!["q".to_string(), "r".to_string()]);
    let without_r = r#"
        pred e(u32, u32).
        pred q(u32, u32).
        q(X, Z) :- e(X, Y), e(Y, Z).
    "#;
    let (_, base) = run(without_r, &[("e", e)]);
    assert_eq!(base.stats.rows_materialized, rep.stats.rows_materialized);
}

fn run_demanded(
    src: &str,
    pairs: &[(&str, &[(u32, u32)])],
    demand: &[&str],
) -> (CubeExecutor, Report) {
    let mut c = Compiler::new();
    let plan = c.compile(src).expect("compile");
    let mut ex = CubeExecutor::new();
    for (name, id) in c.rel_ids() {
        ex.register_relation(*id, name);
    }
    for (n, rows) in pairs {
        ex.put_relation(n, rows);
    }
    ex.demand(demand);
    let report = ex.execute_plan(&plan).expect("plan");
    (ex, report)
}

const TWO_HOP: &str = r#"
    pred e(u32, u32).
    pred q(u32, u32).
    pred r(u32, u32).
    q(X, Z) :- e(X, Y), e(Y, Z).
    r(X, W) :- q(X, Z), e(Z, W).
"#;

/// FAILS IF: demanding only `r` changes `r`'s answer, or still builds `q`'s
/// rows. `q` is read by `r` as its recipe, so nothing needs its rows.
#[test]
fn an_undemanded_recipe_head_is_never_built() {
    let e: &[(u32, u32)] = &[(1, 2), (2, 3), (3, 4), (4, 5)];
    let (all_ex, all) = run(TWO_HOP, &[("e", e)]);
    let (ex, rep) = run_demanded(TWO_HOP, &[("e", e)], &["r"]);
    assert_eq!(rows(&ex, "r"), set(&[[1, 4], [2, 5]]));
    assert_eq!(rows(&ex, "r"), rows(&all_ex, "r"));
    assert_eq!(ex.relation("q"), None, "q is not decoded");
    assert_eq!(rep.undecoded, vec!["q".to_string()]);
    assert!(all.undecoded.is_empty(), "no demand decodes everything");
    // The three rows `q`'s join built are gone; nothing replaced them.
    assert_eq!(all.stats.rows_materialized, 3);
    assert_eq!(rep.stats.rows_materialized, 0, "{:?}", rep.stats);
    assert_eq!(rep.ran, all.ran);
    assert!(rep.refused.is_empty() && rep.reingested.is_empty());
}

/// FAILS IF: a demanded head is skipped. Demanding both keeps both decoded.
#[test]
fn a_demanded_recipe_head_is_built() {
    let e: &[(u32, u32)] = &[(1, 2), (2, 3), (3, 4), (4, 5)];
    let (ex, rep) = run_demanded(TWO_HOP, &[("e", e)], &["q", "r"]);
    assert_eq!(rows(&ex, "q"), set(&[[1, 3], [2, 4], [3, 5]]));
    assert!(rep.undecoded.is_empty());
    assert_eq!(rep.stats.rows_materialized, 3);
}

/// FAILS IF: an undemanded head that a recursive reader must add to the cube
/// is added without its rows. It is decoded then, from its recipes.
#[test]
fn an_undemanded_head_read_by_recursion_is_decoded_when_added() {
    let src = r#"
        pred e(u32, u32).
        pred q(u32, u32).
        pred t(u32, u32).
        q(X, Z) :- e(X, Y), e(Y, Z).
        t(X, Y) :- q(X, Y).
        t(X, Z) :- t(X, Y), q(Y, Z).
    "#;
    let e: &[(u32, u32)] = &[(1, 2), (2, 3), (3, 4), (4, 5)];
    let (ex, rep) = run_demanded(src, &[("e", e)], &["t"]);
    assert_eq!(rows(&ex, "t"), set(&[[1, 3], [2, 4], [3, 5], [1, 5]]));
    assert!(reingested(&rep, "q"), "{:?}", rep.reingested);
    assert!(
        rep.undecoded.is_empty(),
        "q was decoded to be added: {:?}",
        rep.undecoded
    );
}

fn run_queries(src: &str, pairs: &[(&str, &[(u32, u32)])]) -> (CubeExecutor, Report) {
    let mut c = Compiler::new();
    let plan = c.compile(src).expect("compile");
    assert!(!plan.generated_query_rules.is_empty() || !src.contains("?-"));
    let mut ex = CubeExecutor::new();
    for (name, id) in c.rel_ids() {
        ex.register_relation(*id, name);
    }
    for (n, rows) in pairs {
        ex.put_relation(n, rows);
    }
    ex.demand_queries();
    let report = ex.execute_plan(&plan).expect("plan");
    (ex, report)
}

/// FAILS IF: the demand is not the plan's query heads. With one `?-` over `r`,
/// only `__xlog_query_0` is demanded, so neither `q` nor `r` is built, and the
/// query's answer is unchanged.
#[test]
fn the_query_rules_are_the_demand() {
    let src = format!("{TWO_HOP}\n?- r(X, W).\n");
    let e: &[(u32, u32)] = &[(1, 2), (2, 3), (3, 4), (4, 5)];
    let (all_ex, _) = run(&src, &[("e", e)]);
    let (ex, rep) = run_queries(&src, &[("e", e)]);
    let want = rows(&all_ex, "__xlog_query_0");
    assert_eq!(want, set(&[[1, 4], [2, 5]]), "the query's answer");
    assert_eq!(rows(&ex, "__xlog_query_0"), want);
    assert_eq!(rep.undecoded, vec!["q".to_string(), "r".to_string()]);
    assert_eq!(ex.relation("r"), None);
}

/// FAILS IF: a plan without queries loses heads under `demand_queries`: it
/// asks nothing, so every head is decoded.
#[test]
fn a_plan_without_queries_decodes_every_head() {
    let e: &[(u32, u32)] = &[(1, 2), (2, 3), (3, 4), (4, 5)];
    let (ex, rep) = run_queries(TWO_HOP, &[("e", e)]);
    assert!(rep.undecoded.is_empty());
    assert_eq!(rows(&ex, "q"), set(&[[1, 3], [2, 4], [3, 5]]));
}

const MAGIC_REACH: &str = r#"
    #pragma magic_sets = on
    pred edge(u32, u32).
    pred reach(u32, u32).
    pred two(u32, u32).
    reach(X, Y) :- edge(X, Y).
    reach(X, Z) :- reach(X, Y), edge(Y, Z).
    two(X, Z) :- edge(X, Y), edge(Y, Z).
    ?- reach(1, Y).
"#;

fn run_magic(
    src: &str,
    report_src: &str,
    pairs: &[(&str, &[(u32, u32)])],
) -> Result<(CubeExecutor, Report), xlog_cube::PlanRefusal> {
    let mut c = Compiler::new();
    let plan = c.compile(src).expect("compile");
    let program = xlog_logic::parse_program(report_src).expect("parse");
    let report = xlog_logic::rewrite_magic_sets(&program)
        .expect("rewrite")
        .report;
    let mut ex = CubeExecutor::new();
    for (name, id) in c.rel_ids() {
        ex.register_relation(*id, name);
    }
    for (n, rows) in pairs {
        ex.put_relation(n, rows);
    }
    ex.demand_magic(&report);
    let rep = ex.execute_plan(&plan)?;
    Ok((ex, rep))
}

/// FAILS IF: the magic demand drops the query or the adorned predicate, or
/// keeps the unread recipe head `two`. `reach/bf` demands `reach`.
///
/// Pinned as it stands: the cube cannot run the rewritten recursion yet. Its
/// rules read the one-column seed `__xlog_magic_reach_bf`, and a recursive
/// SCC reads its inputs only as binary relations, so `reach` and the query
/// are refused, with or without a demand. The demand is still validated and
/// still keeps `two` undecoded. When the cube runs magic-guarded recursion
/// this test fails, and the query's answer `{(1,2), (1,3)}` replaces the
/// refusal assertions.
#[test]
fn the_adorned_predicates_join_the_demand() {
    let e: &[(u32, u32)] = &[(1, 2), (2, 3), (10, 11), (11, 12)];
    let (_, all) = run(MAGIC_REACH, &[("edge", e)]);
    let (ex, rep) =
        run_magic(MAGIC_REACH, MAGIC_REACH, &[("edge", e)]).expect("the report matches the plan");
    let not_binary = "`__xlog_magic_reach_bf` is not a binary relation in the cube";
    assert!(
        rep.refused
            .iter()
            .any(|(h, why)| h == "reach" && why == not_binary),
        "{:?}",
        rep.refused
    );
    assert!(refused(&rep, "__xlog_query_0"));
    assert_eq!(rep.refused, all.refused, "the demand changes no refusal");
    assert_eq!(rep.undecoded, vec!["two".to_string()]);
    assert_eq!(ex.relation("two"), None);
    assert!(all.undecoded.is_empty());
}

/// FAILS IF: a report from another program is accepted: its adorned
/// predicate is not a head of this plan.
#[test]
fn a_report_from_another_program_is_refused() {
    let other = r#"
        #pragma magic_sets = on
        pred edge(u32, u32).
        pred path(u32, u32).
        path(X, Y) :- edge(X, Y).
        path(X, Z) :- path(X, Y), edge(Y, Z).
        ?- path(1, Y).
    "#;
    let e: &[(u32, u32)] = &[(1, 2), (2, 3)];
    let got = run_magic(MAGIC_REACH, other, &[("edge", e)]).err();
    assert!(
        matches!(got, Some(xlog_cube::PlanRefusal::DemandMismatch(ref p)) if p == "path"),
        "{got:?}"
    );
}
