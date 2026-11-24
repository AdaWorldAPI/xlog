//! EXO-396: wide derived relations kept as recipes, and `!=` over a join as
//! a distinctness pair inside the cube's walk. A rule reading two columns of
//! a wide relation runs the relation's recipes (`Cube::body_pair`); it never
//! reads the wide rows. Every expected answer is written out by hand from the
//! facts; nothing is computed by the code under test.

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

fn refused(r: &Report, head: &str) -> bool {
    r.refused.iter().any(|(h, _)| h == head)
}

/// Literal row set from fixed-width arrays.
fn set<const N: usize>(v: &[[u32; N]]) -> BTreeSet<Vec<u32>> {
    v.iter().map(|r| r.to_vec()).collect()
}

const SG_SRC: &str = r#"
    pred parent(u32, u32).
    pred parent_rev(u32, u32).
    pred all_child_pairs(u32, u32).
    pred sg(u32, u32).
    pred sg_witness(u32, u32, u32, u32).
    pred sg_result(u32, u32).
    sg(X, Y) :- parent(X, P), parent(Y, P).
    sg(X, Y) :- parent(X, A), sg(A, B), parent(Y, B).
    sg_result(X, Y) :- parent(X, P), parent(Y, P).
    sg_witness(W, X, Y, Z) :- parent(W, X), sg(X, Y), parent_rev(Y, Z), all_child_pairs(Z, W).
    sg_result(W, Z) :- sg_witness(W, X, Y, Z).
"#;

#[test]
fn same_generation_result_projects_the_witness() {
    // parent(child, parent): 1->3, 6->3, 7->4, 3->5, 4->5.
    let parent: &[(u32, u32)] = &[(1, 3), (6, 3), (7, 4), (3, 5), (4, 5)];
    // parent_rev is the reverse of parent.
    let parent_rev: &[(u32, u32)] = &[(3, 1), (3, 6), (4, 7), (5, 3), (5, 4)];
    // all_child_pairs(Z, W). (9,9) matches no witness chain.
    let all_child_pairs: &[(u32, u32)] = &[(7, 1), (1, 7), (1, 6), (9, 9)];

    let (ex, r) = run(
        SG_SRC,
        &[],
        &[
            ("parent", parent),
            ("parent_rev", parent_rev),
            ("all_child_pairs", all_child_pairs),
        ],
    );

    // sg, by hand, iterated to fixpoint.
    // Rule 1 (shared parent): parent 3 has children {1,6}; parent 4 has {7};
    //   parent 5 has {3,4}. So (1,1)(1,6)(6,1)(6,6) (7,7) (3,3)(3,4)(4,3)(4,4).
    // Iteration 2, rule 2 (parent(X,A), sg(A,B), parent(Y,B)): A=3 and A=4 each
    //   see B in {3,4}; X in {1,6,7} (their parents are 3,3,4); Y with parent 3
    //   is {1,6}, with parent 4 is {7}. So X,Y both range over {1,6,7}; new
    //   pairs (1,7)(7,1)(6,7)(7,6). A=5 has no sg row (5 is nobody's child).
    // Iteration 3: new sg pairs are among {1,6,7}, none of which is a parent,
    //   so rule 2 derives nothing new. Fixpoint.
    // sg is internal; its 13 rows are not asserted directly, only through the
    // witness and result below.

    // sg_witness(W,X,Y,Z): parent(W,X), sg(X,Y), parent_rev(Y,Z), all_child_pairs(Z,W).
    // all_child_pairs(7,1): W=1, X=3, Y=4 (parent_rev(4,7)), sg(3,4) holds -> (1,3,4,7).
    // all_child_pairs(1,7): W=7, X=4, Y=3 (parent_rev(3,1)), sg(4,3) holds -> (7,4,3,1).
    // all_child_pairs(1,6): W=6, X=3, Y=3 (parent_rev(3,1)), sg(3,3) holds -> (6,3,3,1).
    // all_child_pairs(9,9): no parent_rev(_,9), no tuple.
    let witness = set(&[[1, 3, 4, 7], [7, 4, 3, 1], [6, 3, 3, 1]]);
    assert_eq!(rows(&ex, "sg_witness"), witness);

    // sg_result = rule 1 (siblings) plus the witness projection (W,Z).
    // Rule 1: (1,1)(1,6)(6,1)(6,6) (7,7) (3,3)(3,4)(4,3)(4,4).
    // Witness (W,Z): (1,7) (7,1) (6,1). (6,1) is already a sibling pair;
    // (1,7) and (7,1) are cousins, so they appear only through the witness.
    let result = set(&[
        [1, 1],
        [1, 6],
        [6, 1],
        [6, 6],
        [7, 7],
        [3, 3],
        [3, 4],
        [4, 3],
        [4, 4],
        [1, 7],
        [7, 1],
    ]);
    assert_eq!(rows(&ex, "sg_result"), result);

    // Anti-vacuity: the witness rule contributes pairs rule 1 does not.
    assert!(rows(&ex, "sg_result").contains(&vec![1, 7]));
    assert!(rows(&ex, "sg_result").contains(&vec![7, 1]));
    assert!(!refused(&r, "sg_result"), "{:?}", r.refused);
}

const WIDE_SRC: &str = r#"
    pred a(u32, u32).
    pred b(u32, u32).
    pred w(u32, u32, u32).
    pred h(u32, u32).
    w(A, B, C) :- a(A, B), b(B, C).
    h(C, A) :- w(A, B, C).
"#;

#[test]
fn wide_projection_reorders_columns() {
    let a: &[(u32, u32)] = &[(1, 2), (3, 4), (5, 2)];
    let b: &[(u32, u32)] = &[(2, 10), (2, 11), (4, 12), (9, 13)];
    let (ex, r) = run(WIDE_SRC, &[], &[("a", a), ("b", b)]);

    // w(A,B,C): a(1,2) joins b(2,10),(2,11); a(3,4) joins b(4,12);
    // a(5,2) joins b(2,10),(2,11); b(9,13) has no a with B=9.
    // w = (1,2,10)(1,2,11)(3,4,12)(5,2,10)(5,2,11).
    // h(C,A) swaps the columns: (10,1)(11,1)(12,3)(10,5)(11,5).
    let swapped = set(&[[10, 1], [11, 1], [12, 3], [10, 5], [11, 5]]);
    assert_eq!(rows(&ex, "h"), swapped);

    // Negative twin: the unswapped (A,C) set is a different set.
    // (A,C) = (1,10)(1,11)(3,12)(5,10)(5,11).
    let unswapped = set(&[[1, 10], [1, 11], [3, 12], [5, 10], [5, 11]]);
    assert_ne!(rows(&ex, "h"), unswapped);
    assert!(!refused(&r, "h"), "{:?}", r.refused);
}

#[test]
fn fof_drops_the_diagonal() {
    let src = r#"
        pred friend(u32, u32).
        pred fof(u32, u32).
        fof(X, Z) :- friend(X, Y), friend(Y, Z), X != Z.
    "#;
    let friend: &[(u32, u32)] = &[(1, 2), (2, 1), (2, 3), (3, 4)];
    let (ex, r) = run(src, &[], &[("friend", friend)]);

    // Unfiltered join friend(X,Y), friend(Y,Z):
    //   X=1,Y=2: Z in {1,3} -> (1,1)(1,3)
    //   X=2,Y=1: Z=2        -> (2,2)
    //   X=2,Y=3: Z=4        -> (2,4)
    //   X=3,Y=4: friend(4,_) is empty.
    // Removed by X != Z: (1,1) and (2,2). Kept: (1,3) and (2,4).
    let expected = set(&[[1, 3], [2, 4]]);
    assert_eq!(rows(&ex, "fof"), expected);
    assert!(!rows(&ex, "fof").contains(&vec![1, 1]));
    assert!(!rows(&ex, "fof").contains(&vec![2, 2]));
    assert!(!refused(&r, "fof"), "{:?}", r.refused);
}

#[test]
fn ordering_compare_stays_refused() {
    let src = r#"
        pred r(u32, u32).
        pred h(u32, u32).
        h(X, Z) :- r(X, Y), r(Y, Z), X < Z.
    "#;
    let rel: &[(u32, u32)] = &[(1, 2), (2, 3)];
    let (_, rep) = run(src, &[], &[("r", rel)]);
    assert!(refused(&rep, "h"), "ran: {:?}", rep.ran);
}

#[test]
fn join_over_a_wide_relation_stays_refused() {
    let src = r#"
        pred a(u32, u32).
        pred b(u32, u32).
        pred w(u32, u32, u32).
        pred h(u32, u32).
        w(A, B, C) :- a(A, B), b(B, C).
        h(A, D) :- w(A, B, C), a(C, D).
    "#;
    let a: &[(u32, u32)] = &[(1, 2), (3, 4)];
    let b: &[(u32, u32)] = &[(2, 3)];
    let (_, rep) = run(src, &[], &[("a", a), ("b", b)]);
    assert!(refused(&rep, "h"), "ran: {:?}", rep.ran);
}

/// An empty wide head is still wide: `sg_witness` has no rows when
/// `all_child_pairs` matches no chain, and `sg_result(W, Z) :- sg_witness(..)`
/// must read it as an empty 4-ary relation, not as an empty pair relation
/// whose column 3 is out of range.
#[test]
fn an_empty_wide_head_is_still_projected() {
    let parent: &[(u32, u32)] = &[(1, 3), (6, 3), (7, 4), (3, 5), (4, 5)];
    let parent_rev: &[(u32, u32)] = &[(3, 1), (3, 6), (4, 7), (5, 3), (5, 4)];
    // (9, 9) matches no chain, so sg_witness is empty.
    let all_child_pairs: &[(u32, u32)] = &[(9, 9)];
    let (ex, r) = run(
        SG_SRC,
        &[],
        &[
            ("parent", parent),
            ("parent_rev", parent_rev),
            ("all_child_pairs", all_child_pairs),
        ],
    );
    assert!(!refused(&r, "sg_result"), "refused: {:?}", r.refused);
    assert!(rows(&ex, "sg_witness").is_empty());
    // Rule 1 only: siblings under parents 3 ({1, 6}), 4 ({7}) and 5 ({3, 4}).
    let expected = set(&[
        [1, 1],
        [1, 6],
        [6, 1],
        [6, 6],
        [7, 7],
        [3, 3],
        [3, 4],
        [4, 3],
        [4, 4],
    ]);
    assert_eq!(rows(&ex, "sg_result"), expected);
}

/// `mutual(X, Y, M)` keeps every body variable, so its answer is the
/// witnesses under `X != Y`.
#[test]
fn mutual_keeps_every_variable_and_drops_x_equals_y() {
    let src = r#"
        pred friend(u32, u32).
        pred mutual(u32, u32, u32).
        mutual(X, Y, M) :- friend(X, M), friend(Y, M), X != Y.
    "#;
    // friend(X, M): 1->2, 3->2, 4->2, 1->5, 3->5.
    let friend: &[(u32, u32)] = &[(1, 2), (3, 2), (4, 2), (1, 5), (3, 5)];
    let (ex, r) = run(src, &[], &[("friend", friend)]);
    // M = 2 has friends {1, 3, 4}: ordered pairs with X != Y, six of them.
    // M = 5 has friends {1, 3}: (1,3,5) (3,1,5).
    let expected = set(&[
        [1, 3, 2],
        [1, 4, 2],
        [3, 1, 2],
        [3, 4, 2],
        [4, 1, 2],
        [4, 3, 2],
        [1, 3, 5],
        [3, 1, 5],
    ]);
    assert_eq!(rows(&ex, "mutual"), expected);
    // Negative twin: no X = Y witness survives.
    assert!(rows(&ex, "mutual").iter().all(|w| w[0] != w[1]));
    assert!(!refused(&r, "mutual"), "{:?}", r.refused);
}

/// A wide head that drops a body variable would need a projection of
/// witnesses, which is not built: refused, not answered.
#[test]
fn a_wide_head_dropping_a_variable_is_refused() {
    let src = r#"
        pred friend(u32, u32).
        pred tri(u32, u32, u32).
        tri(X, Y, X) :- friend(X, M), friend(Y, M), X != Y.
    "#;
    let friend: &[(u32, u32)] = &[(1, 2), (3, 2)];
    let (_, r) = run(src, &[], &[("friend", friend)]);
    assert!(refused(&r, "tri"), "ran: {:?}", r.ran);
}

/// Two `!=` in one body: both pairs are cleared in the walk.
#[test]
fn two_distinctness_pairs_in_one_body() {
    let src = r#"
        pred e(u32, u32).
        pred p(u32, u32).
        p(X, Z) :- e(X, Y), e(Y, Z), X != Z, X != Y.
    "#;
    // e: 1->1, 1->2, 2->1, 2->3.
    let e: &[(u32, u32)] = &[(1, 1), (1, 2), (2, 1), (2, 3)];
    let (ex, r) = run(src, &[], &[("e", e)]);
    // Paths X->Y->Z: 1->1->1, 1->1->2, 1->2->1, 1->2->3, 2->1->1, 2->1->2.
    // X != Z drops 1->1->1, 1->2->1, 2->1->2.
    // X != Y drops 1->1->2 (and 1->1->1).
    // Kept: 1->2->3 (1,3) and 2->1->1 (2,1).
    assert_eq!(rows(&ex, "p"), set(&[[1, 3], [2, 1]]));
    assert!(!refused(&r, "p"), "{:?}", r.refused);
}
