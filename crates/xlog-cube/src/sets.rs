//! One-column heads on the cube (D-XLOG-W5-C).
//!
//! A one-column relation is a stored set in the cube
//! (`lance_graph_xlog::cube::Cube::add_set`), and a one-column head is a
//! `SetExpr` over stored sets, never stored itself. Two shapes are routed
//! here; every other SCC takes the existing paths unchanged.
//!
//! - **Set**: a non-recursive SCC whose rules all have a one-column head and
//!   a body of one-column set atoms on the head's variable:
//!   `h(X) :- a(X), b(X).` → `a ∧ b`, rules or'ed.
//! - **Reach**: a recursive SCC with one one-column member `h`, base rules
//!   of the Set shape, and exactly one step rule
//!   `h(Y) :- h(X), e(X, Y), f1(Y), …` where `e` is a binary relation in the
//!   cube and at least one `f` is a set: `Reach { e, seed: or(base),
//!   target: and(f) }`.
//!
//! A non-recursive SCC whose rules do not have the Set shape is also routed
//! when every rule body is a **one-column set term** (EXO-394): a one-column
//! scan, a column of a binary relation in the cube (`h(X) :- r(_, X).` is
//! `Project(Scan r, [1])`, a `SetExpr::Column`), a `Diff` of two terms (XLOG's
//! lowering of `not`), or a semi-join of two terms on column 0. So
//! `no_in(X) :- node(X), not has_in(X).` is `node ∧ ¬col(edge, 1)` in one
//! terminal. Binary differences (`keep(X, Y) :- pair(X, Y), not
//! blocked(X, Y).`) and nullary heads are not one-column terms and keep the
//! existing paths.
//!
//! A head read later is composed from its expression, so `rollback(X) :-
//! replay(X), blocked(X).` evaluates `Reach(..) ∧ blocked` inside one
//! terminal. The rows a head reports are downloaded from
//! `Cube::set_mask`, which is an answer, never an input.

use std::collections::{BTreeMap, HashMap};

use lance_graph_xlog::cube::SetExpr;
use xlog_core::RelId;
use xlog_ir::{JoinType, ProjectExpr, RirNode};

use super::{CubeExecutor, Label, Scc};

/// Which routed path an SCC took.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Route {
    Set,
    Reach,
}

impl Route {
    pub(crate) fn name(self) -> &'static str {
        match self {
            Route::Set => "set",
            Route::Reach => "reach",
        }
    }
}

/// A conjunctive body: each atom's relation and the variable class of each
/// of its columns, and the head's classes.
struct Conj {
    atoms: Vec<(RelId, Vec<usize>)>,
    head: Vec<usize>,
}

/// The conjunctive form of a body, read from its fallback where it has one.
/// `None` for anything that is not scans joined by inner equi-joins under a
/// column projection (filters, constants, negation, aggregates).
fn conj(body: &RirNode, arity: &dyn Fn(RelId) -> Option<usize>) -> Option<Conj> {
    let node = match body {
        RirNode::ChainJoin { fallback, .. } | RirNode::MultiWayJoin { fallback, .. } => {
            fallback.as_ref()
        }
        other => other,
    };
    let (input, columns): (&RirNode, Option<&[ProjectExpr]>) = match node {
        RirNode::Project { input, columns } => (input.as_ref(), Some(columns.as_slice())),
        RirNode::Scan { .. } => (node, None),
        _ => return None,
    };
    fn walk(
        n: &RirNode,
        arity: &dyn Fn(RelId) -> Option<usize>,
        atoms: &mut Vec<(RelId, usize, usize)>,
        eqs: &mut Vec<(usize, usize)>,
        offset: usize,
    ) -> Option<usize> {
        match n {
            RirNode::Scan { rel } => {
                let a = arity(*rel)?;
                atoms.push((*rel, offset, a));
                Some(a)
            }
            RirNode::Join {
                left,
                right,
                left_keys,
                right_keys,
                join_type: JoinType::Inner,
            } => {
                let wl = walk(left, arity, atoms, eqs, offset)?;
                let wr = walk(right, arity, atoms, eqs, offset + wl)?;
                for (&l, &r) in left_keys.iter().zip(right_keys) {
                    if l >= wl || r >= wr {
                        return None;
                    }
                    eqs.push((offset + l, offset + wl + r));
                }
                Some(wl + wr)
            }
            _ => None,
        }
    }
    let mut raw = Vec::new();
    let mut eqs = Vec::new();
    let width = walk(input, arity, &mut raw, &mut eqs, 0)?;
    let mut parent: Vec<usize> = (0..width).collect();
    fn root(p: &mut [usize], mut x: usize) -> usize {
        while p[x] != x {
            p[x] = p[p[x]];
            x = p[x];
        }
        x
    }
    for (a, b) in eqs {
        let (ra, rb) = (root(&mut parent, a), root(&mut parent, b));
        parent[ra] = rb;
    }
    let class: Vec<usize> = (0..width).map(|c| root(&mut parent, c)).collect();
    let head = match columns {
        Some(cols) => cols
            .iter()
            .map(|c| match c {
                ProjectExpr::Column(i) if *i < width => Some(class[*i]),
                _ => None,
            })
            .collect::<Option<Vec<_>>>()?,
        None => class.clone(),
    };
    Some(Conj {
        atoms: raw
            .into_iter()
            .map(|(rel, off, a)| (rel, class[off..off + a].to_vec()))
            .collect(),
        head,
    })
}

/// The set expression of a body whose result has exactly one column, or
/// `None`. Leaves: a one-column relation with an expression (a stored set or
/// a routed head), or column `c` of a binary relation the cube holds.
/// Members of the SCC are never read: their expressions are not built yet.
fn set_term(
    node: &RirNode,
    labels: &HashMap<RelId, Label>,
    exprs: &HashMap<Label, SetExpr>,
    current: &HashMap<Label, u32>,
    members: &[Label],
    arity: &dyn Fn(RelId) -> Option<usize>,
) -> Option<SetExpr> {
    let leaf = |rel: &RelId| -> Option<Label> {
        let l = *labels.get(rel)?;
        (!members.contains(&l)).then_some(l)
    };
    let go = |n: &RirNode| set_term(n, labels, exprs, current, members, arity);
    match node {
        RirNode::ChainJoin { fallback, .. } | RirNode::MultiWayJoin { fallback, .. } => {
            go(fallback.as_ref())
        }
        RirNode::Scan { rel } if arity(*rel)? == 1 => exprs.get(&leaf(rel)?).cloned(),
        RirNode::Project { input, columns } => {
            let [ProjectExpr::Column(c)] = columns.as_slice() else {
                return None;
            };
            match input.as_ref() {
                RirNode::Scan { rel } if arity(*rel)? == 2 && *c < 2 => {
                    let id = *current.get(&leaf(rel)?)?;
                    Some(SetExpr::column(id, *c as u8))
                }
                other if *c == 0 => go(other),
                _ => None,
            }
        }
        RirNode::Diff { left, right } => Some(SetExpr::and_not(go(left)?, go(right)?)),
        RirNode::Join {
            left,
            right,
            left_keys,
            right_keys,
            join_type: JoinType::Semi,
        } if left_keys.as_slice() == [0] && right_keys.as_slice() == [0] => {
            Some(SetExpr::and(go(left)?, go(right)?))
        }
        _ => None,
    }
}

fn or_all(mut es: Vec<SetExpr>) -> Option<SetExpr> {
    let first = es.pop()?;
    Some(es.into_iter().fold(first, |acc, e| SetExpr::or(e, acc)))
}

fn and_all(mut es: Vec<SetExpr>) -> Option<SetExpr> {
    let first = es.pop()?;
    Some(es.into_iter().fold(first, |acc, e| SetExpr::and(e, acc)))
}

impl CubeExecutor {
    /// A Set-shape body: every atom a one-column set on the head's variable,
    /// none of them a member of the SCC. Its expression is their `and`.
    fn set_body(
        &self,
        c: &Conj,
        exprs: &HashMap<Label, SetExpr>,
        members: &[Label],
        arity: &dyn Fn(RelId) -> Option<usize>,
    ) -> Option<SetExpr> {
        let [v] = c.head.as_slice() else {
            return None;
        };
        let mut parts = Vec::with_capacity(c.atoms.len());
        for (rel, cols) in &c.atoms {
            let l = *self.labels.get(rel)?;
            if arity(*rel)? != 1 || cols[0] != *v || members.contains(&l) {
                return None;
            }
            parts.push(exprs.get(&l)?.clone());
        }
        and_all(parts)
    }

    /// The route and the head expressions for `scc`, or `None` when the SCC
    /// is not one of the two shapes and takes the existing paths.
    pub(crate) fn route(
        &self,
        scc: &Scc<'_>,
        exprs: &HashMap<Label, SetExpr>,
        current: &HashMap<Label, u32>,
        arity: &dyn Fn(RelId) -> Option<usize>,
    ) -> Option<(Route, BTreeMap<Label, SetExpr>)> {
        let members: Vec<Label> = scc.members.iter().copied().collect();
        // Facts a host stored for a head are part of its answer.
        let put = |h: Label| exprs.get(&h).cloned();
        if !scc.recursive {
            let mut per_head: BTreeMap<Label, Vec<SetExpr>> = BTreeMap::new();
            for (head, body) in &scc.rules {
                let e = conj(body, arity)
                    .and_then(|c| self.set_body(&c, exprs, &members, arity))
                    .or_else(|| set_term(body, &self.labels, exprs, current, &members, arity))?;
                per_head.entry(*head).or_default().push(e);
            }
            let out = per_head
                .into_iter()
                .map(|(h, mut es)| {
                    es.extend(put(h));
                    Some((h, or_all(es)?))
                })
                .collect::<Option<BTreeMap<_, _>>>()?;
            return Some((Route::Set, out));
        }
        let [h] = members.as_slice() else {
            return None;
        };
        let mut base = Vec::new();
        let mut step: Option<(u32, SetExpr)> = None;
        for (head, body) in &scc.rules {
            if head != h {
                return None;
            }
            let c = conj(body, arity)?;
            let reads_h = c
                .atoms
                .iter()
                .any(|(rel, _)| self.labels.get(rel) == Some(h));
            if !reads_h {
                base.push(self.set_body(&c, exprs, &members, arity)?);
                continue;
            }
            if step.is_some() {
                return None;
            }
            let [y] = c.head.as_slice() else {
                return None;
            };
            let (mut x, mut edge, mut filters) = (None, None, Vec::new());
            for (rel, cols) in &c.atoms {
                let l = *self.labels.get(rel)?;
                match (l == *h, cols.as_slice()) {
                    (true, [v]) if x.is_none() => x = Some(*v),
                    (false, [a, b]) if edge.is_none() && !exprs.contains_key(&l) => {
                        edge = Some((l, *a, *b));
                    }
                    (false, [v]) if v == y => filters.push(exprs.get(&l)?.clone()),
                    _ => return None,
                }
            }
            let (x, (e, a, b)) = (x?, edge?);
            if a != x || b != *y || x == *y {
                return None;
            }
            step = Some((*current.get(&e)?, and_all(filters)?));
        }
        let (rel, target) = step?;
        base.extend(put(*h));
        let seed = or_all(base)?;
        let mut out = BTreeMap::new();
        out.insert(*h, SetExpr::reach(rel, seed, target));
        Some((Route::Reach, out))
    }
}
