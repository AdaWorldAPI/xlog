//! XLOG's compiled plans on lance-graph's resident hypercube.
//!
//! [`CubeExecutor`] takes the same inputs as `xlog_runtime::Executor` (a
//! compiled `ExecutionPlan`, relation ids, base relations by name) and the
//! same output (relations by name), but executes on one
//! `lance_graph_xlog::cube::Cube` instead of CUDA. No device, no kernels,
//! no `xlog-cuda` dependency.
//!
//! One cube is built per [`CubeExecutor::execute_plan`] over every base
//! relation: each id is looked up once. A rule body becomes cube views
//! (`scan`, `bind`, `join`); rows are built where a rule's head relation is
//! produced. Each SCC's binary heads are added to the cube when the SCC is
//! complete, so later rules read derived relations like base ones.
//!
//! Recursive SCCs run semi-naively: a round evaluates each recursive rule
//! once per SCC atom in its body, with that atom reading the previous
//! round's new rows (the delta) and every other atom the full relation. The
//! SCC stops when a round derives nothing new.
//!
//! Admitted rule bodies (binary `u32` relations only):
//!
//! - `Scan(r)`, `Filter { Scan(r), col == const }`, and `Project` over
//!   either (single-relation bodies: read from the rows recorded for that
//!   relation, no cube program);
//! - `ChainJoin { l, r, left_key, right_key, output_columns }` and
//!   `Project { Join { l, r, [k], [k], Inner }, columns }`, where each side is
//!   `Scan(rel)` or `Filter { Scan(rel), col == const }` on the side's free
//!   column.
//!
//! Every other body is refused and recorded in [`Report::refused`] with its
//! reason. A refused rule refuses its whole SCC: a head is never handed back
//! with only some of its rules applied.

use std::collections::{BTreeMap, BTreeSet, HashMap};

use lance_graph_dt64::dr::{LocalMap, MapTable, Width};
use lance_graph_xlog::cube::{Cube, CubeRefusal, CubeStats, View};
use xlog_core::RelId;
use xlog_ir::{
    CompareOp, CompiledRule, ConstValue, ExecutionPlan, Expr, JoinType, ProjectExpr, RirNode,
};

/// The one tile the probe uses.
const TILE: u32 = 0;

/// What one `execute_plan` did.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Report {
    /// Rules run on the cube, by head.
    pub ran: Vec<String>,
    /// Rules refused: `(head, reason)`.
    pub refused: Vec<(String, String)>,
    /// Semi-naive rounds that derived new rows, over all recursive SCCs.
    pub rounds: usize,
    /// The cube's counters after the plan.
    pub stats: CubeStats,
}

/// Why the plan could not run.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum PlanRefusal {
    /// The base relations hold more distinct ids than one tile (256).
    Domain,
    /// The cube refused its intake.
    Cube(CubeRefusal),
    /// A completed SCC's head could not be added to the cube.
    Store(String),
}

/// The executor. See the module docs.
#[derive(Default)]
pub struct CubeExecutor {
    names: HashMap<RelId, String>,
    base: BTreeMap<String, Vec<(u32, u32)>>,
    out: BTreeMap<String, Rows>,
}

type Rows = BTreeSet<Vec<u32>>;

/// Which cube relation an atom reads, given the relation's name and the
/// atom's position in the body.
type Resolve<'a> = dyn Fn(&str, usize) -> Result<u32, String> + 'a;

/// One side of a join: a cube relation, and an optional bound
/// `(column, value)`.
#[derive(Clone, Copy)]
struct Side {
    rel: u32,
    bound: Option<(usize, u32)>,
}

/// The cube, its map table, and the rows of every relation added to it. A
/// copy rule (`p :- q`) reads those rows; every other body runs on the cube.
struct Store {
    cube: Cube,
    table: MapTable,
    rows: HashMap<u32, Rows>,
}

impl Store {
    /// Add binary rows as a cube relation; `None` for rows of another arity.
    fn add(&mut self, rows: &Rows) -> Result<Option<u32>, String> {
        let Some(pairs) = binary(rows) else {
            return Ok(None);
        };
        let id = self
            .cube
            .add_relation(&self.table, &pairs)
            .map_err(|e| format!("{e:?}"))?;
        self.rows.insert(id, rows.clone());
        Ok(Some(id))
    }
}

impl CubeExecutor {
    pub fn new() -> Self {
        Self::default()
    }

    /// As `Executor::register_relation`.
    pub fn register_relation(&mut self, rel: RelId, name: &str) {
        self.names.insert(rel, name.to_string());
    }

    /// As `Executor::put_relation`, for a binary `u32` relation given as
    /// rows.
    pub fn put_relation(&mut self, name: &str, rows: &[(u32, u32)]) {
        self.base.insert(name.to_string(), rows.to_vec());
    }

    /// A relation's rows: a derived relation the plan produced, or a base
    /// relation. The download boundary.
    pub fn relation(&self, name: &str) -> Option<Vec<Vec<u32>>> {
        if let Some(rows) = self.out.get(name) {
            return Some(rows.iter().cloned().collect());
        }
        self.base
            .get(name)
            .map(|r| pairs_to_rows(r).into_iter().collect())
    }

    /// Run every SCC of every stratum, in order, on one cube.
    pub fn execute_plan(&mut self, plan: &ExecutionPlan) -> Result<Report, PlanRefusal> {
        let order: Vec<&String> = self.base.keys().collect();
        let rels: Vec<&[(u32, u32)]> = order.iter().map(|n| self.base[*n].as_slice()).collect();
        let mut ids: Vec<u64> = rels
            .iter()
            .flat_map(|r| r.iter().flat_map(|&(a, b)| [u64::from(a), u64::from(b)]))
            .collect();
        ids.sort_unstable();
        ids.dedup();
        let map = LocalMap::prepare(Width::W256, &ids).map_err(|_| PlanRefusal::Domain)?;
        let mut table = MapTable::new(1, 1);
        table.advance(1, 1, vec![(TILE, Some(map))]);
        let cube = Cube::new(&table, &table.seal(), TILE, &rels).map_err(PlanRefusal::Cube)?;
        let mut store = Store {
            cube,
            table,
            rows: order
                .iter()
                .enumerate()
                .map(|(i, n)| (i as u32, pairs_to_rows(&self.base[*n])))
                .collect(),
        };
        // Relation name -> the cube relation holding its current contents.
        let mut current: HashMap<String, u32> = order
            .iter()
            .enumerate()
            .map(|(i, n)| ((*n).clone(), i as u32))
            .collect();

        let mut report = Report::default();
        let mut produced: BTreeMap<String, Rows> = BTreeMap::new();
        for stratum in &plan.strata {
            for &scc in &stratum.sccs {
                let Some(rules) = plan.rules_by_scc.get(scc as usize) else {
                    continue;
                };
                if rules.is_empty() {
                    continue;
                }
                let info = plan.sccs.get(scc as usize);
                let members: BTreeSet<String> = info
                    .map(|s| s.predicates.iter().cloned().collect())
                    .unwrap_or_default();
                let outcome = if info.is_some_and(|s| s.is_recursive) {
                    self.recursive_scc(&mut store, &current, rules, &members, &mut report.rounds)
                } else {
                    self.plain_scc(&mut store, &current, rules)
                };
                match outcome {
                    Ok(heads) => {
                        report.ran.extend(rules.iter().map(|r| r.head.clone()));
                        for (head, rows) in heads {
                            if let Some(id) = store.add(&rows).map_err(PlanRefusal::Store)? {
                                current.insert(head.clone(), id);
                            }
                            produced.insert(head, rows);
                        }
                    }
                    Err(reason) => report
                        .refused
                        .extend(rules.iter().map(|r| (r.head.clone(), reason.clone()))),
                }
            }
        }
        self.out = produced;
        report.stats = store.cube.stats();
        Ok(report)
    }

    /// Every rule of a non-recursive SCC, once.
    fn plain_scc(
        &self,
        store: &mut Store,
        current: &HashMap<String, u32>,
        rules: &[CompiledRule],
    ) -> Result<BTreeMap<String, Rows>, String> {
        let resolve = |name: &str, _: usize| lookup(current, name);
        let mut heads: BTreeMap<String, Rows> = BTreeMap::new();
        for rule in rules {
            let rows = self.rule(store, &resolve, &rule.body)?;
            heads.entry(rule.head.clone()).or_default().extend(rows);
        }
        Ok(heads)
    }

    /// Semi-naive evaluation of a recursive SCC (see the module docs).
    /// Terminates: every row is a pair over the map's ids, and a round that
    /// adds no row ends the loop.
    fn recursive_scc(
        &self,
        store: &mut Store,
        outer: &HashMap<String, u32>,
        rules: &[CompiledRule],
        members: &BTreeSet<String>,
        rounds: &mut usize,
    ) -> Result<BTreeMap<String, Rows>, String> {
        // Rules reading no SCC member seed the first delta; the others are
        // recursive, with the positions of their SCC atoms.
        let mut delta: BTreeMap<String, Rows> = BTreeMap::new();
        let mut recursive: Vec<(&CompiledRule, Vec<usize>)> = Vec::new();
        for rule in rules {
            let atoms = self.atoms(&rule.body)?;
            let inside: Vec<usize> = (0..atoms.len())
                .filter(|&i| members.contains(&atoms[i]))
                .collect();
            if inside.is_empty() {
                let resolve = |name: &str, _: usize| lookup(outer, name);
                let rows = self.rule(store, &resolve, &rule.body)?;
                delta.entry(rule.head.clone()).or_default().extend(rows);
            } else {
                recursive.push((rule, inside));
            }
        }

        let mut full: BTreeMap<String, Rows> =
            members.iter().map(|m| (m.clone(), Rows::new())).collect();
        let mut current = outer.clone();
        loop {
            // Keep only rows the full relation lacks; fold them in.
            let mut delta_ids: HashMap<String, u32> = HashMap::new();
            let mut grew = false;
            for m in members {
                let new: Rows = delta
                    .remove(m)
                    .unwrap_or_default()
                    .into_iter()
                    .filter(|r| !full[m].contains(r))
                    .collect();
                grew |= !new.is_empty();
                full.get_mut(m)
                    .expect("every member has a full relation")
                    .extend(new.iter().cloned());
                let not_binary = || format!("`{m}` is recursive and not binary");
                delta_ids.insert(m.clone(), store.add(&new)?.ok_or_else(not_binary)?);
                current.insert(m.clone(), store.add(&full[m])?.ok_or_else(not_binary)?);
            }
            if !grew {
                return Ok(full);
            }
            *rounds += 1;
            for (rule, inside) in &recursive {
                for &at in inside {
                    let resolve = |name: &str, pos: usize| {
                        if pos == at {
                            lookup(&delta_ids, name)
                        } else {
                            lookup(&current, name)
                        }
                    };
                    let rows = self.rule(store, &resolve, &rule.body)?;
                    delta.entry(rule.head.clone()).or_default().extend(rows);
                }
            }
        }
    }

    /// The relation names a body reads, by atom position.
    fn atoms(&self, body: &RirNode) -> Result<Vec<String>, String> {
        fn atom(n: &RirNode) -> Option<RelId> {
            match n {
                RirNode::Scan { rel } => Some(*rel),
                RirNode::Filter { input, .. } => atom(input),
                _ => None,
            }
        }
        let rels = match body {
            RirNode::Scan { .. } | RirNode::Filter { .. } => vec![atom(body)],
            RirNode::ChainJoin { left, right, .. } => vec![atom(left), atom(right)],
            RirNode::Project { input, .. } => match input.as_ref() {
                RirNode::Join { left, right, .. } => vec![atom(left), atom(right)],
                RirNode::Scan { .. } | RirNode::Filter { .. } => vec![atom(input)],
                other => return Err(format!("Project over {}", kind(other))),
            },
            other => return Err(format!("{} body", kind(other))),
        };
        rels.into_iter()
            .map(|r| {
                let r = r.ok_or_else(|| "join side other than a scan".to_string())?;
                Ok(self.name(r)?.to_string())
            })
            .collect()
    }

    fn rule(
        &self,
        store: &mut Store,
        resolve: &Resolve<'_>,
        body: &RirNode,
    ) -> Result<Vec<Vec<u32>>, String> {
        match body {
            RirNode::Scan { .. } | RirNode::Filter { .. } => self.single(store, resolve, body),
            RirNode::Project { input, columns }
                if matches!(
                    input.as_ref(),
                    RirNode::Scan { .. } | RirNode::Filter { .. }
                ) =>
            {
                let cols = columns
                    .iter()
                    .map(|c| match c {
                        ProjectExpr::Column(i) if *i < 2 => Ok(*i),
                        _ => Err("computed or out-of-range output column".to_string()),
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                Ok(self
                    .single(store, resolve, input)?
                    .into_iter()
                    .map(|row| cols.iter().map(|&i| row[i]).collect())
                    .collect())
            }
            RirNode::ChainJoin {
                left,
                right,
                left_key,
                right_key,
                output_columns,
                ..
            } => self.join(
                store,
                resolve,
                (left, *left_key),
                (right, *right_key),
                output_columns,
            ),
            RirNode::Project { input, columns } => match input.as_ref() {
                RirNode::Join {
                    left,
                    right,
                    left_keys,
                    right_keys,
                    join_type: JoinType::Inner,
                } if left_keys.len() == 1 && right_keys.len() == 1 => self.join(
                    store,
                    resolve,
                    (left, left_keys[0]),
                    (right, right_keys[0]),
                    columns,
                ),
                other => Err(format!("Project over {}", kind(other))),
            },
            other => Err(format!("{} body", kind(other))),
        }
    }

    /// A single relation's rows, optionally restricted by `col == const`:
    /// read from the rows recorded for that cube relation. No join, so no
    /// cube program runs.
    fn single(
        &self,
        store: &Store,
        resolve: &Resolve<'_>,
        node: &RirNode,
    ) -> Result<Vec<Vec<u32>>, String> {
        let (rel, bound) = match node {
            RirNode::Scan { rel } => (rel, None),
            RirNode::Filter { input, predicate } => match input.as_ref() {
                RirNode::Scan { rel } => (rel, Some(eq_const(predicate)?)),
                other => return Err(format!("Filter over {}", kind(other))),
            },
            other => return Err(format!("{} body", kind(other))),
        };
        let id = resolve(self.name(*rel)?, 0)?;
        if bound.is_some_and(|(col, _)| col > 1) {
            return Err("filter column outside a binary relation".into());
        }
        Ok(store.rows[&id]
            .iter()
            .filter(|row| bound.is_none_or(|(col, v)| row[col] == v))
            .cloned()
            .collect())
    }

    fn name(&self, rel: RelId) -> Result<&str, String> {
        self.names
            .get(&rel)
            .map(String::as_str)
            .ok_or_else(|| format!("unregistered {rel:?}"))
    }

    fn side(
        &self,
        resolve: &Resolve<'_>,
        pos: usize,
        node: &RirNode,
        key: usize,
    ) -> Result<Side, String> {
        match node {
            RirNode::Scan { rel } => Ok(Side {
                rel: resolve(self.name(*rel)?, pos)?,
                bound: None,
            }),
            RirNode::Filter { input, predicate } => {
                let RirNode::Scan { rel } = input.as_ref() else {
                    return Err(format!("Filter over {}", kind(input)));
                };
                let (col, v) = eq_const(predicate)?;
                if col == key || col > 1 {
                    return Err("bound on the join column".into());
                }
                Ok(Side {
                    rel: resolve(self.name(*rel)?, pos)?,
                    bound: Some((col, v)),
                })
            }
            other => Err(format!("join side {}", kind(other))),
        }
    }

    fn view(store: &mut Store, side: Side) -> Result<View, String> {
        let scan = store.cube.scan(side.rel).map_err(|e| format!("{e:?}"))?;
        match side.bound {
            None => Ok(scan),
            Some((col, v)) => store
                .cube
                .bind(&store.table, scan, col as u8, v)
                .map_err(|e| format!("{e:?}")),
        }
    }

    fn join(
        &self,
        store: &mut Store,
        resolve: &Resolve<'_>,
        (left, lk): (&RirNode, usize),
        (right, rk): (&RirNode, usize),
        columns: &[ProjectExpr],
    ) -> Result<Vec<Vec<u32>>, String> {
        if lk > 1 || rk > 1 {
            return Err("join key outside a binary relation".into());
        }
        let cols = columns
            .iter()
            .map(|c| match c {
                ProjectExpr::Column(i) if *i < 4 => Ok(*i),
                _ => Err("computed or out-of-range output column".to_string()),
            })
            .collect::<Result<Vec<_>, _>>()?;
        let l = self.side(resolve, 0, left, lk)?;
        let r = self.side(resolve, 1, right, rk)?;
        let (lv, rv) = (Self::view(store, l)?, Self::view(store, r)?);
        let j = store
            .cube
            .join(lv, lk as u8, rv, rk as u8)
            .map_err(|e| format!("{e:?}"))?;
        let rows = store
            .cube
            .materialize_rows(&store.table, j)
            .map_err(|e| format!("{e:?}"))?;
        // The join's columns are left ++ right; the cube hands back
        // (left free, key, right free).
        let at = |(lf, key, rf): (u64, u64, u64), i: usize| -> u32 {
            let v = match i {
                i if i == lk => key,
                i if i < 2 => lf,
                i if i - 2 == rk => key,
                _ => rf,
            };
            v as u32
        };
        Ok(rows
            .into_iter()
            .map(|t| cols.iter().map(|&i| at(t, i)).collect())
            .collect())
    }
}

/// `col == const`, the only filter admitted.
fn eq_const(predicate: &Expr) -> Result<(usize, u32), String> {
    let Expr::Compare {
        left,
        op: CompareOp::Eq,
        right,
    } = predicate
    else {
        return Err("filter other than `col == const`".into());
    };
    // Without facts to type the column, XLOG types the constant `U64`.
    match (left.as_ref(), right.as_ref()) {
        (Expr::Column(c), Expr::Const(ConstValue::U32(v))) => Ok((*c, *v)),
        (Expr::Column(c), Expr::Const(ConstValue::U64(v))) => Ok((
            *c,
            u32::try_from(*v).map_err(|_| "constant outside u32".to_string())?,
        )),
        _ => Err("filter other than `col == const`".into()),
    }
}

fn lookup(ids: &HashMap<String, u32>, name: &str) -> Result<u32, String> {
    ids.get(name)
        .copied()
        .ok_or_else(|| format!("`{name}` is not a binary relation in the cube"))
}

/// Rows as pairs, when every row has two columns.
fn binary(rows: &Rows) -> Option<Vec<(u32, u32)>> {
    rows.iter()
        .map(|r| match r.as_slice() {
            [a, b] => Some((*a, *b)),
            _ => None,
        })
        .collect()
}

fn pairs_to_rows(pairs: &[(u32, u32)]) -> Rows {
    pairs.iter().map(|&(a, b)| vec![a, b]).collect()
}

fn kind(n: &RirNode) -> &'static str {
    match n {
        RirNode::Unit => "Unit",
        RirNode::Scan { .. } => "Scan",
        RirNode::Filter { .. } => "Filter",
        RirNode::Project { .. } => "Project",
        RirNode::Join { .. } => "Join",
        RirNode::ChainJoin { .. } => "ChainJoin",
        RirNode::GroupBy { .. } => "GroupBy",
        RirNode::Union { .. } => "Union",
        RirNode::Distinct { .. } => "Distinct",
        RirNode::Diff { .. } => "Diff",
        RirNode::Fixpoint { .. } => "Fixpoint",
        RirNode::MultiWayJoin { .. } => "MultiWayJoin",
        RirNode::TensorMaskedJoin { .. } => "TensorMaskedJoin",
    }
}
