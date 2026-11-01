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
//! (`scan`, `bind`, `join`); the rule's head relation is the one place rows
//! are built, because that is where XLOG's store hands a relation back.
//!
//! Admitted rule bodies (binary `u32` base relations only):
//!
//! - `Scan(r)`;
//! - `ChainJoin { l, r, left_key, right_key, output_columns }` and
//!   `Project { Join { l, r, [k], [k], Inner }, columns }`, where each side is
//!   `Scan(rel)` or `Filter { Scan(rel), col == const }` on the side's free
//!   column.
//!
//! Every other body is refused and recorded in [`Report::refused`] with its
//! reason; its head relation is not produced. A rule reading a relation that
//! is not a base relation (a derived input) is refused: the cube holds base
//! relations only.

use std::collections::{BTreeMap, BTreeSet, HashMap};

use lance_graph_dt64::dr::{LocalMap, MapTable, Width};
use lance_graph_xlog::cube::{Cube, CubeRefusal, CubeStats, View};
use xlog_core::RelId;
use xlog_ir::{CompareOp, ConstValue, ExecutionPlan, Expr, JoinType, ProjectExpr, RirNode};

/// The one tile the probe uses.
const TILE: u32 = 0;

/// What one `execute_plan` did.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Report {
    /// Rules run on the cube, by head.
    pub ran: Vec<String>,
    /// Rules refused: `(head, reason)`.
    pub refused: Vec<(String, String)>,
    /// The cube's counters after the plan.
    pub stats: CubeStats,
}

/// Why the plan could not start at all.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum PlanRefusal {
    /// The base relations hold more distinct ids than one tile (256).
    Domain,
    /// The cube refused its intake.
    Cube(CubeRefusal),
}

/// The executor. See the module docs.
#[derive(Default)]
pub struct CubeExecutor {
    names: HashMap<RelId, String>,
    base: BTreeMap<String, Vec<(u32, u32)>>,
    out: BTreeMap<String, BTreeSet<Vec<u32>>>,
}

/// One side of a join: the base relation's cube index, and an optional bound
/// `(column, value)`.
#[derive(Clone, Copy)]
struct Side {
    rel: u32,
    bound: Option<(usize, u32)>,
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
        self.base.get(name).map(|r| {
            r.iter()
                .map(|&(a, b)| vec![a, b])
                .collect::<BTreeSet<_>>()
                .into_iter()
                .collect()
        })
    }

    /// Run every rule of every stratum, in order, on one cube over the base
    /// relations.
    pub fn execute_plan(&mut self, plan: &ExecutionPlan) -> Result<Report, PlanRefusal> {
        let order: Vec<&String> = self.base.keys().collect();
        let index: HashMap<&str, u32> = order
            .iter()
            .enumerate()
            .map(|(i, n)| (n.as_str(), i as u32))
            .collect();
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
        let mut cube = Cube::new(&table, &table.seal(), TILE, &rels).map_err(PlanRefusal::Cube)?;

        let mut report = Report::default();
        let mut produced: BTreeMap<String, BTreeSet<Vec<u32>>> = BTreeMap::new();
        for stratum in &plan.strata {
            for &scc in &stratum.sccs {
                let Some(rules) = plan.rules_by_scc.get(scc as usize) else {
                    continue;
                };
                let recursive = plan.sccs.get(scc as usize).is_some_and(|s| s.is_recursive);
                for rule in rules {
                    if recursive {
                        report
                            .refused
                            .push((rule.head.clone(), "recursive SCC".into()));
                        continue;
                    }
                    match self.rule(&mut cube, &table, &index, &rule.body) {
                        Ok(rows) => {
                            produced.entry(rule.head.clone()).or_default().extend(rows);
                            report.ran.push(rule.head.clone());
                        }
                        Err(reason) => report.refused.push((rule.head.clone(), reason)),
                    }
                }
            }
        }
        // A head with any refused rule is incomplete: never hand it back.
        for (head, _) in &report.refused {
            produced.remove(head);
        }
        self.out = produced;
        report.stats = cube.stats();
        Ok(report)
    }

    fn rule(
        &self,
        cube: &mut Cube,
        table: &MapTable,
        index: &HashMap<&str, u32>,
        body: &RirNode,
    ) -> Result<Vec<Vec<u32>>, String> {
        match body {
            RirNode::Scan { rel } => {
                let name = self.base_name(*rel, index)?;
                Ok(self.base[name].iter().map(|&(a, b)| vec![a, b]).collect())
            }
            RirNode::ChainJoin {
                left,
                right,
                left_key,
                right_key,
                output_columns,
                ..
            } => self.join(
                cube,
                table,
                index,
                left,
                right,
                *left_key,
                *right_key,
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
                    cube,
                    table,
                    index,
                    left,
                    right,
                    left_keys[0],
                    right_keys[0],
                    columns,
                ),
                other => Err(format!("Project over {}", kind(other))),
            },
            other => Err(format!("{} body", kind(other))),
        }
    }

    fn base_name<'a>(&self, rel: RelId, index: &HashMap<&'a str, u32>) -> Result<&'a str, String> {
        let name = self
            .names
            .get(&rel)
            .ok_or_else(|| format!("unregistered {rel:?}"))?;
        index
            .get_key_value(name.as_str())
            .map(|(k, _)| *k)
            .ok_or_else(|| format!("`{name}` is not a binary base relation"))
    }

    fn side(&self, index: &HashMap<&str, u32>, node: &RirNode, key: usize) -> Result<Side, String> {
        match node {
            RirNode::Scan { rel } => Ok(Side {
                rel: index[self.base_name(*rel, index)?],
                bound: None,
            }),
            RirNode::Filter { input, predicate } => {
                let RirNode::Scan { rel } = input.as_ref() else {
                    return Err(format!("Filter over {}", kind(input)));
                };
                let Expr::Compare {
                    left,
                    op: CompareOp::Eq,
                    right,
                } = predicate
                else {
                    return Err("filter other than `col == const`".into());
                };
                // Without facts to type the column, XLOG types the constant `U64`.
                let (col, v) = match (left.as_ref(), right.as_ref()) {
                    (Expr::Column(c), Expr::Const(ConstValue::U32(v))) => (c, *v),
                    (Expr::Column(c), Expr::Const(ConstValue::U64(v))) => (
                        c,
                        u32::try_from(*v).map_err(|_| "constant outside u32".to_string())?,
                    ),
                    _ => return Err("filter other than `col == const`".into()),
                };
                if *col == key || *col > 1 {
                    return Err("bound on the join column".into());
                }
                Ok(Side {
                    rel: index[self.base_name(*rel, index)?],
                    bound: Some((*col, v)),
                })
            }
            other => Err(format!("join side {}", kind(other))),
        }
    }

    fn view(cube: &mut Cube, table: &MapTable, side: Side) -> Result<View, String> {
        let scan = cube.scan(side.rel).map_err(|e| format!("{e:?}"))?;
        match side.bound {
            None => Ok(scan),
            Some((col, v)) => cube
                .bind(table, scan, col as u8, v)
                .map_err(|e| format!("{e:?}")),
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn join(
        &self,
        cube: &mut Cube,
        table: &MapTable,
        index: &HashMap<&str, u32>,
        left: &RirNode,
        right: &RirNode,
        lk: usize,
        rk: usize,
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
        let (l, r) = (self.side(index, left, lk)?, self.side(index, right, rk)?);
        let (lv, rv) = (Self::view(cube, table, l)?, Self::view(cube, table, r)?);
        let j = cube
            .join(lv, lk as u8, rv, rk as u8)
            .map_err(|e| format!("{e:?}"))?;
        let rows = cube
            .materialize_rows(table, j)
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
        _ => "other",
    }
}
