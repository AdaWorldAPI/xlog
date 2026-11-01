//! XLOG's compiled plans on lance-graph's resident hypercube.
//!
//! [`CubeExecutor`] takes the same inputs as `xlog_runtime::Executor` (a
//! compiled `ExecutionPlan`, relation ids, base relations by name) and the
//! same output (relations by name), but executes on one
//! `lance_graph_xlog::cube::Cube` instead of CUDA. No device, no kernels,
//! no `xlog-cuda` dependency.
//!
//! Names are codebook labels. At intake every relation is interned once
//! into the program's `lance_graph_contract::codebook::Codebook`: the
//! relations given rows (`put_relation`), the relations XLOG registers, and
//! the heads the plan derives. Execution is keyed by the 1-byte codebook
//! index; names come back out only through `Codebook::label` (`relation`,
//! the [`Report`]).
//!
//! Behaviour is not data. XLOG's term built-ins (`functor/3`, `arg/3`,
//! `=../2`) are XLOG-exclusive behaviour, addressed in lance-graph's
//! `0xFFFF :: 0xF100` lens (`lance_graph_xlog::lens`). XLOG's compiler
//! rewrites each into an atom over a helper relation; a registered name the
//! lens lists resolves to that word's address, is never interned, and a rule
//! reading it is refused with [`Refusal::Behaviour`] (no word executes yet).
//! Every other relation is data. A registered binary data relation nobody
//! supplied rows for is empty, as in Datalog; that includes XLOG's
//! `__xlog_meta_fail`, whose meaning is exactly a relation with no rows.
//! Symbol values are already ids from XLOG's own symbol table, so no string
//! reaches execution.
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
//! - `MultiWayJoin`, read through its `fallback` (scan leaves joined by
//!   inner joins): generic join over adjacency masks on the cube's domain;
//! - `ChainJoin { l, r, left_key, right_key, output_columns }` and
//!   `Project { Join { l, r, [k], [k], Inner }, columns }`, where each side is
//!   `Scan(rel)` or `Filter { Scan(rel), col == const }` on the side's free
//!   column.
//!
//! Every other body is refused and recorded in [`Report::refused`] with its
//! reason. A refused rule refuses its whole SCC: a head is never handed back
//! with only some of its rules applied.

use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::fmt;

use lance_graph_contract::codebook::Codebook;
use lance_graph_dt64::dr::{LocalMap, MapTable, Width};
use lance_graph_dt64::AlgebraRef;
use lance_graph_xlog::cube::{Cube, CubeRefusal, CubeStats, View};
use lance_graph_xlog::lens::word_of;
use xlog_core::RelId;
use xlog_ir::{
    CompareOp, CompiledRule, ConstValue, ExecutionPlan, Expr, JoinType, ProjectExpr, RirNode,
};

/// The one tile the probe uses.
const TILE: u32 = 0;

/// What one `execute_plan` did, by name: written once, after execution.
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

/// Why a rule was not run on the cube.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Refusal {
    /// A body node kind the executor does not admit.
    Body(&'static str),
    /// `Project` over a node kind other than a scan, a filter or a join.
    ProjectOver(&'static str),
    /// A filter over something other than a scan.
    FilterOver(&'static str),
    /// A join side other than a scan or a filtered scan.
    JoinSide(&'static str),
    /// A multi-way fallback node other than `Project`, inner `Join`, `Scan`.
    Fallback(&'static str),
    /// A filter other than `col == const`.
    Filter,
    /// A filter column outside a binary relation.
    FilterColumn,
    /// A constant outside `u32`.
    Constant,
    /// A bound on the join column.
    BoundOnJoinColumn,
    /// A computed or out-of-range output column.
    OutputColumn,
    /// A join key outside its input or outside a binary relation.
    JoinKey,
    /// A relation the plan reads that is not a binary relation in the cube.
    NotBinary(Label),
    /// A relation id with no name: never registered, and not a head.
    Unregistered(RelId),
    /// The relation is an XLOG behaviour word (`0xFFFF :: 0xF100`), not
    /// data; no word executes on the cube yet.
    Behaviour(AlgebraRef),
    /// The cube refused.
    Cube(CubeRefusal),
}

impl fmt::Display for Refusal {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Refusal::Body(k) => write!(f, "{k} body"),
            Refusal::ProjectOver(k) => write!(f, "Project over {k}"),
            Refusal::FilterOver(k) => write!(f, "Filter over {k}"),
            Refusal::JoinSide(k) => write!(f, "join side {k}"),
            Refusal::Fallback(k) => write!(f, "multi-way fallback {k}"),
            Refusal::Filter => write!(f, "filter other than `col == const`"),
            Refusal::FilterColumn => write!(f, "filter column outside a binary relation"),
            Refusal::Constant => write!(f, "constant outside u32"),
            Refusal::BoundOnJoinColumn => write!(f, "bound on the join column"),
            Refusal::OutputColumn => write!(f, "computed or out-of-range output column"),
            Refusal::JoinKey => write!(f, "join key outside its input"),
            Refusal::NotBinary(l) => write!(f, "label {l} is not a binary relation in the cube"),
            Refusal::Unregistered(r) => write!(f, "unregistered {r:?}"),
            Refusal::Behaviour(w) => write!(f, "XLOG word {:#010x}/{}", w.lens, w.word),
            Refusal::Cube(e) => write!(f, "cube: {e:?}"),
        }
    }
}

/// Why the plan could not run.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum PlanRefusal {
    /// The base relations hold more distinct ids than one tile (256).
    Domain,
    /// The cube refused its intake or a completed head.
    Cube(CubeRefusal),
    /// More than 255 data relations: the program's codebook is full and the
    /// vocabulary would have to split.
    Vocabulary,
}

/// The executor. See the module docs.
#[derive(Default)]
pub struct CubeExecutor {
    /// XLOG's `RelId` -> name pairing, as registered. Read only at intake.
    registered: HashMap<RelId, String>,
    base: BTreeMap<String, Vec<(u32, u32)>>,
    /// The program's data relations, interned at intake.
    vocab: Codebook,
    /// `RelId` -> codebook label, for the relations that are data.
    labels: HashMap<RelId, Label>,
    /// `RelId` -> XLOG behaviour word, for the relations that are behaviour.
    words: HashMap<RelId, AlgebraRef>,
    /// Registered binary relations nobody supplied: empty, at intake.
    empty: BTreeSet<String>,
    out: BTreeMap<Label, Rows>,
}

/// A data relation's identity: its 1-byte index in the program's codebook.
type Label = u8;

type Rows = BTreeSet<Vec<u32>>;

/// Which cube relation an atom reads, given the relation and the atom's
/// position in the body.
type Resolve<'a> = dyn Fn(Label, usize) -> Result<u32, Refusal> + 'a;

/// A flattened multi-way fallback: scan leaves, each leaf column's variable
/// class, and the projected leaf columns.
type Flat = (Vec<RelId>, Vec<usize>, Vec<usize>);

/// One side of a join: a cube relation, and an optional bound
/// `(column, value)`.
#[derive(Clone, Copy)]
struct Side {
    rel: u32,
    bound: Option<(usize, u32)>,
}

/// The cube, its map table, and the rows of every relation added to it. A
/// single-relation body reads those rows; every other body runs on the cube.
struct Store {
    cube: Cube,
    table: MapTable,
    rows: HashMap<u32, Rows>,
}

impl Store {
    /// Add binary rows as a cube relation; `None` for rows of another arity.
    fn add(&mut self, rows: &Rows) -> Result<Option<u32>, CubeRefusal> {
        let Some(pairs) = binary(rows) else {
            return Ok(None);
        };
        let id = self.cube.add_relation(&self.table, &pairs)?;
        self.rows.insert(id, rows.clone());
        Ok(Some(id))
    }
}

/// One SCC, resolved to ids before it runs.
struct Scc<'p> {
    members: BTreeSet<Label>,
    rules: Vec<(Label, &'p RirNode)>,
    recursive: bool,
}

impl CubeExecutor {
    pub fn new() -> Self {
        Self::default()
    }

    /// As `Executor::register_relation`.
    pub fn register_relation(&mut self, rel: RelId, name: &str) {
        self.registered.insert(rel, name.to_string());
    }

    /// As `Executor::put_relation`, for a binary `u32` relation given as
    /// rows.
    pub fn put_relation(&mut self, name: &str, rows: &[(u32, u32)]) {
        self.base.insert(name.to_string(), rows.to_vec());
    }

    /// The codebook label of a data relation, after [`Self::execute_plan`]
    /// interned it. `None` for a name that is not data, which includes the
    /// behaviour words of the XLOG lens.
    pub fn label_of(&self, name: &str) -> Option<u8> {
        self.vocab.index_of(name)
    }

    /// A relation's rows: a derived relation the plan produced, or a base
    /// relation. The download boundary.
    pub fn relation(&self, name: &str) -> Option<Vec<Vec<u32>>> {
        if let Some(rows) = self.vocab.index_of(name).and_then(|l| self.out.get(&l)) {
            return Some(rows.iter().cloned().collect());
        }
        self.base
            .get(name)
            .map(|r| pairs_to_rows(r).into_iter().collect())
    }

    /// Run every SCC of every stratum, in order, on one cube.
    pub fn execute_plan(&mut self, plan: &ExecutionPlan) -> Result<Report, PlanRefusal> {
        // Boundary: intern the data relations, once.
        self.intern(plan)?;
        let none: &[(u32, u32)] = &[];
        let base: Vec<(Label, &[(u32, u32)])> = self
            .base
            .iter()
            .map(|(n, r)| (n, r.as_slice()))
            .chain(self.empty.iter().map(|n| (n, none)))
            .filter_map(|(n, r)| self.vocab.index_of(n).map(|l| (l, r)))
            .collect();
        let rels: Vec<&[(u32, u32)]> = base.iter().map(|(_, r)| *r).collect();
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
            rows: base
                .iter()
                .enumerate()
                .map(|(i, (_, r))| (i as u32, pairs_to_rows(r)))
                .collect(),
        };
        let mut current: HashMap<Label, u32> = base
            .iter()
            .enumerate()
            .map(|(i, (id, _))| (*id, i as u32))
            .collect();
        let sccs = self.resolve_sccs(plan);

        // Execution: codebook labels and cube ids only.
        let mut ran: Vec<Label> = Vec::new();
        let mut refused: Vec<(Label, Refusal)> = Vec::new();
        let mut rounds = 0;
        let mut produced: BTreeMap<Label, Rows> = BTreeMap::new();
        for scc in &sccs {
            let outcome = if scc.recursive {
                self.recursive_scc(&mut store, &current, scc, &mut rounds)
            } else {
                self.plain_scc(&mut store, &current, scc)
            };
            match outcome {
                Ok(heads) => {
                    ran.extend(scc.rules.iter().map(|(h, _)| *h));
                    for (head, rows) in heads {
                        if let Some(id) = store.add(&rows).map_err(PlanRefusal::Cube)? {
                            current.insert(head, id);
                        }
                        produced.insert(head, rows);
                    }
                }
                Err(reason) => {
                    refused.extend(scc.rules.iter().map(|(h, _)| (*h, reason.clone())));
                }
            }
        }
        self.out = produced;

        // Boundary: labels to names, through the codebook.
        let name = |l: &Label| self.vocab.label(*l).unwrap_or_default().to_string();
        Ok(Report {
            ran: ran.iter().map(name).collect(),
            refused: refused
                .iter()
                .map(|(h, r)| (name(h), self.describe(r)))
                .collect(),
            rounds,
            stats: store.cube.stats(),
        })
    }

    /// A refusal's text, with labels written as names.
    fn describe(&self, r: &Refusal) -> String {
        match r {
            Refusal::NotBinary(l) => match self.vocab.label(*l) {
                Some(n) => format!("`{n}` is not a binary relation in the cube"),
                None => r.to_string(),
            },
            other => other.to_string(),
        }
    }

    /// Intern every relation into a fresh codebook: the relations given
    /// rows, the registered ones, then the heads the plan derives. Registered
    /// binary relations nobody supplied are given no rows: they are empty.
    fn intern(&mut self, plan: &ExecutionPlan) -> Result<(), PlanRefusal> {
        let mut vocab = Codebook::new();
        // Names the XLOG lens lists are behaviour: addressed, not interned.
        self.words = self
            .registered
            .iter()
            .filter_map(|(rel, name)| word_of(name).map(|w| (*rel, w)))
            .collect();
        let mut registered: Vec<(&RelId, &String)> = self
            .registered
            .iter()
            .filter(|(rel, _)| !self.words.contains_key(*rel))
            .collect();
        registered.sort();
        let heads = plan
            .rules_by_scc
            .iter()
            .flatten()
            .map(|r| &r.head)
            .filter(|h| word_of(h).is_none());
        let names = self
            .base
            .keys()
            .chain(registered.iter().map(|(_, n)| *n))
            .chain(heads);
        for name in names {
            vocab.intern(name).ok_or(PlanRefusal::Vocabulary)?;
        }
        self.empty = registered
            .iter()
            .filter(|(rel, name)| {
                plan.rel_arities.get(*rel) == Some(&2) && !self.base.contains_key(*name)
            })
            .map(|(_, name)| (*name).clone())
            .collect();
        self.labels = self
            .registered
            .iter()
            .filter_map(|(rel, name)| vocab.index_of(name).map(|l| (*rel, l)))
            .collect();
        self.vocab = vocab;
        Ok(())
    }

    /// The label of a relation a body reads.
    fn label(&self, rel: RelId) -> Result<Label, Refusal> {
        self.labels
            .get(&rel)
            .copied()
            .ok_or_else(|| match self.words.get(&rel) {
                Some(w) => Refusal::Behaviour(*w),
                None => Refusal::Unregistered(rel),
            })
    }

    /// The plan's SCCs in stratum order, with heads and members as ids.
    fn resolve_sccs<'p>(&self, plan: &'p ExecutionPlan) -> Vec<Scc<'p>> {
        let id = |n: &String| self.vocab.index_of(n);
        plan.strata
            .iter()
            .flat_map(|s| s.sccs.iter())
            .filter_map(|&scc| {
                let rules = plan.rules_by_scc.get(scc as usize)?;
                if rules.is_empty() {
                    return None;
                }
                let info = plan.sccs.get(scc as usize);
                Some(Scc {
                    members: info
                        .map(|s| s.predicates.iter().filter_map(id).collect())
                        .unwrap_or_default(),
                    rules: rules
                        .iter()
                        .filter_map(|r: &CompiledRule| id(&r.head).map(|h| (h, &r.body)))
                        .collect(),
                    recursive: info.is_some_and(|s| s.is_recursive),
                })
            })
            .collect()
    }

    /// Every rule of a non-recursive SCC, once.
    fn plain_scc(
        &self,
        store: &mut Store,
        current: &HashMap<Label, u32>,
        scc: &Scc<'_>,
    ) -> Result<BTreeMap<Label, Rows>, Refusal> {
        let resolve = |l: Label, _: usize| lookup(current, l);
        let mut heads: BTreeMap<Label, Rows> = BTreeMap::new();
        for (head, body) in &scc.rules {
            let rows = self.rule(store, &resolve, body)?;
            heads.entry(*head).or_default().extend(rows);
        }
        Ok(heads)
    }

    /// Semi-naive evaluation of a recursive SCC (see the module docs).
    /// Terminates: every row is a pair over the map's ids, and a round that
    /// adds no row ends the loop.
    fn recursive_scc(
        &self,
        store: &mut Store,
        outer: &HashMap<Label, u32>,
        scc: &Scc<'_>,
        rounds: &mut usize,
    ) -> Result<BTreeMap<Label, Rows>, Refusal> {
        // Rules reading no SCC member seed the first delta; the others are
        // recursive, with the positions of their SCC atoms.
        let mut delta: BTreeMap<Label, Rows> = BTreeMap::new();
        let mut recursive: Vec<(Label, &RirNode, Vec<usize>)> = Vec::new();
        for (head, body) in &scc.rules {
            let atoms = atoms(body)?
                .into_iter()
                .map(|rel| self.label(rel))
                .collect::<Result<Vec<_>, _>>()?;
            let inside: Vec<usize> = (0..atoms.len())
                .filter(|&i| scc.members.contains(&atoms[i]))
                .collect();
            if inside.is_empty() {
                let resolve = |l: Label, _: usize| lookup(outer, l);
                let rows = self.rule(store, &resolve, body)?;
                delta.entry(*head).or_default().extend(rows);
            } else {
                recursive.push((*head, *body, inside));
            }
        }

        let mut full: BTreeMap<Label, Rows> =
            scc.members.iter().map(|m| (*m, Rows::new())).collect();
        let mut current = outer.clone();
        loop {
            // Keep only rows the full relation lacks; fold them in.
            let mut delta_ids: HashMap<Label, u32> = HashMap::new();
            let mut grew = false;
            for m in &scc.members {
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
                let add = |store: &mut Store, rows: &Rows| {
                    store
                        .add(rows)
                        .map_err(Refusal::Cube)?
                        .ok_or(Refusal::NotBinary(*m))
                };
                delta_ids.insert(*m, add(store, &new)?);
                current.insert(*m, add(store, &full[m])?);
            }
            if !grew {
                return Ok(full);
            }
            *rounds += 1;
            for (head, body, inside) in &recursive {
                for &at in inside {
                    let resolve = |l: Label, pos: usize| {
                        if pos == at {
                            lookup(&delta_ids, l)
                        } else {
                            lookup(&current, l)
                        }
                    };
                    let rows = self.rule(store, &resolve, body)?;
                    delta.entry(*head).or_default().extend(rows);
                }
            }
        }
    }

    fn rule(
        &self,
        store: &mut Store,
        resolve: &Resolve<'_>,
        body: &RirNode,
    ) -> Result<Vec<Vec<u32>>, Refusal> {
        // A body reading behaviour is refused as such, whatever its shape:
        // the reason must not depend on which shape check runs first.
        let mut word = None;
        body.visit_referenced_relations(&mut |rel| {
            word = word.or_else(|| self.words.get(&rel).copied());
        });
        if let Some(w) = word {
            return Err(Refusal::Behaviour(w));
        }
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
                        _ => Err(Refusal::OutputColumn),
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                Ok(self
                    .single(store, resolve, input)?
                    .into_iter()
                    .map(|row| cols.iter().map(|&i| row[i]).collect())
                    .collect())
            }
            RirNode::MultiWayJoin { fallback, .. } => self.multiway(store, resolve, fallback),
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
                other => Err(Refusal::ProjectOver(kind(other))),
            },
            other => Err(Refusal::Body(kind(other))),
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
    ) -> Result<Vec<Vec<u32>>, Refusal> {
        let (rel, bound) = match node {
            RirNode::Scan { rel } => (*rel, None),
            RirNode::Filter { input, predicate } => match input.as_ref() {
                RirNode::Scan { rel } => (*rel, Some(eq_const(predicate)?)),
                other => return Err(Refusal::FilterOver(kind(other))),
            },
            other => return Err(Refusal::Body(kind(other))),
        };
        if bound.is_some_and(|(col, _)| col > 1) {
            return Err(Refusal::FilterColumn);
        }
        let id = resolve(self.label(rel)?, 0)?;
        Ok(store.rows[&id]
            .iter()
            .filter(|row| bound.is_none_or(|(col, v)| row[col] == v))
            .cloned()
            .collect())
    }

    /// A multi-way conjunctive join, evaluated as generic join over the
    /// cube's coordinates. The atoms, the variables and the output come from
    /// `fallback`, the IR-equivalent binary-join plan: its scan leaves are the
    /// atoms, its join keys (unified) the variables, its projection the head.
    /// One variable is bound at a time; its candidates are the AND of one
    /// adjacency mask per atom it occurs in (`lance_graph_xlog::and`).
    fn multiway(
        &self,
        store: &Store,
        resolve: &Resolve<'_>,
        fallback: &RirNode,
    ) -> Result<Vec<Vec<u32>>, Refusal> {
        let (leaves, class, columns) = flatten_fallback(fallback)?;
        let map = store
            .table
            .map(&store.table.seal(), TILE)
            .map_err(|e| Refusal::Cube(CubeRefusal::Map(e)))?;
        let n = map.len();
        let words = n.div_ceil(64);
        // Per atom: its two variables and its adjacency masks in both
        // directions, over local positions.
        let mut atoms = Vec::with_capacity(leaves.len());
        for (i, rel) in leaves.iter().enumerate() {
            let id = resolve(self.label(*rel)?, i)?;
            let pairs = binary(&store.rows[&id]).ok_or(Refusal::NotBinary(self.label(*rel)?))?;
            let mut adj = Adjacency::new(n, words);
            for (a, b) in pairs {
                let local = |g: u32| {
                    map.local(u64::from(g))
                        .map(usize::from)
                        .map_err(|e| Refusal::Cube(CubeRefusal::Domain(e)))
                };
                adj.insert(local(a)?, local(b)?);
            }
            atoms.push(((class[2 * i], class[2 * i + 1]), adj));
        }
        let vars = class.iter().copied().max().map_or(0, |m| m + 1);
        let mut binding = vec![usize::MAX; vars];
        let mut out = BTreeSet::new();
        generic_join(&atoms, words, 0, &mut binding, &mut |b: &[usize]| {
            let row: Vec<u32> = columns
                .iter()
                .map(|&c| {
                    map.global(b[class[c]] as u8)
                        .expect("local position in the map") as u32
                })
                .collect();
            out.insert(row);
        });
        Ok(out.into_iter().collect())
    }

    fn side(
        &self,
        resolve: &Resolve<'_>,
        pos: usize,
        node: &RirNode,
        key: usize,
    ) -> Result<Side, Refusal> {
        match node {
            RirNode::Scan { rel } => Ok(Side {
                rel: resolve(self.label(*rel)?, pos)?,
                bound: None,
            }),
            RirNode::Filter { input, predicate } => {
                let RirNode::Scan { rel } = input.as_ref() else {
                    return Err(Refusal::FilterOver(kind(input)));
                };
                let (col, v) = eq_const(predicate)?;
                if col == key || col > 1 {
                    return Err(Refusal::BoundOnJoinColumn);
                }
                Ok(Side {
                    rel: resolve(self.label(*rel)?, pos)?,
                    bound: Some((col, v)),
                })
            }
            other => Err(Refusal::JoinSide(kind(other))),
        }
    }

    fn view(store: &mut Store, side: Side) -> Result<View, Refusal> {
        let scan = store.cube.scan(side.rel).map_err(Refusal::Cube)?;
        match side.bound {
            None => Ok(scan),
            Some((col, v)) => store
                .cube
                .bind(&store.table, scan, col as u8, v)
                .map_err(Refusal::Cube),
        }
    }

    fn join(
        &self,
        store: &mut Store,
        resolve: &Resolve<'_>,
        (left, lk): (&RirNode, usize),
        (right, rk): (&RirNode, usize),
        columns: &[ProjectExpr],
    ) -> Result<Vec<Vec<u32>>, Refusal> {
        if lk > 1 || rk > 1 {
            return Err(Refusal::JoinKey);
        }
        let cols = columns
            .iter()
            .map(|c| match c {
                ProjectExpr::Column(i) if *i < 4 => Ok(*i),
                _ => Err(Refusal::OutputColumn),
            })
            .collect::<Result<Vec<_>, _>>()?;
        let l = self.side(resolve, 0, left, lk)?;
        let r = self.side(resolve, 1, right, rk)?;
        let (lv, rv) = (Self::view(store, l)?, Self::view(store, r)?);
        let j = store
            .cube
            .join(lv, lk as u8, rv, rk as u8)
            .map_err(Refusal::Cube)?;
        let rows = store
            .cube
            .materialize_rows(&store.table, j)
            .map_err(Refusal::Cube)?;
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

/// The relations a body reads, by atom position.
fn atoms(body: &RirNode) -> Result<Vec<RelId>, Refusal> {
    fn atom(n: &RirNode) -> Result<RelId, Refusal> {
        match n {
            RirNode::Scan { rel } => Ok(*rel),
            RirNode::Filter { input, .. } => atom(input),
            other => Err(Refusal::JoinSide(kind(other))),
        }
    }
    match body {
        RirNode::Scan { .. } | RirNode::Filter { .. } => Ok(vec![atom(body)?]),
        RirNode::ChainJoin { left, right, .. } => Ok(vec![atom(left)?, atom(right)?]),
        RirNode::MultiWayJoin { fallback, .. } => Ok(flatten_fallback(fallback)?.0),
        RirNode::Project { input, .. } => match input.as_ref() {
            RirNode::Join { left, right, .. } => Ok(vec![atom(left)?, atom(right)?]),
            RirNode::Scan { .. } | RirNode::Filter { .. } => Ok(vec![atom(input)?]),
            other => Err(Refusal::ProjectOver(kind(other))),
        },
        other => Err(Refusal::Body(kind(other))),
    }
}

/// One atom's relation as bit masks over local positions: `fwd[x]` holds
/// the `y` with `(x, y)`, `bwd[y]` the `x`, and `col0`/`col1` the values each
/// column takes.
struct Adjacency {
    fwd: Vec<Vec<u64>>,
    bwd: Vec<Vec<u64>>,
    col0: Vec<u64>,
    col1: Vec<u64>,
}

impl Adjacency {
    fn new(n: usize, words: usize) -> Self {
        Self {
            fwd: vec![vec![0; words]; n],
            bwd: vec![vec![0; words]; n],
            col0: vec![0; words],
            col1: vec![0; words],
        }
    }

    fn insert(&mut self, a: usize, b: usize) {
        self.fwd[a][b / 64] |= 1 << (b % 64);
        self.bwd[b][a / 64] |= 1 << (a % 64);
        self.col0[a / 64] |= 1 << (a % 64);
        self.col1[b / 64] |= 1 << (b % 64);
    }
}

/// Bind variable `v`, then the rest. A variable's candidates are the AND,
/// over every atom it occurs in, of: the partner's row or column when the
/// partner is bound (or is `v` itself), else the column's value set.
fn generic_join(
    atoms: &[((usize, usize), Adjacency)],
    words: usize,
    v: usize,
    binding: &mut Vec<usize>,
    emit: &mut dyn FnMut(&[usize]),
) {
    if v == binding.len() {
        emit(binding);
        return;
    }
    let mut cand = vec![u64::MAX; words];
    let mut scratch = vec![0u64; words];
    let mut constrained = false;
    for ((a, b), adj) in atoms {
        let mask: Option<Vec<u64>> = match (*a == v, *b == v) {
            // `r(V, V)`: the diagonal.
            (true, true) => Some(
                (0..words)
                    .map(|w| {
                        (0..64)
                            .filter(|i| {
                                let x = w * 64 + i;
                                x < adj.fwd.len() && adj.fwd[x][w] >> i & 1 == 1
                            })
                            .fold(0u64, |m, i| m | 1 << i)
                    })
                    .collect(),
            ),
            (true, false) if binding[*b] != usize::MAX => Some(adj.bwd[binding[*b]].clone()),
            (true, false) => Some(adj.col0.clone()),
            (false, true) if binding[*a] != usize::MAX => Some(adj.fwd[binding[*a]].clone()),
            (false, true) => Some(adj.col1.clone()),
            _ => None,
        };
        if let Some(m) = mask {
            lance_graph_xlog::and(&cand, &m, &mut scratch);
            std::mem::swap(&mut cand, &mut scratch);
            constrained = true;
        }
    }
    if !constrained {
        return;
    }
    let n = atoms.first().map_or(0, |(_, adj)| adj.fwd.len());
    for x in 0..n {
        if cand[x / 64] >> (x % 64) & 1 == 1 {
            binding[v] = x;
            generic_join(atoms, words, v + 1, binding, emit);
        }
    }
    binding[v] = usize::MAX;
}

/// The scan leaves of a `MultiWayJoin` fallback (`Project` over a tree of
/// inner `Join`s), the variable class of every leaf column (join keys
/// unified, classes renumbered by first appearance), and the projected
/// leaf columns. Leaves are binary: two columns each.
fn flatten_fallback(fallback: &RirNode) -> Result<Flat, Refusal> {
    let RirNode::Project { input, columns } = fallback else {
        return Err(Refusal::Fallback(kind(fallback)));
    };
    fn walk(
        n: &RirNode,
        leaves: &mut Vec<RelId>,
        eqs: &mut Vec<(usize, usize)>,
    ) -> Result<usize, Refusal> {
        match n {
            RirNode::Scan { rel } => {
                leaves.push(*rel);
                Ok(2)
            }
            RirNode::Join {
                left,
                right,
                left_keys,
                right_keys,
                join_type: JoinType::Inner,
            } => {
                let off = 2 * leaves.len();
                let wl = walk(left, leaves, eqs)?;
                let wr = walk(right, leaves, eqs)?;
                for (&l, &r) in left_keys.iter().zip(right_keys) {
                    if l >= wl || r >= wr {
                        return Err(Refusal::JoinKey);
                    }
                    eqs.push((off + l, off + wl + r));
                }
                Ok(wl + wr)
            }
            other => Err(Refusal::Fallback(kind(other))),
        }
    }
    let mut leaves = Vec::new();
    let mut eqs = Vec::new();
    let width = walk(input, &mut leaves, &mut eqs)?;
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
    let mut ids: HashMap<usize, usize> = HashMap::new();
    let class: Vec<usize> = (0..width)
        .map(|c| {
            let r = root(&mut parent, c);
            let next = ids.len();
            *ids.entry(r).or_insert(next)
        })
        .collect();
    let columns = columns
        .iter()
        .map(|c| match c {
            ProjectExpr::Column(i) if *i < width => Ok(*i),
            _ => Err(Refusal::OutputColumn),
        })
        .collect::<Result<Vec<_>, _>>()?;
    Ok((leaves, class, columns))
}

/// `col == const`, the only filter admitted.
fn eq_const(predicate: &Expr) -> Result<(usize, u32), Refusal> {
    let Expr::Compare {
        left,
        op: CompareOp::Eq,
        right,
    } = predicate
    else {
        return Err(Refusal::Filter);
    };
    // Without facts to type the column, XLOG types the constant `U64`.
    match (left.as_ref(), right.as_ref()) {
        (Expr::Column(c), Expr::Const(ConstValue::U32(v))) => Ok((*c, *v)),
        (Expr::Column(c), Expr::Const(ConstValue::U64(v))) => {
            Ok((*c, u32::try_from(*v).map_err(|_| Refusal::Constant)?))
        }
        _ => Err(Refusal::Filter),
    }
}

fn lookup(ids: &HashMap<Label, u32>, l: Label) -> Result<u32, Refusal> {
    ids.get(&l).copied().ok_or(Refusal::NotBinary(l))
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
