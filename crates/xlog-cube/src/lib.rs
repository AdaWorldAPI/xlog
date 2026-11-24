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
//! One-column relations given with [`CubeExecutor::put_set`] are cube sets.
//! Two SCC shapes over them are routed to cube set expressions before the
//! paths above (the `sets` module; [`Report::routed`] names the path): a
//! non-recursive SCC of one-column set bodies, and a recursive SCC with one
//! one-column member whose single step is `h(Y) :- h(X), e(X, Y), f(Y)`.
//!
//! Every other body is refused and recorded in [`Report::refused`] with its
//! reason. A refused rule refuses its whole SCC: a head is never handed back
//! with only some of its rules applied.

use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::fmt;

use lance_graph_contract::codebook::Codebook;
use lance_graph_dt64::dr::{LocalMap, MapTable, Width};
use lance_graph_dt64::AlgebraRef;
use lance_graph_xlog::cube::{Cube, CubeRefusal, CubeStats, SetExpr, View};
use lance_graph_xlog::lens::word_of;
use xlog_core::RelId;
use xlog_ir::{
    CompareOp, CompiledRule, ConstValue, ExecutionPlan, Expr, JoinType, ProjectExpr, RirNode,
};

mod sets;

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
    /// Heads evaluated as set expressions on the cube instead of rows:
    /// `(head, "set" | "reach")` (D-XLOG-W5-C). Every routed head is also in
    /// `ran`.
    pub routed: Vec<(String, String)>,
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
    /// The rule reads a relation whose own rules were refused: it would read
    /// an empty stand-in, not that relation.
    ReadsRefused(Label),
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
            Refusal::ReadsRefused(l) => write!(f, "reads refused relation (label {l})"),
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
    /// One-column relations given ids ([`Self::put_set`]).
    sets: BTreeMap<String, Vec<u32>>,
    /// Registered one-column relations nobody supplied: empty, at intake.
    empty_sets: BTreeSet<String>,
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
    /// Derived heads whose arity is not 2, by label. The cube holds binary
    /// relations only, so these are never cube relations: a single-relation
    /// projection may read them, and every other body still refuses them.
    wide: HashMap<Label, Rows>,
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

    /// A one-column `u32` relation, given as its ids. It is stored in the
    /// cube as a set, and the rules reading it are routed as set expressions
    /// when their shape allows (see the `sets` module).
    pub fn put_set(&mut self, name: &str, ids: &[u32]) {
        self.sets.insert(name.to_string(), ids.to_vec());
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
        if let Some(ids) = self.sets.get(name) {
            let rows: BTreeSet<Vec<u32>> = ids.iter().map(|&i| vec![i]).collect();
            return Some(rows.into_iter().collect());
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
            .chain(self.sets.values().flatten().map(|&i| u64::from(i)))
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
            wide: HashMap::new(),
        };
        // One-column relations: stored sets, and the expressions of the
        // heads routed so far.
        let mut exprs: HashMap<Label, SetExpr> = HashMap::new();
        let no_ids: Vec<u32> = Vec::new();
        for (name, ids) in self
            .sets
            .iter()
            .chain(self.empty_sets.iter().map(|n| (n, &no_ids)))
        {
            if let Some(l) = self.vocab.index_of(name) {
                let id = store
                    .cube
                    .add_set(&store.table, ids)
                    .map_err(PlanRefusal::Cube)?;
                exprs.insert(l, SetExpr::Set(id));
            }
        }
        let arity = |rel: RelId| plan.rel_arities.get(&rel).copied();
        let mut routed: Vec<(Label, sets::Route)> = Vec::new();
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
        // Heads whose SCC was refused. Every arity-2 relation has an empty
        // stand-in in the cube, so a reader of a refused head would run over
        // that stand-in and report a wrong answer as ran: refuse it instead.
        let mut refused_heads: BTreeSet<Label> = BTreeSet::new();
        for scc in &sccs {
            // Rows a host put for a member of this SCC (facts of a derived
            // predicate) are part of its answer.
            let put: BTreeMap<Label, Rows> = scc
                .members
                .iter()
                .filter_map(|m| {
                    let rows = self.base.get(self.vocab.label(*m)?)?;
                    Some((*m, pairs_to_rows(rows)))
                })
                .collect();
            let outcome = if let Some(l) = self.reads_any(scc, &refused_heads) {
                Err(Refusal::ReadsRefused(l))
            } else if let Some((route, heads)) = self.route(scc, &exprs, &current, &arity) {
                // Download each head's answer; keep its expression, not the
                // mask, for later readers.
                let mut out = BTreeMap::new();
                let mut failed = None;
                for (h, e) in heads {
                    match store.cube.set_mask(&store.table, &e) {
                        Ok(mask) => {
                            out.insert(h, mask_rows(&store, &mask));
                            exprs.insert(h, e);
                            routed.push((h, route));
                        }
                        Err(e) => failed = Some(Refusal::Cube(e)),
                    }
                }
                match failed {
                    Some(r) => Err(r),
                    None => Ok(out),
                }
            } else if scc.recursive {
                // Not routed: a member's stored facts are not its whole
                // answer, so no later rule may read them as the head.
                for m in &scc.members {
                    exprs.remove(m);
                }
                self.recursive_scc(&mut store, &current, scc, &put, &mut rounds)
            } else {
                for m in &scc.members {
                    exprs.remove(m);
                }
                self.plain_scc(&mut store, &current, scc).map(|mut heads| {
                    for (m, rows) in &put {
                        heads.entry(*m).or_default().extend(rows.iter().cloned());
                    }
                    heads
                })
            };
            match outcome {
                Ok(heads) => {
                    ran.extend(scc.rules.iter().map(|(h, _)| *h));
                    for (head, rows) in heads {
                        if let Some(id) = store.add(&rows).map_err(PlanRefusal::Cube)? {
                            current.insert(head, id);
                        } else {
                            store.wide.insert(head, rows.clone());
                        }
                        produced.insert(head, rows);
                    }
                }
                Err(reason) => {
                    refused_heads.extend(scc.rules.iter().map(|(h, _)| *h));
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
            routed: routed
                .iter()
                .map(|(h, r)| (name(h), r.name().to_string()))
                .collect(),
        })
    }

    /// A refusal's text, with labels written as names.
    fn describe(&self, r: &Refusal) -> String {
        match r {
            Refusal::NotBinary(l) => match self.vocab.label(*l) {
                Some(n) => format!("`{n}` is not a binary relation in the cube"),
                None => r.to_string(),
            },
            Refusal::ReadsRefused(l) => match self.vocab.label(*l) {
                Some(n) => format!("reads refused relation `{n}`"),
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
            .chain(self.sets.keys())
            .chain(registered.iter().map(|(_, n)| *n))
            .chain(heads);
        for name in names {
            vocab.intern(name).ok_or(PlanRefusal::Vocabulary)?;
        }
        // A derived head never gets an empty stand-in: a reader would read
        // it instead of the head's rows.
        let derived: BTreeSet<&String> = plan
            .rules_by_scc
            .iter()
            .flatten()
            .map(|r| &r.head)
            .collect();
        self.empty_sets = registered
            .iter()
            .filter(|(rel, name)| {
                plan.rel_arities.get(*rel) == Some(&1)
                    && !derived.contains(name)
                    && !self.sets.contains_key(*name)
                    && !self.base.contains_key(*name)
            })
            .map(|(_, name)| (*name).clone())
            .collect();
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

    /// The first relation in `labels` that a rule of `scc` reads.
    fn reads_any(&self, scc: &Scc<'_>, labels: &BTreeSet<Label>) -> Option<Label> {
        let mut hit = None;
        for (_, body) in &scc.rules {
            body.visit_referenced_relations(&mut |rel| {
                if hit.is_none() {
                    hit = self
                        .labels
                        .get(&rel)
                        .copied()
                        .filter(|l| labels.contains(l));
                }
            });
        }
        hit
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
        put: &BTreeMap<Label, Rows>,
        rounds: &mut usize,
    ) -> Result<BTreeMap<Label, Rows>, Refusal> {
        // Rows put for a member, and rules reading no SCC member, seed the
        // first delta; the others are recursive, with the positions of their
        // SCC atoms.
        let mut delta: BTreeMap<Label, Rows> = put.clone();
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
            // `X != Z` (or `X = Z`) over a join: filter the join's rows, then
            // project. Only a column-against-column equality or inequality;
            // an ordering comparison is refused below.
            RirNode::Project { input, columns } if join_compare(input).is_some() => {
                let Some((join, (a, eq, b))) = join_compare(input) else {
                    unreachable!("guard checked join_compare");
                };
                let RirNode::Join {
                    left,
                    right,
                    left_keys,
                    right_keys,
                    ..
                } = join
                else {
                    unreachable!("join_compare returns a join");
                };
                let all: Vec<ProjectExpr> = (0..4).map(ProjectExpr::Column).collect();
                let joined = self.join(
                    store,
                    resolve,
                    (left, left_keys[0]),
                    (right, right_keys[0]),
                    &all,
                )?;
                let cols = columns
                    .iter()
                    .map(|c| match c {
                        ProjectExpr::Column(i) if *i < 4 => Ok(*i),
                        _ => Err(Refusal::OutputColumn),
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                Ok(joined
                    .into_iter()
                    .filter(|row| (row[a] == row[b]) == eq)
                    .map(|row| cols.iter().map(|&i| row[i]).collect())
                    .collect())
            }
            RirNode::Project { input, columns }
                if matches!(
                    input.as_ref(),
                    RirNode::Scan { .. } | RirNode::Filter { .. }
                ) =>
            {
                let width = self.width(store, input)?;
                let cols = columns
                    .iter()
                    .map(|c| match c {
                        ProjectExpr::Column(i) if *i < width => Ok(*i),
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
        let (rel, keep) = match node {
            RirNode::Scan { rel } => (*rel, Keep::All),
            RirNode::Filter { input, predicate } => match input.as_ref() {
                // `r(V,V)`: a repeated variable compiles to `col0 == col1`.
                RirNode::Scan { rel } => match eq_cols(predicate) {
                    Some((a, b)) => (*rel, Keep::Diagonal(a, b)),
                    None => {
                        let (col, v) = eq_const(predicate)?;
                        (*rel, Keep::Const(col, v))
                    }
                },
                other => return Err(Refusal::FilterOver(kind(other))),
            },
            other => return Err(Refusal::Body(kind(other))),
        };
        if !keep.in_binary() {
            return Err(Refusal::FilterColumn);
        }
        let l = self.label(rel)?;
        let rows = match resolve(l, 0) {
            Ok(id) => &store.rows[&id],
            // A wide derived head is not a cube relation; its rows are.
            Err(Refusal::NotBinary(_)) if store.wide.contains_key(&l) => &store.wide[&l],
            Err(e) => return Err(e),
        };
        Ok(rows.iter().filter(|row| keep.holds(row)).cloned().collect())
    }

    /// The column count of a single-relation body: the arity of a wide
    /// derived head, else 2.
    fn width(&self, store: &Store, node: &RirNode) -> Result<usize, Refusal> {
        let rel = match node {
            RirNode::Scan { rel } => *rel,
            RirNode::Filter { input, .. } => match input.as_ref() {
                RirNode::Scan { rel } => *rel,
                other => return Err(Refusal::FilterOver(kind(other))),
            },
            other => return Err(Refusal::Body(kind(other))),
        };
        let l = self.label(rel)?;
        Ok(store
            .wide
            .get(&l)
            .and_then(|rows| rows.iter().next())
            .map_or(2, Vec::len))
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
/// What a single-relation body keeps.
#[derive(Clone, Copy)]
enum Keep {
    All,
    /// `col == const`.
    Const(usize, u32),
    /// `col == col`.
    Diagonal(usize, usize),
}

impl Keep {
    fn in_binary(self) -> bool {
        match self {
            Keep::All => true,
            Keep::Const(c, _) => c < 2,
            Keep::Diagonal(a, b) => a < 2 && b < 2,
        }
    }

    fn holds(self, row: &[u32]) -> bool {
        match self {
            Keep::All => true,
            Keep::Const(c, v) => row[c] == v,
            Keep::Diagonal(a, b) => row[a] == row[b],
        }
    }
}

/// `Column(a) == Column(b)`, the filter a repeated variable compiles to.
/// Both columns hold ids of the same relation, so equality needs no typing.
/// `Filter{Join, column op column}` with one inner join key per side and
/// `op` equality or inequality: the join and `(a, eq, b)`, where `eq` is true
/// for `=` and false for `!=`.
fn join_compare(node: &RirNode) -> Option<(&RirNode, (usize, bool, usize))> {
    let RirNode::Filter { input, predicate } = node else {
        return None;
    };
    let RirNode::Join {
        left_keys,
        right_keys,
        join_type: JoinType::Inner,
        ..
    } = input.as_ref()
    else {
        return None;
    };
    if left_keys.len() != 1 || right_keys.len() != 1 {
        return None;
    }
    let Expr::Compare { left, op, right } = predicate else {
        return None;
    };
    let eq = match op {
        CompareOp::Eq => true,
        CompareOp::Ne => false,
        _ => return None,
    };
    match (left.as_ref(), right.as_ref()) {
        (Expr::Column(a), Expr::Column(b)) if *a < 4 && *b < 4 => Some((input, (*a, eq, *b))),
        _ => None,
    }
}

fn eq_cols(predicate: &Expr) -> Option<(usize, usize)> {
    match predicate {
        Expr::Compare {
            left,
            op: CompareOp::Eq,
            right,
        } => match (left.as_ref(), right.as_ref()) {
            (Expr::Column(a), Expr::Column(b)) => Some((*a, *b)),
            _ => None,
        },
        _ => None,
    }
}

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

/// A set head's rows: one `[id]` per position of `mask`.
fn mask_rows(store: &Store, mask: &[u64]) -> Rows {
    let Ok(map) = store.table.map(&store.table.seal(), TILE) else {
        return Rows::new();
    };
    map.globals()
        .iter()
        .enumerate()
        .filter(|(p, _)| mask.get(p / 64).is_some_and(|w| w >> (p % 64) & 1 == 1))
        .map(|(_, &g)| vec![g as u32])
        .collect()
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
