//! Stratification analysis for negation and aggregation

use crate::ast::{BodyLiteral, Program};
use std::collections::{HashMap, HashSet};
use xlog_core::{Result, XlogError};

struct StratificationAllocations<'a, 'b> {
    scope: Option<&'a mut crate::compile::CompilationAllocationScope<'b>>,
}
impl StratificationAllocations<'_, '_> {
    fn acquire(&mut self, bytes: u64) -> Result<()> {
        if bytes != 0 {
            if let Some(scope) = self.scope.as_deref_mut() {
                scope.acquire(bytes)?;
            }
        }
        Ok(())
    }
    fn vector<T>(&mut self, count: usize) -> Result<()> {
        if self.scope.is_some() {
            self.acquire(crate::compile::push_vector_allocation_bytes::<T>(count)?)?;
        }
        Ok(())
    }
    fn hash<K, V>(&mut self, count: usize) -> Result<()> {
        if self.scope.is_some() {
            self.acquire(
                crate::compile::hash_allocation_bytes::<K, V>(count)?
                    .checked_mul(2)
                    .ok_or_else(|| XlogError::allocation_extent_overflow("stratification table"))?,
            )?;
        }
        Ok(())
    }
}

/// Dependency edge type
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum DepType {
    Positive,
    Negative,
    Aggregate,
}

/// Dependency graph edge
#[derive(Debug, Clone)]
pub(crate) struct DepEdge {
    pub from: String,
    pub to: String,
    pub dep_type: DepType,
}

/// Dependency graph for stratification analysis.
#[derive(Debug, Default)]
pub struct DependencyGraph {
    /// Set of all predicate names in the graph.
    pub predicates: HashSet<String>,
    /// Private so every insertion goes through `add_edge`, which keeps
    /// `adjacency` in sync — a bare `edges.push` would silently drop the edge
    /// from `outgoing()`.
    edges: Vec<DepEdge>,
    /// Outgoing-edge indices per predicate, kept in insertion order so
    /// `outgoing()` returns edges exactly as the linear scan over `edges` did.
    adjacency: std::collections::HashMap<String, Vec<usize>>,
}

impl DependencyGraph {
    /// Create an empty dependency graph.
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a predicate node to the graph.
    pub fn add_predicate(&mut self, name: String) {
        self.predicates.insert(name);
    }

    pub(crate) fn add_edge(&mut self, from: String, to: String, dep_type: DepType) {
        self.predicates.insert(from.clone());
        self.predicates.insert(to.clone());
        self.adjacency
            .entry(from.clone())
            .or_default()
            .push(self.edges.len());
        self.edges.push(DepEdge { from, to, dep_type });
    }

    pub(crate) fn outgoing(&self, pred: &str) -> Vec<&DepEdge> {
        match self.adjacency.get(pred) {
            Some(idxs) => idxs
                .iter()
                .map(|&i| {
                    let edge = &self.edges[i];
                    debug_assert_eq!(edge.from, pred, "adjacency index out of sync");
                    edge
                })
                .collect(),
            None => Vec::new(),
        }
    }

    pub(crate) fn outgoing_len(&self, pred: &str) -> usize {
        self.adjacency.get(pred).map(Vec::len).unwrap_or(0)
    }
}

/// Build dependency graph from program
pub fn build_dependency_graph(program: &Program) -> DependencyGraph {
    build_dependency_graph_with_owner(program, &mut StratificationAllocations { scope: None })
        .expect("ordinary dependency construction has no allocation callback")
}

pub(crate) fn build_dependency_graph_with_allocation_owner(
    program: &Program,
    scope: &mut crate::compile::CompilationAllocationScope<'_>,
) -> Result<DependencyGraph> {
    build_dependency_graph_with_owner(
        program,
        &mut StratificationAllocations { scope: Some(scope) },
    )
}

fn build_dependency_graph_with_owner(
    program: &Program,
    allocations: &mut StratificationAllocations<'_, '_>,
) -> Result<DependencyGraph> {
    let mut graph = DependencyGraph::new();

    for rule in &program.rules {
        let head = &rule.head.predicate;
        allocations.acquire(head.len() as u64)?;
        if !graph.predicates.contains(head) && graph.predicates.len() == graph.predicates.capacity()
        {
            allocations.hash::<String, ()>(graph.predicates.len() + 1)?;
        }
        graph.add_predicate(head.clone());

        for lit in &rule.body {
            match lit {
                BodyLiteral::Positive(atom) => {
                    acquire_edge(&graph, head, &atom.predicate, allocations)?;
                    graph.add_edge(head.clone(), atom.predicate.clone(), DepType::Positive);
                }
                BodyLiteral::Negated(atom) => {
                    acquire_edge(&graph, head, &atom.predicate, allocations)?;
                    graph.add_edge(head.clone(), atom.predicate.clone(), DepType::Negative);
                }
                BodyLiteral::Epistemic(lit) => {
                    acquire_edge(&graph, head, &lit.atom.predicate, allocations)?;
                    graph.add_edge(head.clone(), lit.atom.predicate.clone(), DepType::Negative);
                }
                BodyLiteral::Comparison(_) | BodyLiteral::IsExpr(_) | BodyLiteral::Univ(_) => {}
            }
        }

        if rule.has_aggregation() {
            for lit in &rule.body {
                if let BodyLiteral::Positive(atom) = lit {
                    acquire_edge(&graph, head, &atom.predicate, allocations)?;
                    graph.add_edge(head.clone(), atom.predicate.clone(), DepType::Aggregate);
                }
            }
        }
    }

    // Learnable rules: head depends on body predicates.
    // At runtime, TensorMaskedJoin dynamically selects which relations
    // to join, but for stratification we conservatively register the
    // template's body predicates as positive dependencies.
    for lr in &program.learnable_rules {
        let head = &lr.head.predicate;
        allocations.acquire(head.len() as u64)?;
        if !graph.predicates.contains(head) && graph.predicates.len() == graph.predicates.capacity()
        {
            allocations.hash::<String, ()>(graph.predicates.len() + 1)?;
        }
        graph.add_predicate(head.clone());
        for body_lit in &lr.body {
            if let Some(atom) = body_lit.atom() {
                allocations.acquire(atom.predicate.len() as u64)?;
                if !graph.predicates.contains(&atom.predicate)
                    && graph.predicates.len() == graph.predicates.capacity()
                {
                    allocations.hash::<String, ()>(graph.predicates.len() + 1)?;
                }
                graph.add_predicate(atom.predicate.clone());
                acquire_edge(&graph, head, &atom.predicate, allocations)?;
                graph.add_edge(head.clone(), atom.predicate.clone(), DepType::Positive);
            }
        }
    }

    Ok(graph)
}

fn acquire_edge(
    graph: &DependencyGraph,
    from: &str,
    to: &str,
    allocations: &mut StratificationAllocations<'_, '_>,
) -> Result<()> {
    if allocations.scope.is_some() {
        let text = (from.len() as u64)
            .checked_mul(3)
            .and_then(|n| n.checked_add((to.len() as u64).checked_mul(2)?))
            .ok_or_else(|| XlogError::allocation_extent_overflow("dependency text"))?;
        allocations.acquire(text)?;
        if graph.predicates.capacity() - graph.predicates.len() < 2 {
            allocations.hash::<String, ()>(graph.predicates.len() + 2)?;
        }
        if graph.edges.len() == graph.edges.capacity() {
            allocations.vector::<DepEdge>(graph.edges.len() + 1)?;
        }
        if !graph.adjacency.contains_key(from)
            && graph.adjacency.len() == graph.adjacency.capacity()
        {
            allocations.hash::<String, Vec<usize>>(graph.adjacency.len() + 1)?;
        }
        let (length, capacity) = graph
            .adjacency
            .get(from)
            .map(|v| (v.len(), v.capacity()))
            .unwrap_or((0, 0));
        if length == capacity {
            allocations.vector::<usize>(length + 1)?;
        }
    }
    Ok(())
}

/// Find strongly connected components using Tarjan's algorithm
/// Returns SCCs in reverse topological order (dependencies first)
fn find_sccs(graph: &DependencyGraph) -> Vec<Vec<String>> {
    find_sccs_with_owner(graph, &mut StratificationAllocations { scope: None })
        .expect("ordinary SCC analysis has no allocation callback")
}

fn find_sccs_with_owner(
    graph: &DependencyGraph,
    allocations: &mut StratificationAllocations<'_, '_>,
) -> Result<Vec<Vec<String>>> {
    if allocations.scope.is_some() {
        let count = graph.predicates.len();
        let mut text = 0u64;
        for name in &graph.predicates {
            text = text
                .checked_add(
                    (name.len() as u64)
                        .checked_mul(4)
                        .ok_or_else(|| XlogError::allocation_extent_overflow("SCC text"))?,
                )
                .ok_or_else(|| XlogError::allocation_extent_overflow("SCC text"))?;
        }
        for edge in &graph.edges {
            text = text
                .checked_add(edge.to.len() as u64)
                .and_then(|n| n.checked_add(edge.from.len() as u64))
                .ok_or_else(|| XlogError::allocation_extent_overflow("SCC text"))?;
        }
        allocations.acquire(text)?;
        allocations.hash::<String, usize>(count)?;
        allocations.hash::<String, usize>(count)?;
        allocations.hash::<String, ()>(count)?;
        allocations.vector::<String>(count)?;
        allocations.vector::<String>(count)?;
        allocations.vector::<Vec<String>>(count)?;
        allocations.vector::<&String>(count)?;
        allocations.vector::<String>(graph.edges.len())?;
        allocations.vector::<&DepEdge>(graph.edges.len())?;
    }
    struct TarjanState<'a> {
        graph: &'a DependencyGraph,
        next_index: usize,
        stack: Vec<String>,
        indices: HashMap<String, usize>,
        lowlinks: HashMap<String, usize>,
        on_stack: HashSet<String>,
        components: Vec<Vec<String>>,
    }

    impl TarjanState<'_> {
        fn strongconnect(&mut self, vertex: &str) {
            self.indices.insert(vertex.to_string(), self.next_index);
            self.lowlinks.insert(vertex.to_string(), self.next_index);
            self.next_index += 1;
            self.stack.push(vertex.to_string());
            self.on_stack.insert(vertex.to_string());

            let successors: Vec<String> = self
                .graph
                .outgoing(vertex)
                .iter()
                .map(|edge| edge.to.clone())
                .collect();
            for successor in successors {
                if !self.indices.contains_key(&successor) {
                    self.strongconnect(&successor);
                    let low_vertex = self.lowlinks[vertex];
                    let low_successor = self.lowlinks[&successor];
                    self.lowlinks
                        .insert(vertex.to_string(), low_vertex.min(low_successor));
                } else if self.on_stack.contains(&successor) {
                    let low_vertex = self.lowlinks[vertex];
                    let successor_index = self.indices[&successor];
                    self.lowlinks
                        .insert(vertex.to_string(), low_vertex.min(successor_index));
                }
            }

            if self.lowlinks[vertex] == self.indices[vertex] {
                let mut component = Vec::new();
                loop {
                    let member = self
                        .stack
                        .pop()
                        .expect("the current Tarjan root remains on the traversal stack");
                    self.on_stack.remove(&member);
                    let is_root = member == vertex;
                    component.push(member);
                    if is_root {
                        break;
                    }
                }
                self.components.push(component);
            }
        }
    }

    let mut state = TarjanState {
        graph,
        next_index: 0,
        stack: Vec::new(),
        indices: HashMap::new(),
        lowlinks: HashMap::new(),
        on_stack: HashSet::new(),
        components: Vec::new(),
    };

    // Deterministic visit order: `predicates` is a HashSet, whose iteration
    // order varies per process and used to make SCC ids (and every downstream
    // RelId assignment) process-nondeterministic.
    let mut preds_sorted: Vec<&String> = graph.predicates.iter().collect();
    preds_sorted.sort();
    for pred in preds_sorted {
        if !state.indices.contains_key(pred) {
            state.strongconnect(pred);
        }
    }

    Ok(state.components)
}

/// Check for cycles through negation/aggregation in an SCC
fn check_scc_for_negation_cycle(scc: &[String], graph: &DependencyGraph) -> Option<Vec<String>> {
    if scc.len() == 1 {
        let pred = &scc[0];
        for edge in graph.outgoing(pred) {
            if edge.to == *pred && edge.dep_type != DepType::Positive {
                return Some(vec![pred.clone()]);
            }
        }
        return None;
    }

    let scc_set: HashSet<&str> = scc.iter().map(|s| s.as_str()).collect();
    for pred in scc {
        for edge in graph.outgoing(pred) {
            if scc_set.contains(edge.to.as_str()) && edge.dep_type != DepType::Positive {
                return Some(scc.to_vec());
            }
        }
    }
    None
}

/// Stratum assignment result.
#[derive(Debug, Clone)]
pub struct Stratum {
    /// Stratum index (0 = base stratum).
    pub id: usize,
    /// Predicates assigned to this stratum.
    pub predicates: Vec<String>,
}

/// Result of stratification analysis for probabilistic inference
#[derive(Debug, Clone)]
pub struct StratificationResult {
    /// SCCs in evaluation order (dependencies first)
    pub sccs: Vec<Vec<String>>,
    /// Indices of SCCs that have cycles through negation (non-monotone)
    pub non_monotone_sccs: HashSet<usize>,
    /// Stratum number for each predicate (if fully stratified)
    pub strata: HashMap<String, usize>,
}

/// Perform stratification analysis
pub fn stratify(program: &Program) -> Result<Vec<Stratum>> {
    stratify_with_owner(program, &mut StratificationAllocations { scope: None })
}

/// Stratify through the same original dependency and SCC cores with custody.
pub fn stratify_with_allocation_owner(
    program: &Program,
    scope: &mut crate::compile::CompilationAllocationScope<'_>,
) -> Result<Vec<Stratum>> {
    stratify_with_owner(
        program,
        &mut StratificationAllocations { scope: Some(scope) },
    )
}

fn stratify_with_owner(
    program: &Program,
    allocations: &mut StratificationAllocations<'_, '_>,
) -> Result<Vec<Stratum>> {
    let graph = build_dependency_graph_with_owner(program, allocations)?;
    let sccs = find_sccs_with_owner(&graph, allocations)?;

    acquire_scc_observations(&sccs, &graph, allocations)?;

    for scc in &sccs {
        if let Some(cycle) = check_scc_for_negation_cycle(scc, &graph) {
            // Non-monotone programs are now supported for exact_ddnnf via WFS
            // Only reject for non-probabilistic programs that require stratification
            if !program.is_probabilistic_profile() {
                return Err(XlogError::StratificationCycle(cycle));
            }
            // Probabilistic programs (both exact_ddnnf and mc) can handle non-monotone cycles:
            // - exact_ddnnf uses WFS to compute well-founded model
            // - mc uses sampling which naturally handles non-stratified programs
        }
    }

    acquire_stratum_map(&sccs, allocations)?;
    let mut stratum_map: HashMap<String, usize> = HashMap::new();
    let mut max_stratum = 0;

    // Tarjan produces SCCs in reverse topological order.
    // Since edges go from "dependent" to "dependency" (head -> body predicate),
    // SCCs of dependencies come first. Process in order to ensure dependencies
    // are assigned strata before dependents.
    for scc in &sccs {
        let mut min_stratum = 0;
        for pred in scc {
            for edge in graph.outgoing(pred) {
                if let Some(&dep_stratum) = stratum_map.get(&edge.to) {
                    let required = match edge.dep_type {
                        DepType::Positive => dep_stratum,
                        DepType::Negative | DepType::Aggregate => dep_stratum + 1,
                    };
                    min_stratum = min_stratum.max(required);
                }
            }
        }
        for pred in scc {
            stratum_map.insert(pred.clone(), min_stratum);
        }
        max_stratum = max_stratum.max(min_stratum);
    }

    allocations.vector::<Stratum>(
        max_stratum
            .checked_add(1)
            .ok_or_else(|| XlogError::allocation_extent_overflow("stratum"))?,
    )?;
    allocations.vector::<String>(graph.predicates.len())?;
    let mut strata: Vec<Stratum> = (0..=max_stratum)
        .map(|id| Stratum {
            id,
            predicates: vec![],
        })
        .collect();

    for (pred, stratum) in stratum_map {
        strata[stratum].predicates.push(pred);
    }

    // `stratum_map` is a HashMap, so the drain order above varies per process;
    // sort so `Stratum::predicates` is deterministic output.
    for stratum in &mut strata {
        stratum.predicates.sort();
    }

    strata.retain(|s| !s.predicates.is_empty());
    for (i, stratum) in strata.iter_mut().enumerate() {
        stratum.id = i;
    }

    Ok(strata)
}

/// Analyze stratification for probabilistic inference
/// Returns detailed information about SCCs and which ones are non-monotone
pub fn analyze_stratification(program: &Program) -> StratificationResult {
    analyze_stratification_with_owner(program, &mut StratificationAllocations { scope: None })
        .expect("ordinary stratification analysis has no allocation callback")
}

/// Analyze the same original non-monotone SCCs with allocation custody.
pub fn analyze_stratification_with_allocation_owner(
    program: &Program,
    scope: &mut crate::compile::CompilationAllocationScope<'_>,
) -> Result<StratificationResult> {
    analyze_stratification_with_owner(
        program,
        &mut StratificationAllocations { scope: Some(scope) },
    )
}

fn analyze_stratification_with_owner(
    program: &Program,
    allocations: &mut StratificationAllocations<'_, '_>,
) -> Result<StratificationResult> {
    let graph = build_dependency_graph_with_owner(program, allocations)?;
    let sccs = find_sccs_with_owner(&graph, allocations)?;
    acquire_scc_observations(&sccs, &graph, allocations)?;
    allocations.hash::<usize, ()>(sccs.len())?;

    let mut non_monotone_sccs: HashSet<usize> = HashSet::new();
    for (i, scc) in sccs.iter().enumerate() {
        if check_scc_for_negation_cycle(scc, &graph).is_some() {
            non_monotone_sccs.insert(i);
        }
    }

    // Compute strata for predicates in stratified SCCs
    acquire_stratum_map(&sccs, allocations)?;
    let mut strata: HashMap<String, usize> = HashMap::new();
    let mut max_stratum = 0;

    for (scc_idx, scc) in sccs.iter().enumerate() {
        if non_monotone_sccs.contains(&scc_idx) {
            continue; // Skip non-monotone SCCs for stratum assignment
        }

        let mut min_stratum = 0;
        for pred in scc {
            for edge in graph.outgoing(pred) {
                if let Some(&dep_stratum) = strata.get(&edge.to) {
                    let required = match edge.dep_type {
                        DepType::Positive => dep_stratum,
                        DepType::Negative | DepType::Aggregate => dep_stratum + 1,
                    };
                    min_stratum = min_stratum.max(required);
                }
            }
        }
        for pred in scc {
            strata.insert(pred.clone(), min_stratum);
        }
        max_stratum = max_stratum.max(min_stratum);
    }

    Ok(StratificationResult {
        sccs,
        non_monotone_sccs,
        strata,
    })
}

fn acquire_scc_observations(
    sccs: &[Vec<String>],
    graph: &DependencyGraph,
    allocations: &mut StratificationAllocations<'_, '_>,
) -> Result<()> {
    if allocations.scope.is_some() {
        let count = sccs.iter().map(Vec::len).max().unwrap_or(0);
        allocations.hash::<&str, ()>(count)?;
        allocations.vector::<&DepEdge>(graph.edges.len())?;
        allocations.vector::<String>(count)?;
        let mut text = 0u64;
        for scc in sccs {
            for name in scc {
                text = text
                    .checked_add(name.len() as u64)
                    .ok_or_else(|| XlogError::allocation_extent_overflow("SCC copy"))?;
            }
        }
        allocations.acquire(text)?;
    }
    Ok(())
}

fn acquire_stratum_map(
    sccs: &[Vec<String>],
    allocations: &mut StratificationAllocations<'_, '_>,
) -> Result<()> {
    if allocations.scope.is_some() {
        let mut count = 0usize;
        let mut text = 0u64;
        for scc in sccs {
            count = count
                .checked_add(scc.len())
                .ok_or_else(|| XlogError::allocation_extent_overflow("stratum"))?;
            for name in scc {
                text = text
                    .checked_add(name.len() as u64)
                    .ok_or_else(|| XlogError::allocation_extent_overflow("stratum copy"))?;
            }
        }
        allocations.hash::<String, usize>(count)?;
        allocations.acquire(text)?;
    }
    Ok(())
}

/// Find SCCs for the lowering phase
/// Returns SCCs in reverse topological order (dependencies first)
pub fn find_sccs_for_lowering(graph: &DependencyGraph) -> Vec<Vec<String>> {
    find_sccs(graph)
}

pub(crate) fn find_sccs_for_lowering_with_allocation_owner(
    graph: &DependencyGraph,
    scope: &mut crate::compile::CompilationAllocationScope<'_>,
) -> Result<Vec<Vec<String>>> {
    find_sccs_with_owner(graph, &mut StratificationAllocations { scope: Some(scope) })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast::*;

    fn create_tc_program() -> Program {
        let mut program = Program::new();
        program.rules.push(Rule {
            head: Atom {
                predicate: "edge".into(),
                terms: vec![Term::Integer(1), Term::Integer(2)],
            },
            body: vec![],
        });
        program.rules.push(Rule {
            head: Atom {
                predicate: "reach".into(),
                terms: vec![Term::Variable("X".into()), Term::Variable("Y".into())],
            },
            body: vec![BodyLiteral::Positive(Atom {
                predicate: "edge".into(),
                terms: vec![Term::Variable("X".into()), Term::Variable("Y".into())],
            })],
        });
        program.rules.push(Rule {
            head: Atom {
                predicate: "reach".into(),
                terms: vec![Term::Variable("X".into()), Term::Variable("Z".into())],
            },
            body: vec![
                BodyLiteral::Positive(Atom {
                    predicate: "reach".into(),
                    terms: vec![Term::Variable("X".into()), Term::Variable("Y".into())],
                }),
                BodyLiteral::Positive(Atom {
                    predicate: "edge".into(),
                    terms: vec![Term::Variable("Y".into()), Term::Variable("Z".into())],
                }),
            ],
        });
        program
    }

    fn create_isolated_program() -> Program {
        let mut program = Program::new();
        for i in 1..=3 {
            program.rules.push(Rule {
                head: Atom {
                    predicate: "node".into(),
                    terms: vec![Term::Integer(i)],
                },
                body: vec![],
            });
        }
        program.rules.push(Rule {
            head: Atom {
                predicate: "edge".into(),
                terms: vec![Term::Integer(1), Term::Integer(2)],
            },
            body: vec![],
        });
        program.rules.push(Rule {
            head: Atom {
                predicate: "isolated".into(),
                terms: vec![Term::Variable("X".into())],
            },
            body: vec![
                BodyLiteral::Positive(Atom {
                    predicate: "node".into(),
                    terms: vec![Term::Variable("X".into())],
                }),
                BodyLiteral::Negated(Atom {
                    predicate: "edge".into(),
                    terms: vec![Term::Variable("X".into()), Term::Variable("Y".into())],
                }),
            ],
        });
        program
    }

    fn create_unstratifiable_program() -> Program {
        let mut program = Program::new();
        program.rules.push(Rule {
            head: Atom {
                predicate: "p".into(),
                terms: vec![],
            },
            body: vec![BodyLiteral::Negated(Atom {
                predicate: "q".into(),
                terms: vec![],
            })],
        });
        program.rules.push(Rule {
            head: Atom {
                predicate: "q".into(),
                terms: vec![],
            },
            body: vec![BodyLiteral::Negated(Atom {
                predicate: "p".into(),
                terms: vec![],
            })],
        });
        program
    }

    #[test]
    fn test_stratify_simple() {
        let program = create_tc_program();
        let result = stratify(&program);
        assert!(result.is_ok(), "Stratification failed: {:?}", result.err());
    }

    #[test]
    fn test_stratify_with_negation() {
        let program = create_isolated_program();
        let result = stratify(&program);
        assert!(result.is_ok(), "Stratification failed: {:?}", result.err());
        let strata = result.unwrap();
        assert!(
            strata.len() >= 2,
            "Expected at least 2 strata, got {}",
            strata.len()
        );
    }

    #[test]
    fn test_stratify_cycle_through_negation() {
        let program = create_unstratifiable_program();
        let result = stratify(&program);
        assert!(result.is_err(), "Should fail with cycle through negation");
        if let Err(XlogError::StratificationCycle(preds)) = result {
            assert!(preds.contains(&"p".to_string()) || preds.contains(&"q".to_string()));
        }
    }

    #[test]
    fn test_stratify_probabilistic_non_monotone_allows_exact_ddnnf() {
        // Non-monotone programs are now supported with exact_ddnnf via WFS
        let mut program = create_unstratifiable_program();
        program.directives.prob_engine = Some(ProbEngine::ExactDdnnf);

        let result = stratify(&program);
        assert!(
            result.is_ok(),
            "Expected exact_ddnnf to allow non-monotone recursion (via WFS), got: {:?}",
            result.err()
        );
    }

    #[test]
    fn test_stratify_probabilistic_non_monotone_allows_mc() {
        let mut program = create_unstratifiable_program();
        program.directives.prob_engine = Some(ProbEngine::Mc);

        let result = stratify(&program);
        assert!(
            result.is_ok(),
            "Expected mc to allow non-monotone recursion, got: {:?}",
            result.err()
        );
    }

    #[test]
    fn test_dependency_graph_construction() {
        let program = create_tc_program();
        let graph = build_dependency_graph(&program);
        assert!(graph.predicates.contains("edge"));
        assert!(graph.predicates.contains("reach"));
        let reach_deps = graph.outgoing("reach");
        assert!(!reach_deps.is_empty());
    }

    #[test]
    fn test_analyze_stratification_detects_non_monotone() {
        let program = create_unstratifiable_program(); // p :- not q. q :- not p.
        let result = analyze_stratification(&program);

        assert!(
            !result.non_monotone_sccs.is_empty(),
            "Should detect non-monotone SCC"
        );
        // The SCC containing p and q should be marked as non-monotone
        let has_non_monotone = result.sccs.iter().enumerate().any(|(i, scc)| {
            result.non_monotone_sccs.contains(&i)
                && (scc.contains(&"p".to_string()) || scc.contains(&"q".to_string()))
        });
        assert!(has_non_monotone, "SCC with p/q should be non-monotone");
    }

    #[test]
    fn test_analyze_stratification_stratified_program() {
        let program = create_isolated_program(); // isolated(X) :- node(X), not edge(X, Y).
        let result = analyze_stratification(&program);

        assert!(
            result.non_monotone_sccs.is_empty(),
            "Stratified program has no non-monotone SCCs"
        );
        assert!(
            result.strata.contains_key("isolated"),
            "isolated should have a stratum"
        );
        assert!(
            result.strata.contains_key("edge"),
            "edge should have a stratum"
        );

        // isolated depends negatively on edge, so isolated.stratum > edge.stratum
        let isolated_stratum = result.strata.get("isolated").unwrap();
        let edge_stratum = result.strata.get("edge").unwrap();
        assert!(
            isolated_stratum > edge_stratum,
            "isolated should be in higher stratum than edge"
        );
    }
}
