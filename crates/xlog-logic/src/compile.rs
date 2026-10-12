//! Compilation pipeline for XLOG programs
//!
//! This module compiles the core XLOG AST into execution plans. The compilation
//! process consists of:
//!
//! 1. **Parsing**: Convert source text to AST (`parser::parse_program`)
//! 2. **Stratification**: Analyze negation/aggregation dependencies (`stratify::stratify`)
//! 3. **Lowering**: Transform AST to Relational IR (`lower::Lowerer::lower_program`)
//!
//! The `Compiler` struct orchestrates these phases and provides a single entry
//! point via the `compile` method. User-defined functions are a source-level
//! normalization step: callers using this low-level compiler must expand them
//! before compilation. Execution-facing `LogicProgram` APIs perform that
//! normalization for parsed and source inputs.

use std::path::{Path, PathBuf};

use xlog_core::{Result, XlogError};
use xlog_ir::{ExecutionPlan, GeneratedQueryRuleProvenance};
use xlog_stats::{StatsManager, StatsSnapshot};

use crate::compiler_config::CompilerConfig;
use crate::list_normalize::normalize_list_builtins_owned;
use crate::lower::Lowerer;
use crate::magic_sets::rewrite_magic_sets_owned;
use crate::meta_normalize::normalize_meta_builtins_owned;
use crate::module::ModuleError;
use crate::optimizer::Optimizer;
use crate::parser::parse_program;
use crate::resolver::ModuleResolver;
use crate::stratify::stratify;
use crate::{BodyLiteral, Program, Query, Rule as AstRule, Term};

/// The XLOG compiler orchestrates the full compilation pipeline.
///
/// This is the core AST-to-RIR compiler. It does not resolve imports or expand
/// user-defined functions; callers must perform those source-level steps first,
/// or use the execution-facing `LogicProgram` compilation APIs.
///
/// # Example
///
/// ```ignore
/// use xlog_logic::compile::Compiler;
///
/// let mut compiler = Compiler::new();
/// let plan = compiler.compile(r#"
///     edge(1, 2).
///     edge(2, 3).
///     reach(X, Y) :- edge(X, Y).
///     reach(X, Z) :- reach(X, Y), edge(Y, Z).
/// "#)?;
/// ```
pub struct Compiler {
    lowerer: Lowerer,
}

use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use xlog_core::{RelId, Schema};

/// Original allocation receipts for one canonical compilation. The caller's
/// callback is the capacity authority; this scope neither chooses a budget nor
/// grants parsing, admission, or execution rights.
pub struct CompilationAllocationScope<'a> {
    reserve: &'a mut dyn FnMut(u64) -> Result<Arc<dyn Send + Sync>>,
    receipts: Vec<Arc<dyn Send + Sync>>,
    planner_threshold: Option<f64>,
    receipt_storage: Option<Arc<dyn Send + Sync>>,
    owner_metadata: Option<Arc<dyn Send + Sync>>,
}

struct CompilationAllocationReceipts {
    _receipts: Vec<Arc<dyn Send + Sync>>,
    _receipt_storage: Option<Arc<dyn Send + Sync>>,
    _metadata: Arc<dyn Send + Sync>,
}

pub fn formatted_text_allocation_bytes(arguments: std::fmt::Arguments<'_>) -> Result<u64> {
    struct Extent(u64);
    impl std::fmt::Write for Extent {
        fn write_str(&mut self, text: &str) -> std::fmt::Result {
            self.0 = self
                .0
                .checked_add(u64::try_from(text.len()).map_err(|_| std::fmt::Error)?)
                .ok_or(std::fmt::Error)?;
            Ok(())
        }
    }
    let overflow = || XlogError::Compilation("compiler text allocation extent overflow".into());
    let mut extent = Extent(0);
    std::fmt::write(&mut extent, arguments).map_err(|_| overflow())?;
    // The canonical formatter grows a byte vector. Initial literal estimation
    // and geometric growth need at most twice the actual formatted extent;
    // old and replacement buffers can coexist during a growth operation.
    extent
        .0
        .checked_mul(2)
        .and_then(|n| n.checked_add(8))
        .and_then(|n| n.checked_mul(3))
        .ok_or_else(overflow)
}

pub fn hash_allocation_bytes<K, V>(count: usize) -> Result<u64> {
    if count == 0 {
        return Ok(0);
    }
    let overflow = || XlogError::Compilation("compiler table allocation extent overflow".into());
    // Swiss-table storage has one control byte per bucket, a trailing control
    // group, and a seven-eighths maximum load factor. Eight buckets also cover
    // the original small-table special cases.
    let buckets = count
        .checked_mul(8)
        .and_then(|n| n.checked_add(6))
        .map(|n| n / 7)
        .and_then(|n| n.max(8).checked_next_power_of_two())
        .ok_or_else(overflow)?;
    buckets
        .checked_mul(
            std::mem::size_of::<(K, V)>()
                .checked_add(1)
                .ok_or_else(overflow)?,
        )
        .and_then(|n| n.checked_add(16))
        .and_then(|n| u64::try_from(n).ok())
        .ok_or_else(overflow)
}

pub fn push_vector_allocation_bytes<T>(count: usize) -> Result<u64> {
    let width = std::mem::size_of::<T>();
    if count == 0 || width == 0 {
        return Ok(0);
    }
    let minimum = if width == 1 {
        8
    } else if width <= 1024 {
        4
    } else {
        1
    };
    // Capacity of the actual push-built vector, including its old buffer at a
    // geometric growth operation. This does not allocate a counting vector.
    count
        .checked_mul(2)
        .map(|n| n.max(minimum))
        .and_then(|n| n.checked_mul(3))
        .and_then(|n| n.checked_mul(width))
        .and_then(|n| u64::try_from(n).ok())
        .ok_or_else(|| XlogError::Compilation("compiler vector allocation extent overflow".into()))
}

pub fn btree_insert_allocation_bytes<K, V>(length: usize) -> Result<u64> {
    let overflow = || XlogError::Compilation("compiler tree allocation extent overflow".into());
    let mut height = 1usize;
    let mut entries = length;
    while entries >= 6 {
        entries /= 6;
        height = height.checked_add(1).ok_or_else(overflow)?;
    }
    let node = (2 * std::mem::size_of::<usize>())
        .checked_add(2 * std::mem::size_of::<u16>())
        .and_then(|n| n.checked_add(11usize.checked_mul(std::mem::size_of::<(K, V)>())?))
        .and_then(|n| n.checked_add(12 * std::mem::size_of::<usize>()))
        .ok_or_else(overflow)?;
    let alignment = std::mem::align_of::<(K, V)>().max(std::mem::align_of::<usize>());
    let node = node
        .checked_add(alignment - 1)
        .map(|bytes| bytes / alignment * alignment)
        .ok_or_else(overflow)?;
    node.checked_mul(height.checked_add(1).ok_or_else(overflow)?)
        .and_then(|n| u64::try_from(n).ok())
        .ok_or_else(overflow)
}

impl<'a> CompilationAllocationScope<'a> {
    /// Start an empty scope without allocating receipt storage.
    pub fn new(reserve: &'a mut dyn FnMut(u64) -> Result<Arc<dyn Send + Sync>>) -> Self {
        Self {
            reserve,
            receipts: Vec::new(),
            planner_threshold: None,
            receipt_storage: None,
            owner_metadata: None,
        }
    }

    /// A stage-local receipt scope under the exact same reservation authority
    /// and captured planner configuration. It creates no pool or receipt until
    /// an actual stage allocation is acquired.
    pub fn child_scope(&mut self) -> CompilationAllocationScope<'_> {
        CompilationAllocationScope {
            reserve: &mut *self.reserve,
            receipts: Vec::new(),
            planner_threshold: self.planner_threshold,
            receipt_storage: None,
            owner_metadata: None,
        }
    }

    /// Attach the caller's original parsed planner configuration. Owned
    /// planning never reads or substitutes an environment value.
    pub fn with_planner_threshold(mut self, threshold: f64) -> Result<Self> {
        if !threshold.is_finite() || threshold <= 0.0 {
            return Err(XlogError::Compilation(
                "original planner threshold must be finite and positive".into(),
            ));
        }
        self.planner_threshold = Some(threshold);
        Ok(self)
    }

    pub(crate) fn planner_threshold(&self) -> Option<f64> {
        self.planner_threshold
    }

    /// Receipt boundary for a genuine attempt. The caller may retire the
    /// suffix only after all of that failed attempt's actual data has dropped.
    pub fn receipt_checkpoint(&self) -> usize {
        self.receipts.len()
    }

    /// Retire a known-discarded attempt; retained receipt-vector capacity keeps
    /// its separate original owner and cannot be released by this operation.
    pub fn retire_since(&mut self, checkpoint: usize) -> Result<()> {
        if checkpoint > self.receipts.len() {
            return Err(XlogError::Compilation(
                "compiler receipt boundary is not original".into(),
            ));
        }
        self.receipts.truncate(checkpoint);
        Ok(())
    }

    fn reserve_receipt_storage(&mut self) -> Result<()> {
        if self.receipts.len() == self.receipts.capacity() {
            let capacity = self
                .receipts
                .len()
                .checked_add(1)
                .and_then(|n| n.max(4).checked_next_power_of_two())
                .ok_or_else(|| {
                    XlogError::Compilation("compiler receipt allocation extent overflow".into())
                })?;
            let bytes = capacity
                .checked_mul(std::mem::size_of::<Arc<dyn Send + Sync>>())
                .and_then(|n| u64::try_from(n).ok())
                .ok_or_else(|| {
                    XlogError::Compilation("compiler receipt allocation extent overflow".into())
                })?;
            let owner = (self.reserve)(bytes)?;
            self.receipts.reserve_exact(capacity - self.receipts.len());
            self.receipt_storage = Some(owner);
        }
        Ok(())
    }

    /// Acquire custody for an actual temporary owner. Unlike `acquire`, the
    /// caller retains this receipt in that owner and retires it with its data.
    pub fn reserve_allocation(&mut self, bytes: u64) -> Result<Arc<dyn Send + Sync>> {
        (self.reserve)(bytes)
    }

    /// Acquire the returned owner's own metadata before the result's data is
    /// constructed. Final transfer then cannot refuse capacity after that data
    /// has already outlived this scope's receipts.
    pub fn prepare_owner(&mut self) -> Result<()> {
        if self.owner_metadata.is_none() {
            let bytes = std::mem::size_of::<CompilationAllocationReceipts>()
                .checked_add(2 * std::mem::size_of::<usize>())
                .and_then(|n| n.checked_add(xlog_core::error::OwnedDiagnostic::allocation_bytes()))
                .and_then(|n| u64::try_from(n).ok())
                .ok_or_else(|| {
                    XlogError::Compilation("compiler owner allocation extent overflow".into())
                })?;
            self.owner_metadata = Some((self.reserve)(bytes)?);
        }
        Ok(())
    }

    /// Retain the already-acquired receipt of an actual returned allocation.
    pub fn retain(&mut self, receipt: Arc<dyn Send + Sync>) -> Result<()> {
        self.reserve_receipt_storage()?;
        self.receipts.push(receipt);
        Ok(())
    }

    /// Acquire a receipt before its actual allocation. The receipt collection
    /// itself is charged before growth and retained by the same original scope.
    pub fn acquire(&mut self, bytes: u64) -> Result<Arc<dyn Send + Sync>> {
        self.reserve_receipt_storage()?;
        let receipt = (self.reserve)(bytes)?;
        self.receipts.push(receipt.clone());
        Ok(receipt)
    }

    /// Transfer all receipts into the original returned compiler owner. Its
    /// metadata is acquired before the owning Arc is allocated.
    pub fn into_owner(mut self) -> Result<Arc<dyn Send + Sync>> {
        self.prepare_owner()?;
        let metadata = self
            .owner_metadata
            .take()
            .expect("prepared compiler owner metadata");
        Ok(Arc::new(CompilationAllocationReceipts {
            _receipts: self.receipts,
            _receipt_storage: self.receipt_storage,
            _metadata: metadata,
        }))
    }

    /// Preserve a prepared failed attempt's diagnostic and original receipts.
    /// The two carrier allocations were acquired by `prepare_owner` before
    /// the attempt. No new reservation can replace the original failure here.
    pub fn into_failure(mut self, error: XlogError) -> XlogError {
        let metadata = self
            .owner_metadata
            .take()
            .expect("failed compiler attempt requires its prepared original owner");
        let allocations = Arc::new(CompilationAllocationReceipts {
            _receipts: self.receipts,
            _receipt_storage: self.receipt_storage,
            _metadata: Arc::clone(&metadata),
        });
        error.with_diagnostic_owner(allocations, metadata)
    }
}

/// Capacity of the original compiler's scalar binary subset, before owned
/// normalization, graph, schema, and physical-plan storage is allocated.
/// The caller must first use its canonical binary-rule and ground-query
/// validators; this helper is capacity metadata, not a source admission.
pub fn positive_binary_compiler_allocation_bound(program: &Program) -> Result<u64> {
    use crate::ast::{Atom, PredColumn, PredDecl, TypeRef};
    use crate::stratify::{DepEdge, DependencyGraph, Stratum};
    use std::mem::size_of;
    let overflow = || XlogError::Compilation("binary compiler allocation extent overflow".into());
    let add = |a: usize, b: usize| a.checked_add(b).ok_or_else(overflow);
    let mul = |a: usize, b: usize| a.checked_mul(b).ok_or_else(overflow);
    let vector = |count: usize, width: usize| mul(add(mul(count, 2)?, 4)?, width);
    let table = |count: usize, width: usize| -> Result<usize> {
        if count == 0 {
            return Ok(0);
        }
        let buckets = add(mul(count, 8)?, 6)?
            .checked_div(7)
            .and_then(|n| n.max(8).checked_next_power_of_two())
            .ok_or_else(overflow)?;
        add(mul(buckets, add(width, 1)?)?, 16)
    };
    let mut ast = add(
        size_of::<Program>(),
        add(
            vector(program.predicates.len(), size_of::<PredDecl>())?,
            add(
                vector(program.rules.len(), size_of::<AstRule>())?,
                vector(program.queries.len(), size_of::<Query>())?,
            )?,
        )?,
    )?;
    let mut max_name = "__xlog_query_".len() + 20;
    let mut names = 0usize;
    for declaration in &program.predicates {
        ast = add(ast, size_of::<PredDecl>())?;
        ast = add(ast, declaration.name.len())?;
        names = add(names, declaration.name.len())?;
        max_name = max_name.max(declaration.name.len());
        ast = add(ast, vector(declaration.types.len(), size_of::<TypeRef>())?)?;
        {
            ast = add(
                ast,
                vector(declaration.columns.len(), size_of::<PredColumn>())?,
            )?;
            for column in &declaration.columns {
                if let Some(name) = &column.name {
                    ast = add(ast, name.len())?;
                    max_name = max_name.max(name.len());
                }
            }
        }
    }
    let atom_bytes = |atom: &Atom| -> Result<usize> {
        let mut bytes = add(size_of::<Atom>(), atom.predicate.len())?;
        bytes = add(bytes, vector(atom.terms.len(), size_of::<Term>())?)?;
        for term in &atom.terms {
            match term {
                Term::Variable(name) => bytes = add(bytes, name.len())?,
                Term::Integer(_) | Term::Anonymous => {}
                _ => {
                    return Err(XlogError::Compilation(
                        "binary compiler capacity requires the original scalar subset".into(),
                    ));
                }
            }
        }
        Ok(bytes)
    };
    for rule in &program.rules {
        ast = add(ast, add(size_of::<AstRule>(), atom_bytes(&rule.head)?)?)?;
        names = add(names, rule.head.predicate.len())?;
        max_name = max_name.max(rule.head.predicate.len());
        if rule.body.len() > 2 {
            return Err(XlogError::Compilation(
                "binary compiler capacity requires at most two original body atoms".into(),
            ));
        }
        ast = add(ast, vector(rule.body.len(), size_of::<BodyLiteral>())?)?;
        for literal in &rule.body {
            let BodyLiteral::Positive(atom) = literal else {
                return Err(XlogError::Compilation(
                    "binary compiler capacity requires positive original body atoms".into(),
                ));
            };
            ast = add(ast, atom_bytes(atom)?)?;
            names = add(names, atom.predicate.len())?;
            max_name = max_name.max(atom.predicate.len());
            for term in &atom.terms {
                if let Term::Variable(name) = term {
                    max_name = max_name.max(name.len());
                }
            }
        }
    }
    for query in &program.queries {
        ast = add(ast, add(size_of::<Query>(), atom_bytes(&query.atom)?)?)?;
    }
    let queries = program.queries.len();
    let rules = add(program.rules.len(), queries)?;
    let relations = add(program.predicates.len(), queries)?;
    let edges = add(mul(program.rules.len(), 2)?, queries)?;
    // Source snapshot, outer meta/list normalizers, prepared frontend clone,
    // query desugaring, inner meta/list normalizers, and error/reference copies.
    // These passes retain no generated finite lists or compound terms in the
    // canonically validated scalar binary subset.
    let mut bytes = mul(ast, 8)?;
    let generated_ast = add(
        size_of::<AstRule>(),
        add(
            size_of::<Atom>(),
            add(size_of::<BodyLiteral>(), mul(3, size_of::<Term>())?)?,
        )?,
    )?;
    bytes = add(
        bytes,
        mul(queries, mul(add(generated_ast, mul(max_name, 3)?)?, 8)?)?,
    )?;
    // Stratification and lowering each own a dependency graph. Tarjan's
    // indices/lowlinks, stack/on-stack sets, successor arrays and SCC/stratum
    // snapshots are included separately, not as a caller-supplied graph count.
    let edge = add(size_of::<DepEdge>(), mul(max_name, 2)?)?;
    let graph = add(
        size_of::<DependencyGraph>(),
        add(
            add(vector(edges, edge)?, vector(edges, size_of::<usize>())?)?,
            add(
                table(relations, add(size_of::<String>(), max_name)?)?,
                table(relations, add(size_of::<(String, Vec<usize>)>(), max_name)?)?,
            )?,
        )?,
    )?;
    let tarjan = add(
        mul(
            table(relations, add(size_of::<(String, usize)>(), max_name)?)?,
            2,
        )?,
        add(
            table(relations, add(size_of::<String>(), max_name)?)?,
            add(
                vector(relations, add(size_of::<String>(), max_name)?)?,
                vector(edges, add(size_of::<String>(), max_name)?)?,
            )?,
        )?,
    )?;
    let strata = add(
        vector(relations, size_of::<Stratum>())?,
        add(
            vector(relations, size_of::<Vec<String>>())?,
            vector(relations, add(size_of::<String>(), max_name)?)?,
        )?,
    )?;
    bytes = add(bytes, mul(add(add(graph, tarjan)?, strata)?, 4)?)?;
    // Meta/list/safety environments, schema inference, relation ids and fact
    // counters all use the original relation/coordinate rosters. Three
    // optimizer schema snapshots plus outer returned schemas remain distinct.
    let schema = add(
        size_of::<Schema>(),
        add(
            vector(2, size_of::<(String, xlog_core::ScalarType)>())?,
            add(
                vector(2, size_of::<String>())?,
                add(mul(max_name, 4)?, vector(2, size_of::<usize>())?)?,
            )?,
        )?,
    )?;
    bytes = add(
        bytes,
        mul(
            table(relations, add(size_of::<String>(), add(max_name, schema)?)?)?,
            8,
        )?,
    )?;
    bytes = add(
        bytes,
        mul(
            table(
                relations,
                add(size_of::<(String, xlog_core::RelId)>(), max_name)?,
            )?,
            4,
        )?,
    )?;
    bytes = add(
        bytes,
        mul(
            table(rules, add(size_of::<(String, Vec<usize>)>(), max_name)?)?,
            4,
        )?,
    )?;
    let coordinate = add(size_of::<(String, xlog_core::ScalarType)>(), max_name)?;
    bytes = add(bytes, mul(table(4, coordinate)?, add(rules, 4)?)?)?;
    bytes = add(
        bytes,
        table(
            relations,
            size_of::<(xlog_core::RelId, xlog_stats::RelationStats)>(),
        )?,
    )?;
    bytes = add(
        bytes,
        crate::lower::positive_binary_lowering_allocation_bound(program, max_name)?,
    )?;
    // Generated-head provenance and sorting/reference arrays are live alongside
    // the plan; a failed semantic check may format every original identifier.
    bytes = add(
        bytes,
        vector(queries, size_of::<xlog_ir::GeneratedQueryRuleProvenance>())?,
    )?;
    bytes = add(bytes, vector(rules, size_of::<(usize, usize)>())?)?;
    bytes = add(bytes, mul(names, 2)?)?;
    bytes = add(
        bytes,
        add(
            size_of::<Compiler>(),
            add(
                size_of::<Optimizer>(),
                add(size_of::<StatsManager>(), 6 * size_of::<usize>())?,
            )?,
        )?,
    )?;
    u64::try_from(bytes).map_err(|_| overflow())
}

impl Default for Compiler {
    fn default() -> Self {
        Self::new()
    }
}

impl Compiler {
    /// Create a new compiler instance.
    pub fn new() -> Self {
        Self {
            lowerer: Lowerer::new(),
        }
    }

    /// Set the maximum active rules for TensorMaskedJoin (16..=128).
    pub fn set_max_active_rules(&mut self, max: usize) {
        self.lowerer.set_max_active_rules(max);
    }

    /// Compile XLOG source code into an execution plan.
    ///
    /// This is the main entry point for core compilation. It chains together:
    /// 1. Parsing (source → AST)
    /// 2. Stratification (analyze dependencies, check for cycles)
    /// 3. Lowering (AST → Relational IR execution plan)
    ///
    /// Function definitions and calls must already be expanded into the core
    /// language. Use an execution-facing `LogicProgram` API when source-level
    /// import resolution and function normalization are required.
    ///
    /// # Arguments
    ///
    /// * `source` - The XLOG source code as a string
    ///
    /// # Returns
    ///
    /// * `Ok(ExecutionPlan)` - The compiled execution plan ready for execution
    /// * `Err(XlogError)` - If any compilation phase fails:
    ///   - `XlogError::Parse` - Syntax errors in the source
    ///   - `XlogError::StratificationCycle` - Unstratifiable negation/aggregation
    ///   - `XlogError::Compilation` - Other semantic errors
    ///
    /// # Example
    ///
    /// ```ignore
    /// let mut compiler = Compiler::new();
    ///
    /// // Compile a simple transitive closure program
    /// let plan = compiler.compile(r#"
    ///     edge(1, 2).
    ///     edge(2, 3).
    ///     reach(X, Y) :- edge(X, Y).
    ///     reach(X, Z) :- reach(X, Y), edge(Y, Z).
    /// "#)?;
    ///
    /// // The plan can now be executed by xlog-runtime
    /// ```
    pub fn compile(&mut self, source: &str) -> Result<ExecutionPlan> {
        self.compile_with_stats_snapshot(source, None)
    }

    /// Compile XLOG source code into an execution plan, optionally seeding the optimizer
    /// with a runtime statistics snapshot.
    ///
    /// This entry point delegates through the composable config-aware API
    /// with `CompilerConfig::default()`, which preserves existing triangle,
    /// 4-cycle, recursive, and selectivity-aware dispatch behavior
    /// bit-identically.
    pub fn compile_with_stats_snapshot(
        &mut self,
        source: &str,
        stats_snapshot: Option<&StatsSnapshot>,
    ) -> Result<ExecutionPlan> {
        self.compile_with_config_and_stats_snapshot(
            source,
            &CompilerConfig::default(),
            stats_snapshot,
        )
    }

    /// Composable entry point that accepts a `CompilerConfig`.
    ///
    /// Default-config callers should keep using `compile()` /
    /// `compile_with_stats_snapshot()`. This entry point exists so callers can
    /// enable the variable-ordering cost model per call without an environment
    /// override.
    pub fn compile_with_config_and_stats_snapshot(
        &mut self,
        source: &str,
        config: &CompilerConfig,
        stats_snapshot: Option<&StatsSnapshot>,
    ) -> Result<ExecutionPlan> {
        let program = parse_program(source)?;
        self.compile_owned_program_with_config_and_stats_snapshot(program, config, stats_snapshot)
    }

    /// Compile a parsed XLOG program into an execution plan.
    ///
    /// This is useful for callers that want to inspect the AST (facts, queries,
    /// constraints) while compiling without reparsing. The parsed program must
    /// already be normalized to the core language, including function expansion.
    pub fn compile_program(&mut self, program: &Program) -> Result<ExecutionPlan> {
        self.compile_program_with_stats_snapshot(program, None)
    }

    /// Compile a program whose authored constraint identities were prepared at
    /// the outer full-program boundary.
    pub fn compile_prepared_program(&mut self, program: &Program) -> Result<ExecutionPlan> {
        self.compile_prepared_program_with_stats_snapshot(program, None)
    }

    /// Compile a parsed XLOG program into an execution plan, optionally seeding the optimizer.
    ///
    /// Delegates to [`Self::compile_program_with_config_and_stats_snapshot`]
    /// with `CompilerConfig::default()`.
    pub fn compile_program_with_stats_snapshot(
        &mut self,
        program: &Program,
        stats_snapshot: Option<&StatsSnapshot>,
    ) -> Result<ExecutionPlan> {
        self.compile_program_with_config_and_stats_snapshot(
            program,
            &CompilerConfig::default(),
            stats_snapshot,
        )
    }

    /// Compile an already prepared program with an optional statistics snapshot.
    pub fn compile_prepared_program_with_stats_snapshot(
        &mut self,
        program: &Program,
        stats_snapshot: Option<&StatsSnapshot>,
    ) -> Result<ExecutionPlan> {
        self.compile_prepared_program_with_config_and_stats_snapshot(
            program,
            &CompilerConfig::default(),
            stats_snapshot,
        )
    }

    /// Validate a parsed program through the production lowering contracts without
    /// requiring the dependency graph to be stratifiable.
    ///
    /// This is used by epistemic preparation, where a supported negated modal cycle
    /// must reach well-founded execution rather than fail ordinary stratification.
    /// The normal preprocessing, negation-safety, schema/type, constant, arithmetic,
    /// and projection checks still run unchanged.
    pub(crate) fn validate_program_without_stratification(
        &mut self,
        program: &Program,
    ) -> Result<()> {
        program.validate_prepared_authored_constraint_identity()?;
        let program = run_frontend_passes(program.clone())?;
        self.lowerer.validate_program_without_plan(&program)
    }

    pub(crate) fn validate_program_without_stratification_with_allocation_owner(
        &mut self,
        program: &Program,
        scope: &mut CompilationAllocationScope<'_>,
    ) -> Result<()> {
        program.validate_prepared_authored_constraint_identity()?;
        scope.acquire(program.owned_allocation_bytes()?)?;
        let program = run_frontend_passes_with_owner(program.clone(), &mut Some(&mut *scope))?;
        self.lowerer
            .validate_program_without_plan_with_allocation_owner(&program, scope)
    }

    /// Composable program-level entry point.
    ///
    /// `config` is currently consumed only by the promoter when it wires the
    /// variable-ordering cost model. With `CompilerConfig::default()`, the
    /// promoter keeps the default variable order.
    pub fn compile_program_with_config_and_stats_snapshot(
        &mut self,
        program: &Program,
        config: &CompilerConfig,
        stats_snapshot: Option<&StatsSnapshot>,
    ) -> Result<ExecutionPlan> {
        self.compile_owned_program_with_config_and_stats_snapshot(
            program.clone(),
            config,
            stats_snapshot,
        )
    }

    /// [`Self::compile_program_with_config_and_stats_snapshot`] for a program
    /// the caller hands over by value (no clone of the AST).
    pub fn compile_owned_program_with_config_and_stats_snapshot(
        &mut self,
        mut program: Program,
        config: &CompilerConfig,
        stats_snapshot: Option<&StatsSnapshot>,
    ) -> Result<ExecutionPlan> {
        if program.authored_constraint_source_bound.is_some() {
            program.validate_prepared_authored_constraint_identity()?;
        } else {
            program.prepare_authored_constraint_identity_at_root()?;
        }
        self.compile_prepared_owned_program_with_config_and_stats_snapshot(
            program,
            config,
            stats_snapshot,
        )
    }

    /// Config-aware compilation for an already prepared program.
    pub fn compile_prepared_program_with_config_and_stats_snapshot(
        &mut self,
        program: &Program,
        config: &CompilerConfig,
        stats_snapshot: Option<&StatsSnapshot>,
    ) -> Result<ExecutionPlan> {
        self.compile_prepared_owned_program_with_config_and_stats_snapshot(
            program.clone(),
            config,
            stats_snapshot,
        )
    }

    /// Compile through the original complete pipeline, retaining each original
    /// allocation receipt in the caller's compilation scope.
    pub fn compile_prepared_program_with_allocation_owner(
        &mut self,
        program: &Program,
        scope: &mut CompilationAllocationScope<'_>,
    ) -> Result<ExecutionPlan> {
        scope.acquire(program.owned_allocation_bytes()?)?;
        self.compile_prepared_owned_program_core(
            program.clone(),
            &CompilerConfig::default(),
            None,
            &mut Some(scope),
        )
    }

    /// Statistics-aware form using the same original compilation owner.
    pub fn compile_prepared_program_with_stats_snapshot_and_allocation_owner(
        &mut self,
        program: &Program,
        stats_snapshot: Option<&StatsSnapshot>,
        scope: &mut CompilationAllocationScope<'_>,
    ) -> Result<ExecutionPlan> {
        scope.acquire(program.owned_allocation_bytes()?)?;
        self.compile_prepared_owned_program_core(
            program.clone(),
            &CompilerConfig::default(),
            stats_snapshot,
            &mut Some(scope),
        )
    }

    /// Owned-AST form of the same allocation-aware compilation pipeline.
    pub fn compile_prepared_owned_program_with_allocation_owner(
        &mut self,
        program: Program,
        scope: &mut CompilationAllocationScope<'_>,
    ) -> Result<ExecutionPlan> {
        self.compile_prepared_owned_program_core(
            program,
            &CompilerConfig::default(),
            None,
            &mut Some(scope),
        )
    }

    /// [`Self::compile_prepared_program_with_config_and_stats_snapshot`] for a
    /// program handed over by value.
    ///
    /// The frontend passes (desugar → meta → list → magic sets) each take and
    /// return the `Program` by value, so a fact-heavy AST is never cloned
    /// between passes; the `&Program` entry points clone exactly once.
    pub fn compile_prepared_owned_program_with_config_and_stats_snapshot(
        &mut self,
        program: Program,
        config: &CompilerConfig,
        stats_snapshot: Option<&StatsSnapshot>,
    ) -> Result<ExecutionPlan> {
        self.compile_prepared_owned_program_core(program, config, stats_snapshot, &mut None)
    }

    fn compile_prepared_owned_program_core(
        &mut self,
        program: Program,
        config: &CompilerConfig,
        stats_snapshot: Option<&StatsSnapshot>,
        scope: &mut Option<&mut CompilationAllocationScope<'_>>,
    ) -> Result<ExecutionPlan> {
        program.validate_prepared_authored_constraint_identity()?;
        let generated_query_heads = generated_query_heads_for_program_with_owner(&program, scope)?;
        let program = run_frontend_passes_with_owner(program, scope)?;

        // Phase 2: Stratify (analyze dependencies, detect cycles)
        let strata = match scope.as_deref_mut() {
            Some(scope) => crate::stratify::stratify_with_allocation_owner(&program, scope),
            None => stratify(&program),
        }
        .map_err(map_stratification_to_naf_error)?;

        // Convert strata to the format expected by the lowerer
        acquire_frontend_vector::<Vec<String>>(scope, strata.len())?;
        let strata_preds: Vec<Vec<String>> = strata.into_iter().map(|s| s.predicates).collect();

        // Phase 3: Lower AST to execution plan
        self.lowerer.set_strata(strata_preds);

        // If we have predicate names for the snapshot, use them to seed lowering-time
        // join ordering with better cardinality estimates.
        let mut cardinality_hints: HashMap<String, u64> = HashMap::new();
        if let Some(snapshot) = stats_snapshot {
            if !snapshot.rel_names.is_empty() {
                if scope.is_some() {
                    acquire_frontend(
                        scope,
                        hash_allocation_bytes::<RelId, &str>(snapshot.rel_names.len())?,
                    )?;
                }
                let rel_name_by_id: HashMap<RelId, &str> = snapshot
                    .rel_names
                    .iter()
                    .map(|(id, name)| (*id, name.as_str()))
                    .collect();
                for rel in &snapshot.relations {
                    if let Some(name) = rel_name_by_id.get(&rel.rel_id) {
                        if scope.is_some() {
                            if !cardinality_hints.contains_key(*name)
                                && cardinality_hints.len() == cardinality_hints.capacity()
                            {
                                acquire_frontend(
                                    scope,
                                    hash_allocation_bytes::<String, u64>(
                                        cardinality_hints.len() + 1,
                                    )?,
                                )?;
                            }
                            acquire_frontend(
                                scope,
                                u64::try_from(name.len()).map_err(|_| {
                                    XlogError::Compilation(
                                        "cardinality name extent overflow".into(),
                                    )
                                })?,
                            )?;
                        }
                        cardinality_hints.insert((*name).to_string(), rel.cardinality);
                    }
                }
            }
        }
        self.lowerer.set_cardinality_hints(cardinality_hints);

        let mut plan = match scope.as_deref_mut() {
            Some(scope) => self
                .lowerer
                .lower_program_with_allocation_owner(&program, scope)?,
            None => self.lowerer.lower_program(&program)?,
        };

        // Phase 4: Optimize (predicate pushdown + cost-aware rewrites)
        //
        // Seed statistics with any known fact cardinalities so cost estimation has
        // at least a baseline for EDB relations.
        if scope.is_some() {
            acquire_frontend(
                scope,
                hash_allocation_bytes::<RelId, xlog_stats::RelationStats>(
                    self.lowerer.rel_ids().len(),
                )?,
            )?;
            acquire_frontend(
                scope,
                hash_allocation_bytes::<String, u64>(program.rules.len())?,
            )?;
        }
        let mut mgr = StatsManager::new();
        let mut fact_counts: HashMap<String, u64> = HashMap::new();
        for fact in program.facts() {
            acquire_frontend(scope, fact.head.predicate.len() as u64)?;
            *fact_counts.entry(fact.head.predicate.clone()).or_insert(0) += 1;
        }

        for (pred, rel_id) in self.lowerer.rel_ids() {
            mgr.register_relation(*rel_id);
            let rows = fact_counts.get(pred).copied().unwrap_or(0);
            if rows > 0 {
                mgr.update_cardinality(*rel_id, rows);
                if let Some(schema) = self.lowerer.schemas().get(pred) {
                    mgr.update_byte_size(*rel_id, rows * schema.row_size_bytes() as u64);
                }
            }
        }

        if let Some(snapshot) = stats_snapshot {
            reserve_stats_snapshot_merge(snapshot, self.lowerer.rel_ids().len(), scope)?;
            if snapshot.rel_names.is_empty() {
                mgr.merge_snapshot(snapshot);
            } else {
                let rel_name_by_id: HashMap<RelId, &str> = snapshot
                    .rel_names
                    .iter()
                    .map(|(id, name)| (*id, name.as_str()))
                    .collect();

                for rel in &snapshot.relations {
                    let Some(pred) = rel_name_by_id.get(&rel.rel_id) else {
                        continue;
                    };
                    let Some(rel_id) = self.lowerer.rel_ids().get(*pred) else {
                        continue;
                    };

                    let mut remapped = rel.clone();
                    remapped.rel_id = *rel_id;

                    if let Some(schema) = self.lowerer.schemas().get(*pred) {
                        remapped.column_stats.retain(|col| {
                            col.col_idx < schema.arity()
                                && schema.column_type(col.col_idx) == Some(col.dtype)
                        });
                    } else {
                        remapped.column_stats.clear();
                    }

                    mgr.register_relation(*rel_id);
                    if let Some(stats) = mgr.get_relation_stats_mut(*rel_id) {
                        *stats = remapped;
                    }
                }

                for js in &snapshot.join_selectivities {
                    if js.left_keys.len() != js.right_keys.len() {
                        continue;
                    }

                    let Some(left_pred) = rel_name_by_id.get(&js.left_rel) else {
                        continue;
                    };
                    let Some(right_pred) = rel_name_by_id.get(&js.right_rel) else {
                        continue;
                    };
                    let Some(&left_id) = self.lowerer.rel_ids().get(*left_pred) else {
                        continue;
                    };
                    let Some(&right_id) = self.lowerer.rel_ids().get(*right_pred) else {
                        continue;
                    };

                    let Some(left_schema) = self.lowerer.schemas().get(*left_pred) else {
                        continue;
                    };
                    let Some(right_schema) = self.lowerer.schemas().get(*right_pred) else {
                        continue;
                    };
                    if js.left_keys.iter().any(|&k| k >= left_schema.arity())
                        || js.right_keys.iter().any(|&k| k >= right_schema.arity())
                    {
                        continue;
                    }

                    mgr.set_join_selectivity(
                        left_id,
                        right_id,
                        js.left_keys.clone(),
                        js.right_keys.clone(),
                        js.selectivity,
                    );
                }
            }
        }

        // Build schemas by RelId for the optimizer
        let schemas_by_rel_id = schemas_snapshot_with_owner(&self.lowerer, scope)?;

        acquire_frontend(
            scope,
            (std::mem::size_of::<StatsManager>() + 2 * std::mem::size_of::<usize>()) as u64,
        )?;
        let stats_arc = Arc::new(mgr);

        match scope.as_deref_mut() {
            Some(scope) => {
                crate::optimizer::helper_split_pass::run_with_allocation_owner(
                    &mut plan,
                    &schemas_by_rel_id,
                    &stats_arc,
                    scope,
                    |schema, scope| {
                        self.lowerer
                            .create_helper_relation_with_allocation_owner(schema, scope)
                    },
                )?;
            }
            None => {
                crate::optimizer::helper_split_pass::run(
                    &mut plan,
                    &schemas_by_rel_id,
                    &stats_arc,
                    |schema| self.lowerer.create_helper_relation(schema),
                );
            }
        }

        let schemas_by_rel_id = schemas_snapshot_with_owner(&self.lowerer, scope)?;

        let mut optimizer = Optimizer::new(Arc::clone(&stats_arc));
        optimizer.set_schemas(schemas_by_rel_id);
        for rules in &mut plan.rules_by_scc {
            for rule in rules {
                let body = std::mem::replace(&mut rule.body, xlog_ir::RirNode::Unit);
                rule.body = match scope.as_deref_mut() {
                    Some(scope) => optimizer.optimize_with_allocation_owner(body, scope)?,
                    None => optimizer.optimize(body),
                };
            }
        }

        // Selectivity-aware reordering pass. Runs BETWEEN the optimizer loop
        // and promote_multiway.
        // Locked compile-pipeline ordering:
        //   lower → helper_split_pass → optimizer → selectivity_pass → promote_multiway
        //
        // Takes `rel_ids` so per-body Scans can be resolved against
        // `StatsManager`. Behavior on empty stats / unseeded relations is
        // no-op (safety floor).
        match scope.as_deref_mut() {
            Some(scope) => crate::optimizer::selectivity_pass::run_with_allocation_owner(
                &mut plan,
                &stats_arc,
                self.lowerer.rel_ids(),
                scope,
            )?,
            None => crate::optimizer::selectivity_pass::run(
                &mut plan,
                &stats_arc,
                self.lowerer.rel_ids(),
            ),
        }

        // Promote eligible triangle subtrees to RirNode::MultiWayJoin. Runs
        // *after* the optimizer so the optimizer never has to learn the new
        // variant. Fallback identity preserves binary-join semantics on
        // dispatch decline.
        //
        // Pass the lowerer's predicate→RelId map so the promoter can gate
        // recursive-SCC bodies on the count of in-SCC Scans (≤ 1 = promote,
        // ≥ 2 = skip).
        //
        // Also pass `&stats_arc` and the caller-provided `&CompilerConfig`.
        // With `CompilerConfig::default()` (`Disabled`), the promoter never
        // sets `var_order` and default dispatch is bit-identical.
        match scope.as_deref_mut() {
            Some(scope) => crate::promote::promote_multiway_with_allocation_owner(
                &mut plan,
                self.lowerer.rel_ids(),
                &stats_arc,
                config,
                scope,
            )?,
            None => crate::promote::promote_multiway(
                &mut plan,
                self.lowerer.rel_ids(),
                &stats_arc,
                config,
            ),
        }

        let schemas_by_rel_id = schemas_snapshot_with_owner(&self.lowerer, scope)?;

        match scope.as_deref_mut() {
            Some(scope) => {
                crate::optimizer::helper_split_pass::run_kclique_specs_with_allocation_owner(
                    &mut plan,
                    &schemas_by_rel_id,
                    scope,
                    |schema, scope| {
                        self.lowerer
                            .create_helper_relation_with_allocation_owner(schema, scope)
                    },
                )?;
            }
            None => {
                crate::optimizer::helper_split_pass::run_kclique_specs(
                    &mut plan,
                    &schemas_by_rel_id,
                    |schema| self.lowerer.create_helper_relation(schema),
                );
            }
        }

        acquire_frontend_vector::<GeneratedQueryRuleProvenance>(
            scope,
            generated_query_heads.len(),
        )?;
        plan.generated_query_rules = generated_query_heads
            .iter()
            .enumerate()
            .map(|(query_index, head)| {
                let count = plan
                    .rules_by_scc
                    .iter()
                    .try_fold(0usize, |n, rules| n.checked_add(rules.len()))
                    .ok_or_else(|| {
                        XlogError::Compilation("query provenance allocation extent overflow".into())
                    })?;
                acquire_frontend_vector::<(usize, usize)>(scope, count)?;
                let positions = plan
                    .rules_by_scc
                    .iter()
                    .enumerate()
                    .flat_map(|(scc_index, rules)| {
                        rules
                            .iter()
                            .enumerate()
                            .filter_map(move |(rule_index, rule)| {
                                (rule.head == *head).then_some((scc_index, rule_index))
                            })
                    })
                    .collect::<Vec<_>>();
                let [(scc_index, rule_index)] = positions.as_slice() else {
                    acquire_frontend_format(scope, format_args!(
                        "generated query head {head} must have exactly one compiled rule, found {}",
                        positions.len()
                    ))?;
                    return Err(XlogError::Compilation(format!(
                        "generated query head {head} must have exactly one compiled rule, found {}",
                        positions.len()
                    )));
                };
                Ok(GeneratedQueryRuleProvenance {
                    query_index,
                    scc_index: *scc_index,
                    rule_index: *rule_index,
                })
            })
            .collect::<Result<Vec<_>>>()?;

        Ok(plan)
    }

    /// Reset the compiler state for a fresh compilation.
    ///
    /// This creates a new lowerer, clearing any cached schemas or relation IDs
    /// from previous compilations.
    pub fn reset(&mut self) {
        self.lowerer = Lowerer::new();
    }

    /// Get the mapping from predicate names to relation IDs after compilation.
    ///
    /// This mapping is needed to register relations in the executor with
    /// the correct RelIds.
    pub fn rel_ids(&self) -> &HashMap<String, RelId> {
        self.lowerer.rel_ids()
    }

    /// Get the inferred schemas for predicates after compilation.
    ///
    /// These schemas are needed to create GPU buffers with correct column types.
    pub fn schemas(&self) -> &HashMap<String, Schema> {
        self.lowerer.schemas()
    }
}

fn generated_query_heads_for_program(program: &Program) -> Result<Vec<String>> {
    generated_query_heads_for_program_with_owner(program, &mut None)
}

fn schemas_snapshot_with_owner(
    lowerer: &Lowerer,
    scope: &mut Option<&mut CompilationAllocationScope<'_>>,
) -> Result<HashMap<RelId, Schema>> {
    if scope.is_some() {
        let count = lowerer
            .rel_ids()
            .keys()
            .filter(|name| lowerer.schemas().contains_key(*name))
            .count();
        acquire_frontend(scope, hash_allocation_bytes::<RelId, Schema>(count)?)?;
        for name in lowerer.rel_ids().keys() {
            if let Some(schema) = lowerer.schemas().get(name) {
                acquire_frontend(scope, schema.owned_allocation_bytes()?)?;
            }
        }
    }
    Ok(lowerer
        .rel_ids()
        .iter()
        .filter_map(|(name, id)| {
            lowerer
                .schemas()
                .get(name)
                .map(|schema| (*id, schema.clone()))
        })
        .collect())
}

fn reserve_stats_snapshot_merge(
    snapshot: &StatsSnapshot,
    original_relations: usize,
    scope: &mut Option<&mut CompilationAllocationScope<'_>>,
) -> Result<()> {
    if scope.is_none() {
        return Ok(());
    }
    acquire_frontend(
        scope,
        hash_allocation_bytes::<RelId, xlog_stats::RelationStats>(
            original_relations
                .checked_add(snapshot.relations.len())
                .ok_or_else(|| {
                    XlogError::Compilation("statistics registry allocation extent overflow".into())
                })?,
        )?,
    )?;
    acquire_frontend(
        scope,
        hash_allocation_bytes::<(RelId, RelId), xlog_stats::JoinSelectivity>(
            snapshot.join_selectivities.len(),
        )?,
    )?;
    acquire_frontend(
        scope,
        hash_allocation_bytes::<RelId, &str>(snapshot.rel_names.len())?,
    )?;
    for relation in &snapshot.relations {
        acquire_frontend_vector::<xlog_stats::ColumnStats>(scope, relation.column_stats.len())?;
        acquire_frontend_vector::<xlog_stats::PrefixDegreeStats>(
            scope,
            relation.prefix_degrees.len(),
        )?;
        acquire_frontend_vector::<xlog_stats::KeyHeatStats>(scope, relation.key_heats.len())?;
    }
    for selectivity in &snapshot.join_selectivities {
        // Caller key copies and the genuine cache entry's original key copies.
        for _ in 0..2 {
            acquire_frontend_vector::<usize>(scope, selectivity.left_keys.len())?;
            acquire_frontend_vector::<usize>(scope, selectivity.right_keys.len())?;
        }
    }
    Ok(())
}

fn generated_query_heads_for_program_with_owner(
    program: &Program,
    scope: &mut Option<&mut CompilationAllocationScope<'_>>,
) -> Result<Vec<String>> {
    acquire_frontend_vector::<String>(scope, program.queries.len())?;
    (0..program.queries.len())
        .map(|query_index| {
            acquire_frontend_format(scope, format_args!("__xlog_query_{query_index}"))?;
            let generated_head = format!("__xlog_query_{query_index}");
            let authored_collision = program
                .predicates
                .iter()
                .any(|declaration| declaration.name == generated_head)
                || program
                    .rules
                    .iter()
                    .any(|rule| rule.head.predicate == generated_head)
                || program
                    .learnable_rules
                    .iter()
                    .any(|rule| rule.head.predicate == generated_head);
            if authored_collision {
                acquire_frontend_format(
                    scope,
                    format_args!(
                        "authored relation {generated_head} collides with generated query head"
                    ),
                )?;
                return Err(XlogError::Compilation(format!(
                    "authored relation {generated_head} collides with generated query head"
                )));
            }
            Ok(generated_head)
        })
        .collect()
}

/// The source-level frontend passes shared by compilation and validation:
/// desugar queries/constraints → meta normalization → list normalization →
/// magic sets → negation-safety check. All by value, no AST clones.
fn run_frontend_passes(program: Program) -> Result<Program> {
    run_frontend_passes_with_owner(program, &mut None)
}

fn run_frontend_passes_with_owner(
    program: Program,
    scope: &mut Option<&mut CompilationAllocationScope<'_>>,
) -> Result<Program> {
    let program = desugar_queries_and_constraints_with_owner(program, scope)?;
    let program = match scope.as_deref_mut() {
        Some(scope) => crate::meta_normalize::normalize_meta_builtins_owned_with_allocation_owner(
            program, scope,
        )?,
        None => normalize_meta_builtins_owned(program)?,
    };
    let program = match scope.as_deref_mut() {
        Some(scope) => crate::list_normalize::normalize_list_builtins_owned_with_allocation_owner(
            program, scope,
        )?,
        None => normalize_list_builtins_owned(program)?,
    };
    let program = match scope.as_deref_mut() {
        Some(scope) => {
            crate::magic_sets::rewrite_magic_sets_owned_with_allocation_owner(program, scope)?
                .program
        }
        None => rewrite_magic_sets_owned(program)?.program,
    };
    validate_negation_safety_with_owner(&program, scope)?;
    Ok(program)
}

fn acquire_frontend(
    scope: &mut Option<&mut CompilationAllocationScope<'_>>,
    bytes: u64,
) -> Result<()> {
    if bytes != 0 {
        if let Some(scope) = scope.as_deref_mut() {
            scope.acquire(bytes)?;
        }
    }
    Ok(())
}

fn acquire_frontend_vector<T>(
    scope: &mut Option<&mut CompilationAllocationScope<'_>>,
    count: usize,
) -> Result<()> {
    if scope.is_some() {
        acquire_frontend(scope, push_vector_allocation_bytes::<T>(count)?)?;
    }
    Ok(())
}

fn acquire_frontend_format(
    scope: &mut Option<&mut CompilationAllocationScope<'_>>,
    arguments: std::fmt::Arguments<'_>,
) -> Result<()> {
    if scope.is_some() {
        acquire_frontend(scope, formatted_text_allocation_bytes(arguments)?)?;
    }
    Ok(())
}

fn acquire_frontend_variables(
    scope: &mut Option<&mut CompilationAllocationScope<'_>>,
    literal: &BodyLiteral,
) -> Result<()> {
    if scope.is_none() {
        return Ok(());
    }
    let mut count = 0usize;
    let mut text = 0u64;
    let mut overflow = false;
    literal.visit_variables(&mut |name| {
        if let Some(next) = count.checked_add(1) {
            count = next;
        } else {
            overflow = true;
        }
        if let Some(next) = text.checked_add(name.len() as u64) {
            text = next;
        } else {
            overflow = true;
        }
    });
    if overflow {
        return Err(XlogError::Compilation(
            "frontend variable allocation extent overflow".into(),
        ));
    }
    acquire_frontend_vector::<&str>(scope, count)?;
    acquire_frontend(scope, text)?;
    Ok(())
}

fn desugar_queries_and_constraints_with_owner(
    program: Program,
    scope: &mut Option<&mut CompilationAllocationScope<'_>>,
) -> Result<Program> {
    let mut out = program;
    let final_rules = out
        .rules
        .len()
        .checked_add(out.constraints.len())
        .and_then(|n| n.checked_add(out.queries.len()))
        .ok_or_else(|| XlogError::Compilation("frontend rule allocation extent overflow".into()))?;
    acquire_frontend_vector::<AstRule>(scope, final_rules)?;
    for constraint in &out.constraints {
        let authored_index = constraint.require_authored_index()?;
        acquire_frontend_format(scope, format_args!("__xlog_constraint_{authored_index}"))?;
        acquire_frontend_vector::<Term>(scope, 1)?;
        if scope.is_some() {
            acquire_frontend_vector::<BodyLiteral>(scope, constraint.body.len())?;
            for literal in &constraint.body {
                acquire_frontend(scope, literal.owned_allocation_bytes()?)?;
            }
        }
        let pred = format!("__xlog_constraint_{authored_index}");
        out.rules.push(AstRule {
            head: crate::ast::Atom {
                predicate: pred,
                terms: vec![Term::Integer(1)],
            },
            body: constraint.body.clone(),
        });
    }
    for (index, Query { atom }) in out.queries.iter().enumerate() {
        acquire_frontend_format(scope, format_args!("__xlog_query_{index}"))?;
        let pred = format!("__xlog_query_{index}");
        let mut head_terms = Vec::new();
        let mut seen = HashSet::new();
        for term in &atom.terms {
            if scope.is_some() {
                let mut count = Some(0usize);
                term.visit_variables(&mut |_| count = count.and_then(|n| n.checked_add(1)));
                let count = count.ok_or_else(|| {
                    XlogError::Compilation("query variable allocation extent overflow".into())
                })?;
                acquire_frontend_vector::<&str>(scope, count)?;
            }
            for name in term.variables() {
                if !seen.contains(name) {
                    if scope.is_some() && seen.len() == seen.capacity() {
                        acquire_frontend(
                            scope,
                            hash_allocation_bytes::<&str, ()>(seen.len() + 1)?,
                        )?;
                    }
                    if head_terms.len() == head_terms.capacity() {
                        acquire_frontend_vector::<Term>(scope, head_terms.len() + 1)?;
                    }
                    acquire_frontend(scope, name.len() as u64)?;
                }
                if seen.insert(name) {
                    head_terms.push(Term::Variable(name.to_string()));
                }
            }
        }
        if head_terms.is_empty() {
            acquire_frontend_vector::<Term>(scope, 1)?;
            head_terms.push(Term::Integer(1));
        }
        acquire_frontend_vector::<BodyLiteral>(scope, 1)?;
        if scope.is_some() {
            acquire_frontend(scope, atom.owned_allocation_bytes()?)?;
        }
        out.rules.push(AstRule {
            head: crate::ast::Atom {
                predicate: pred,
                terms: head_terms,
            },
            body: vec![BodyLiteral::Positive(atom.clone())],
        });
    }
    Ok(out)
}

fn validate_negation_safety_with_owner(
    program: &Program,
    scope: &mut Option<&mut CompilationAllocationScope<'_>>,
) -> Result<()> {
    for rule in &program.rules {
        acquire_frontend_format(scope, format_args!("rule {}", rule.head.predicate))?;
        validate_body_naf_safety_with_owner(
            &rule.body,
            &format!("rule {}", rule.head.predicate),
            scope,
        )?;
    }
    for constraint in &program.constraints {
        let index = constraint.require_authored_index()?;
        acquire_frontend_format(scope, format_args!("constraint {index}"))?;
        validate_body_naf_safety_with_owner(
            &constraint.body,
            &format!("constraint {index}"),
            scope,
        )?;
    }
    for (index, rule) in program.learnable_rules.iter().enumerate() {
        acquire_frontend_format(scope, format_args!("learnable rule {index}"))?;
        validate_body_naf_safety_with_owner(&rule.body, &format!("learnable rule {index}"), scope)?;
    }
    Ok(())
}

fn validate_body_naf_safety_with_owner(
    body: &[BodyLiteral],
    context: &str,
    scope: &mut Option<&mut CompilationAllocationScope<'_>>,
) -> Result<()> {
    if scope.is_some() {
        let mut count = Some(0usize);
        for literal in body {
            acquire_frontend_variables(scope, literal)?;
            match literal {
                BodyLiteral::Positive(atom) => {
                    atom.visit_variables(&mut |_| count = count.and_then(|n| n.checked_add(1)))
                }
                BodyLiteral::IsExpr(_) => count = count.and_then(|n| n.checked_add(1)),
                _ => {}
            }
        }
        let count = count.ok_or_else(|| {
            XlogError::Compilation("negation variable allocation extent overflow".into())
        })?;
        acquire_frontend(scope, hash_allocation_bytes::<String, ()>(count)?)?;
    }
    validate_body_naf_safety(body, context)
}

fn validate_body_naf_safety(body: &[BodyLiteral], context: &str) -> Result<()> {
    let mut bound: HashSet<String> = HashSet::new();
    for lit in body {
        match lit {
            BodyLiteral::Positive(atom) => {
                for name in atom.variables() {
                    bound.insert(name.to_string());
                }
            }
            BodyLiteral::Negated(atom) => {
                for name in atom.variables() {
                    if !bound.contains(name) {
                        return Err(naf_error(format!(
                            "unbound variable {} in negated atom {}/{} in {}; bind it before not with a positive atom or deterministic is expression, or use '_' for existential positions",
                            name,
                            atom.predicate,
                            atom.arity(),
                            context
                        )));
                    }
                }
            }
            BodyLiteral::IsExpr(is_expr) => {
                bound.insert(is_expr.target.clone());
            }
            BodyLiteral::Epistemic(_) => {}
            BodyLiteral::Comparison(_) | BodyLiteral::Univ(_) => {}
        }
    }
    Ok(())
}

fn map_stratification_to_naf_error(err: XlogError) -> XlogError {
    match err {
        XlogError::StratificationCycle(cycle) => naf_error(format!(
            "deterministic not atom must be stratified; cycle through negation or aggregation: {}",
            cycle.join(" -> ")
        )),
        other => other,
    }
}

fn naf_error(message: impl Into<String>) -> XlogError {
    XlogError::Compilation(format!("negation safety error: {}", message.into()))
}

/// Convenience function to compile source in one call.
///
/// This creates a short-lived compiler and compiles the source.
/// For multiple compilations, prefer creating a `Compiler` instance directly.
/// Function definitions and calls must already be expanded; execution-facing
/// `LogicProgram` APIs own source normalization and import resolution.
///
/// # Example
///
/// ```ignore
/// use xlog_logic::compile::compile;
///
/// let plan = compile("edge(1, 2). reach(X, Y) :- edge(X, Y).")?;
/// ```
pub fn compile(source: &str) -> Result<ExecutionPlan> {
    let mut compiler = Compiler::new();
    compiler.compile(source)
}

/// Load modules for an entry source file.
///
/// This function:
/// 1. Loads the entry module from the exact supplied path
/// 2. Loads all direct and transitive `.xlog` module dependencies
///
/// # Arguments
///
/// * `entry_file` - Path to the entry source file
/// * `search_paths` - Additional directories to search for modules
///
/// # Returns
///
/// The loaded module resolver with all dependencies resolved, or an error
/// if module resolution fails.
pub fn load_modules(
    entry_file: &Path,
    search_paths: Vec<PathBuf>,
) -> std::result::Result<ModuleResolver, ModuleError> {
    let mut resolver = ModuleResolver::new(search_paths);
    resolver.load_entry_file(entry_file)?;

    Ok(resolver)
}

#[cfg(test)]
mod tests {
    use super::*;
    use xlog_core::ScalarType;
    use xlog_ir::RirNode;
    use xlog_stats::ColumnStats;
    use xlog_stats::RelationStats;
    use xlog_stats::StatsManager;

    #[test]
    fn test_compiler_new() {
        let compiler = Compiler::new();
        // Just verify it can be created
        drop(compiler);
    }

    #[test]
    fn test_compile_fact() {
        let mut compiler = Compiler::new();
        let result = compiler.compile("edge(1, 2).");
        assert!(result.is_ok(), "Failed to compile fact: {:?}", result.err());
    }

    #[test]
    fn test_compile_simple_rule() {
        let mut compiler = Compiler::new();
        let result = compiler.compile(
            r#"
            edge(1, 2).
            reach(X, Y) :- edge(X, Y).
        "#,
        );
        assert!(
            result.is_ok(),
            "Failed to compile simple rule: {:?}",
            result.err()
        );

        let plan = result.unwrap();
        assert!(!plan.sccs.is_empty(), "Expected at least one SCC");
    }

    #[test]
    fn compiler_records_each_program_query_at_its_compiled_rule_position() {
        let plan = Compiler::new()
            .compile(
                r#"
                    source(a).
                    ?- source(X).
                    ?- source(a).
                "#,
            )
            .expect("queries should compile");

        assert_eq!(plan.generated_query_rules.len(), 2);
        for (query_index, provenance) in plan.generated_query_rules.iter().enumerate() {
            assert_eq!(provenance.query_index, query_index);
            let rule = &plan.rules_by_scc[provenance.scc_index][provenance.rule_index];
            assert_eq!(rule.head, format!("__xlog_query_{query_index}"));
        }
    }

    #[test]
    fn compiler_does_not_infer_query_provenance_from_an_authored_prefix() {
        let mut program = parse_program(
            r#"
                source(a).
                authored(X) :- source(X).
            "#,
        )
        .expect("authored program should parse");
        program
            .rules
            .iter_mut()
            .find(|rule| rule.head.predicate == "authored")
            .expect("authored rule should exist")
            .head
            .predicate = "__xlog_query_authored".to_string();

        let plan = Compiler::new()
            .compile_program(&program)
            .expect("authored prefix should remain an ordinary rule");
        assert!(plan.generated_query_rules.is_empty());
    }

    #[test]
    fn compiler_rejects_an_authored_exact_generated_query_head_collision() {
        let mut program = parse_program(
            r#"
                source(a).
                authored(X) :- source(X).
                ?- source(X).
            "#,
        )
        .expect("collision witness should parse");
        program
            .rules
            .iter_mut()
            .find(|rule| rule.head.predicate == "authored")
            .expect("authored rule should exist")
            .head
            .predicate = "__xlog_query_0".to_string();

        let error = Compiler::new()
            .compile_program(&program)
            .expect_err("authored exact query-head collisions must fail compilation");
        assert!(
            error
                .to_string()
                .contains("authored relation __xlog_query_0 collides with generated query head"),
            "unexpected collision error: {error}"
        );
    }

    #[test]
    fn compiler_reports_bound_is_target_before_arithmetic_type_mismatch() {
        let error = Compiler::new()
            .compile(
                r#"
                val(10).
                bad(Z) :- val(Z), Z is Z + 1.
                ?- bad(Z).
                "#,
            )
            .expect_err("an already-bound is-expression target must be rejected");

        assert_eq!(
            error.to_string(),
            "Compilation error: Variable Z already bound; 'is' requires fresh variable"
        );
    }

    #[test]
    fn compiler_rejects_is_target_bound_by_an_earlier_is_expression() {
        let error = Compiler::new()
            .compile(
                r#"
                val(10).
                bad(Z) :- val(X), Z is X + X, Z is X + X.
                ?- bad(Z).
                "#,
            )
            .expect_err("a prior is-expression target must stay bound");

        assert_eq!(
            error.to_string(),
            "Compilation error: Variable Z already bound; 'is' requires fresh variable"
        );
    }

    #[test]
    fn compiler_accepts_fresh_chained_is_targets() {
        Compiler::new()
            .compile(
                r#"
                val(10).
                good(Z) :- val(X), Y is X + X, Z is Y + Y.
                ?- good(Z).
                "#,
            )
            .expect("distinct source-ordered is-expression targets must remain valid");
    }

    #[test]
    fn compiler_reports_unbound_is_input_before_a_later_duplicate_target() {
        let error = Compiler::new()
            .compile(
                r#"
                seed(1).
                bad(Z) :- seed(X), Z is U + 1, Z is 1 + 1.
                ?- bad(Z).
                "#,
            )
            .expect_err("an unreachable later binding must not preempt the first is error");

        assert_eq!(
            error.to_string(),
            "Compilation error: Variable U used in arithmetic but not bound"
        );
    }

    #[test]
    fn test_compile_transitive_closure() {
        let mut compiler = Compiler::new();
        let result = compiler.compile(
            r#"
            edge(1, 2).
            edge(2, 3).
            edge(3, 4).
            reach(X, Y) :- edge(X, Y).
            reach(X, Z) :- reach(X, Y), edge(Y, Z).
        "#,
        );
        assert!(result.is_ok(), "Failed to compile TC: {:?}", result.err());

        let plan = result.unwrap();
        // Should have SCCs for edge and reach
        assert!(!plan.sccs.is_empty());
    }

    #[test]
    fn test_compile_with_negation() {
        let mut compiler = Compiler::new();
        let result = compiler.compile(
            r#"
            node(1).
            node(2).
            node(3).
            edge(1, 2).
            isolated(X) :- node(X), not edge(X, _).
        "#,
        );
        assert!(
            result.is_ok(),
            "Failed to compile with negation: {:?}",
            result.err()
        );
    }

    #[test]
    fn test_compile_with_comparison() {
        let mut compiler = Compiler::new();
        let result = compiler.compile(
            r#"
            value(1).
            value(5).
            value(10).
            value(15).
            small(X) :- value(X), X < 10.
        "#,
        );
        assert!(
            result.is_ok(),
            "Failed to compile with comparison: {:?}",
            result.err()
        );
    }

    #[test]
    fn test_schema_infers_from_rule_body_types() {
        let mut compiler = Compiler::new();
        let result = compiler.compile(
            r#"
            path(X, Y) :- reach(X, Y).
            edge(1, 2).
            edge(2, 3).
            reach(X, Y) :- edge(X, Y).
        "#,
        );
        assert!(
            result.is_ok(),
            "Failed to compile rule for schema inference: {:?}",
            result.err()
        );

        let schema = compiler
            .schemas()
            .get("reach")
            .expect("missing reach schema");
        assert_eq!(
            schema.column_type(0),
            Some(ScalarType::U32),
            "reach column 0 should match edge column type"
        );
        assert_eq!(
            schema.column_type(1),
            Some(ScalarType::U32),
            "reach column 1 should match edge column type"
        );

        let schema = compiler.schemas().get("path").expect("missing path schema");
        assert_eq!(
            schema.column_type(0),
            Some(ScalarType::U32),
            "path column 0 should inherit the transitive body type"
        );
        assert_eq!(
            schema.column_type(1),
            Some(ScalarType::U32),
            "path column 1 should inherit the transitive body type"
        );
    }

    #[test]
    fn test_schema_propagates_known_columns_when_a_sibling_column_is_unresolved() {
        let mut compiler = Compiler::new();
        let result = compiler.compile(
            r#"
            path(X) :- intermediate(X, Z).
            intermediate(X, Z) :- source(X), unresolved(Z).
            unresolved(Z) :- unresolved(Z).
            source(value).
        "#,
        );
        assert!(
            result.is_ok(),
            "Failed to compile partial rule-head schema inference: {:?}",
            result.err()
        );

        let schema = compiler.schemas().get("path").expect("missing path schema");
        assert_eq!(
            schema.column_type(0),
            Some(ScalarType::Symbol),
            "path should inherit the known intermediate column independently of its sibling"
        );
    }

    #[test]
    fn test_schema_rejects_conflicting_rule_head_evidence_in_either_order() {
        let symbol_first = r#"
            pred symbols(symbol).
            pred integers(i64).
            symbols(value).
            integers(5000000000).
            mixed(X) :- symbols(X).
            mixed(X) :- integers(X).
        "#;
        let integer_first = r#"
            pred symbols(symbol).
            pred integers(i64).
            symbols(value).
            integers(5000000000).
            mixed(X) :- integers(X).
            mixed(X) :- symbols(X).
        "#;

        for source in [symbol_first, integer_first] {
            let error = Compiler::new()
                .compile(source)
                .expect_err("conflicting inferred rule-head types must be rejected");
            let message = error.to_string();
            assert!(message.contains("mixed"), "{message}");
            assert!(message.contains("column 1"), "{message}");
        }
    }

    #[test]
    fn test_schema_rejects_conflicting_body_evidence_in_either_order() {
        let first_body_order = r#"
            source_number(1).
            source_symbol(value).
            mixed(X) :- source_number(X), source_symbol(X).
            mixed(1) :- source_number(1).
        "#;
        let reversed_body_order = r#"
            source_number(1).
            source_symbol(value).
            mixed(X) :- source_symbol(X), source_number(X).
            mixed(1) :- source_number(1).
        "#;

        for source in [first_body_order, reversed_body_order] {
            let error = Compiler::new()
                .compile(source)
                .expect_err("conflicting body-column types must be rejected");
            let message = error.to_string();
            assert!(message.contains("mixed"), "{message}");
            assert!(message.contains("variable X"), "{message}");
        }
    }

    #[test]
    fn test_schema_infers_arithmetic_binding_result_type() {
        let mut compiler = Compiler::new();
        compiler
            .compile(
                r#"
                computed(X) :- X is cast(1, u64).
            "#,
            )
            .expect("an arithmetic binding should determine its head-column type");

        let schema = compiler
            .schemas()
            .get("computed")
            .expect("missing computed schema");
        assert_eq!(schema.column_type(0), Some(ScalarType::U64));
    }

    #[test]
    fn test_schema_rejects_arithmetic_and_atom_evidence_in_either_rule_order() {
        let arithmetic_first = r#"
            pred integers(i64).
            integers(5000000000).
            mixed(X) :- X is cast(1, u64).
            mixed(X) :- integers(X).
        "#;
        let atom_first = r#"
            pred integers(i64).
            integers(5000000000).
            mixed(X) :- integers(X).
            mixed(X) :- X is cast(1, u64).
        "#;

        for source in [arithmetic_first, atom_first] {
            let error = Compiler::new()
                .compile(source)
                .expect_err("conflicting arithmetic and atom types must be rejected");
            let message = error.to_string();
            assert!(message.contains("mixed"), "{message}");
            assert!(message.contains("column 1"), "{message}");
        }
    }

    #[test]
    fn test_schema_rejects_incompatible_constant_heads_before_runtime() {
        let sources = [
            r#"
                pred p(u32).
                seed().
                p(value) :- seed().
            "#,
            r#"
                p(1).
                p(value).
            "#,
            r#"
                p(value).
                p(1).
            "#,
            r#"
                p(1).
                p(5000000000).
            "#,
            r#"
                p(5000000000).
                p(1).
            "#,
        ];

        for source in sources {
            let error = Compiler::new()
                .compile(source)
                .expect_err("incompatible constant head types must fail during compilation");
            let message = error.to_string();
            assert!(message.contains("p"), "{message}");
            assert!(message.contains("head term at position 0"), "{message}");
        }
    }

    #[test]
    fn test_declared_scalar_schemas_accept_supported_literal_conversions() {
        Compiler::new()
            .compile(
                r#"
                pred signed(i64).
                pred wide(u64).
                pred real(f64).
                pred flag(bool).
                seed().
                signed(1).
                wide(1).
                real(1).
                real(2) :- seed().
                flag(true).
                flag(false) :- seed().
            "#,
            )
            .expect("representable literals should use declared scalar storage types");
    }

    #[test]
    fn test_compile_unstratifiable_fails() {
        let mut compiler = Compiler::new();
        let result = compiler.compile(
            r#"
            p :- not q.
            q :- not p.
        "#,
        );
        assert!(result.is_err(), "Should fail with stratification cycle");
    }

    #[test]
    fn test_compile_syntax_error_fails() {
        let mut compiler = Compiler::new();
        let result = compiler.compile("edge(1, 2"); // Missing closing paren and period
        assert!(result.is_err(), "Should fail with syntax error");
    }

    #[test]
    fn test_compile_convenience_function() {
        let result = compile("edge(1, 2).");
        assert!(
            result.is_ok(),
            "Convenience compile failed: {:?}",
            result.err()
        );
    }

    #[test]
    fn test_compiler_reset() {
        let mut compiler = Compiler::new();

        // First compilation
        let result1 = compiler.compile("edge(1, 2).");
        assert!(result1.is_ok());

        // Reset and compile again
        compiler.reset();
        let result2 = compiler.compile("node(1). node(2).");
        assert!(result2.is_ok());
    }

    #[test]
    fn test_compile_with_pred_decl() {
        let mut compiler = Compiler::new();
        let result = compiler.compile(
            r#"
            pred edge(u32, u32).
            edge(1, 2).
            edge(2, 3).
            reach(X, Y) :- edge(X, Y).
        "#,
        );
        assert!(
            result.is_ok(),
            "Failed to compile with pred decl: {:?}",
            result.err()
        );
    }

    #[test]
    fn test_compile_multi_stratum() {
        let mut compiler = Compiler::new();
        let result = compiler.compile(
            r#"
            // Base facts
            edge(1, 2).
            edge(2, 3).
            edge(3, 1).

            // Stratum 0: edge (base)
            // Stratum 1: reach (depends on edge, recursive)
            reach(X, Y) :- edge(X, Y).
            reach(X, Z) :- reach(X, Y), edge(Y, Z).

            // Stratum 2: non_reach (negates reach)
            all_pairs(X, Y) :- edge(X, Z), edge(Y, W).
            non_reach(X, Y) :- all_pairs(X, Y), not reach(X, Y).
        "#,
        );
        assert!(
            result.is_ok(),
            "Failed to compile multi-stratum: {:?}",
            result.err()
        );

        let plan = result.unwrap();
        // Should have multiple strata
        assert!(!plan.strata.is_empty(), "Expected multiple strata");
    }

    #[test]
    fn test_compile_aggregation() {
        let mut compiler = Compiler::new();
        let result = compiler.compile(
            r#"
            edge(1, 2).
            edge(1, 3).
            edge(2, 3).
            out_degree(X, count(Y)) :- edge(X, Y).
        "#,
        );
        assert!(
            result.is_ok(),
            "Failed to compile with aggregation: {:?}",
            result.err()
        );

        let plan = result.unwrap();
        let out_degree_rules: Vec<_> = plan
            .rules_by_scc
            .iter()
            .flatten()
            .filter(|r| r.head == "out_degree")
            .collect();
        assert_eq!(out_degree_rules.len(), 1, "Expected one out_degree rule");

        // Aggregation lowering should produce a GroupBy node (wrapped in a Project to match head order).
        let body = &out_degree_rules[0].body;
        match body {
            RirNode::Project { input, .. } => {
                assert!(
                    matches!(input.as_ref(), RirNode::GroupBy { .. }),
                    "Expected Project(GroupBy(..)), got {:?}",
                    input
                );
            }
            other => panic!("Expected Project(GroupBy(..)), got {:?}", other),
        }
    }

    #[test]
    fn test_aggregate_schemas_match_runtime_result_types() {
        let mut compiler = Compiler::new();
        compiler
            .compile(
                r#"
                pred edge(u32, u64).
                pred score(u32, f64).
                pred counted(u32, u64).
                pred summed(u32, u64).
                pred minimum(u32, u64).
                pred maximum(u32, u64).
                pred combined(u32, f64).
                edge(1, 5000000000).
                score(1, 2.5).
                counted(X, count(Y)) :- edge(X, Y).
                summed(X, sum(Y)) :- edge(X, Y).
                minimum(X, min(Y)) :- edge(X, Y).
                maximum(X, max(Y)) :- edge(X, Y).
                combined(X, logsumexp(Y)) :- score(X, Y).
            "#,
            )
            .expect("aggregate declarations should match provider result schemas");

        for predicate in ["counted", "summed", "minimum", "maximum"] {
            assert_eq!(
                compiler
                    .schemas()
                    .get(predicate)
                    .expect("missing aggregate schema")
                    .column_type(1),
                Some(ScalarType::U64),
                "unexpected result type for {predicate}"
            );
        }
        assert_eq!(
            compiler
                .schemas()
                .get("combined")
                .expect("missing log-sum-exp schema")
                .column_type(1),
            Some(ScalarType::F64)
        );
    }

    #[test]
    fn test_undeclared_count_schema_is_u64() {
        let mut compiler = Compiler::new();
        compiler
            .compile(
                r#"
                edge(1, 5000000000).
                degree(X, count(Y)) :- edge(X, Y).
            "#,
            )
            .expect("undeclared count result should compile with its runtime type");
        assert_eq!(
            compiler
                .schemas()
                .get("degree")
                .expect("missing inferred count schema")
                .column_type(1),
            Some(ScalarType::U64)
        );
    }

    #[test]
    fn test_unsupported_aggregate_inputs_are_rejected_before_runtime() {
        let fixtures = [
            r#"
                pred source(u32, i32).
                pred result(u32, u64).
                source(1, 2).
                result(X, sum(Y)) :- source(X, Y).
            "#,
            r#"
                pred source(u32, u32).
                pred result(u32, f64).
                source(1, 2).
                result(X, logsumexp(Y)) :- source(X, Y).
            "#,
        ];

        for source in fixtures {
            let error = Compiler::new()
                .compile(source)
                .expect_err("unsupported aggregate input must fail during compilation");
            let message = error.to_string();
            assert!(message.contains("Unsupported aggregate input"), "{message}");
            assert!(message.contains("execution provider requires"), "{message}");
        }
    }

    #[test]
    fn test_compile_with_stats_snapshot() {
        let mut compiler = Compiler::new();
        let source = r#"
            edge(1, 2).
            edge(2, 3).
            reach(X, Y) :- edge(X, Y).
        "#;

        let _ = compiler.compile(source).expect("Initial compile failed");
        let edge_id = *compiler.rel_ids().get("edge").expect("edge rel_id missing");

        let mut mgr = StatsManager::new();
        mgr.register_relation(edge_id);
        mgr.update_cardinality(edge_id, 42);
        let snapshot = mgr.snapshot();

        let plan = compiler
            .compile_with_stats_snapshot(source, Some(&snapshot))
            .expect("Compile with snapshot failed");
        assert!(!plan.sccs.is_empty());
    }

    #[test]
    fn test_compile_with_named_stats_snapshot_reorders_joins() {
        let mut compiler = Compiler::new();
        let source = r#"
            foo(1).
            edge(1).
            out(X) :- edge(X), foo(X).
        "#;

        // Snapshot uses different RelIds than the compiler will assign for this program.
        // Map: RelId(0) -> edge (small), RelId(1) -> foo (big)
        let mut edge_stats = RelationStats::new(RelId(0));
        edge_stats.update_cardinality(10);
        let mut foo_stats = RelationStats::new(RelId(1));
        foo_stats.update_cardinality(10_000);

        let snapshot = StatsSnapshot {
            relations: vec![edge_stats, foo_stats],
            join_selectivities: Vec::new(),
            rel_names: vec![
                (RelId(0), "edge".to_string()),
                (RelId(1), "foo".to_string()),
            ],
        };

        let plan = compiler
            .compile_with_stats_snapshot(source, Some(&snapshot))
            .expect("Compile with named snapshot failed");

        let foo_id = *compiler.rel_ids().get("foo").expect("foo rel_id missing");
        let edge_id = *compiler.rel_ids().get("edge").expect("edge rel_id missing");

        let out_rule = plan
            .rules_by_scc
            .iter()
            .flatten()
            .find(|r| r.head == "out")
            .expect("out rule missing");

        // Peel projections to reach the join.
        let mut node = &out_rule.body;
        while let RirNode::Project { input, .. } = node {
            node = input;
        }

        match node {
            RirNode::ChainJoin {
                left,
                right,
                fallback,
                ..
            } => {
                // ChainJoin promotion wraps eligible two-atom joins after
                // stats-aware ordering. The chain node and its captured
                // fallback must agree on the build-side choice.
                assert!(matches!(**left, RirNode::Scan { rel } if rel == foo_id));
                assert!(matches!(**right, RirNode::Scan { rel } if rel == edge_id));

                let mut fallback_node = fallback.as_ref();
                while let RirNode::Project { input, .. } = fallback_node {
                    fallback_node = input;
                }
                match fallback_node {
                    RirNode::Join { left, right, .. } => {
                        assert!(matches!(**left, RirNode::Scan { rel } if rel == foo_id));
                        assert!(matches!(**right, RirNode::Scan { rel } if rel == edge_id));
                    }
                    other => panic!("Expected ChainJoin fallback Join node, got {:?}", other),
                }
            }
            RirNode::Join { left, right, .. } => {
                // Prefer building on the smaller relation (right/build side).
                assert!(matches!(**left, RirNode::Scan { rel } if rel == foo_id));
                assert!(matches!(**right, RirNode::Scan { rel } if rel == edge_id));
            }
            other => panic!("Expected Join node, got {:?}", other),
        }
    }

    fn helper_split_source() -> &'static str {
        r#"
            ab(0, 0). bc(0, 0). cd(0, 0). de(0, 0). ef(0, 0). af(0, 0).
            out(A, B, C, D, F) :-
                ab(A, B),
                bc(B, C),
                cd(C, D),
                de(D, E),
                ef(E, F),
                af(A, F).
        "#
    }

    fn helper_split_snapshot(distinct_d: u64) -> StatsSnapshot {
        let mut snapshot_relations = Vec::new();
        for (idx, name) in ["ab", "bc", "cd", "de", "ef", "af"].iter().enumerate() {
            let mut rel_stats = RelationStats::new(RelId(idx as u32));
            rel_stats.update_cardinality(8192);
            if *name == "de" {
                let mut d_col = ColumnStats::new(0, ScalarType::U32);
                d_col.update_distinct(distinct_d);
                rel_stats.add_column(d_col);
            }
            snapshot_relations.push(rel_stats);
        }
        StatsSnapshot {
            relations: snapshot_relations,
            join_selectivities: Vec::new(),
            rel_names: ["ab", "bc", "cd", "de", "ef", "af"]
                .iter()
                .enumerate()
                .map(|(idx, name)| (RelId(idx as u32), (*name).to_string()))
                .collect(),
        }
    }

    #[test]
    fn test_compile_with_named_stats_snapshot_creates_helper_relation() {
        let mut compiler = Compiler::new();
        let snapshot = helper_split_snapshot(1);
        let plan = compiler
            .compile_with_stats_snapshot(helper_split_source(), Some(&snapshot))
            .expect("compile with helper stats");
        let helper = compiler
            .rel_ids()
            .iter()
            .find_map(|(name, rel)| {
                name.starts_with("__kclique_helper_")
                    .then_some((name.clone(), *rel))
            })
            .expect("helper relation allocated");

        let helper_rule_count = plan
            .rules_by_scc
            .iter()
            .flatten()
            .filter(|rule| rule.head == helper.0)
            .count();
        assert_eq!(helper_rule_count, 1);

        let helper_rule = plan
            .rules_by_scc
            .iter()
            .flatten()
            .find(|rule| rule.head == helper.0)
            .expect("helper rule");
        assert!(
            matches!(helper_rule.body, RirNode::ChainJoin { .. }),
            "helper split output should be eligible for ChainJoin promotion"
        );

        let out_rule = plan
            .rules_by_scc
            .iter()
            .flatten()
            .find(|rule| rule.head == "out")
            .expect("out rule");
        assert!(contains_scan(&out_rule.body, helper.1));
    }

    #[test]
    fn test_compile_with_flat_named_stats_keeps_original_rule() {
        let mut compiler = Compiler::new();
        let snapshot = helper_split_snapshot(8192);
        let plan = compiler
            .compile_with_stats_snapshot(helper_split_source(), Some(&snapshot))
            .expect("compile with flat stats");

        assert!(!compiler
            .rel_ids()
            .keys()
            .any(|name| name.starts_with("__kclique_helper_")));
        let out_rules = plan
            .rules_by_scc
            .iter()
            .flatten()
            .filter(|rule| rule.head == "out")
            .count();
        assert_eq!(out_rules, 1);
    }

    fn contains_scan(node: &RirNode, rel: RelId) -> bool {
        match node {
            RirNode::Scan { rel: scan_rel } => *scan_rel == rel,
            RirNode::Join { left, right, .. } | RirNode::ChainJoin { left, right, .. } => {
                contains_scan(left, rel) || contains_scan(right, rel)
            }
            RirNode::Project { input, .. }
            | RirNode::Filter { input, .. }
            | RirNode::Distinct { input, .. }
            | RirNode::GroupBy { input, .. } => contains_scan(input, rel),
            RirNode::Union { inputs } => inputs.iter().any(|input| contains_scan(input, rel)),
            RirNode::Diff { left, right } => contains_scan(left, rel) || contains_scan(right, rel),
            RirNode::Fixpoint {
                base, recursive, ..
            } => contains_scan(base, rel) || contains_scan(recursive, rel),
            RirNode::MultiWayJoin { inputs, .. } => {
                inputs.iter().any(|input| contains_scan(input, rel))
            }
            RirNode::TensorMaskedJoin { rel_index, .. } => {
                rel_index.iter().any(|(input_rel, _)| *input_rel == rel)
            }
            RirNode::Unit => false,
        }
    }
}
