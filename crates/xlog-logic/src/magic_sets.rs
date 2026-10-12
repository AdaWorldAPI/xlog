//! Magic-set rewriting for the deterministic language-completeness subset.

use std::collections::{BTreeSet, HashMap, HashSet};
use std::sync::Arc;

use xlog_core::{Result, XlogError};

use crate::ast::{Atom, BodyLiteral, MagicSetsMode, Program, Rule, Term};
use crate::program_custody::{
    PayloadCustody, ProgramAllocationCustody, RetainedProgram, RetainedVectorCustody,
};

struct RetainedMagicValue<T> {
    value: T,
    custody: PayloadCustody,
}

struct RetainedMagicRules {
    value: Vec<Rule>,
    custody: RetainedVectorCustody<PayloadCustody>,
}

pub(crate) struct RetainedMagicSetRewrite {
    pub(crate) program: RetainedProgram,
    pub(crate) report: MagicSetReport,
}

struct ScopedMagicCustody {
    _custody: ProgramAllocationCustody,
    _storage_owner: Arc<dyn Send + Sync>,
}

struct ScopedMagicRewrite {
    value: MagicSetRewrite,
    owner: Arc<ScopedMagicCustody>,
}

fn magic_storage_bytes<T>(count: usize) -> Result<u64> {
    count
        .checked_mul(std::mem::size_of::<T>())
        .and_then(|bytes| u64::try_from(bytes).ok())
        .ok_or_else(|| XlogError::allocation_extent_overflow("magic vector storage"))
}

struct MagicAllocations<'a, 'b> {
    scope: Option<&'a mut crate::compile::CompilationAllocationScope<'b>>,
    retained_input: bool,
}

impl MagicAllocations<'_, '_> {
    fn output_unit(&mut self, custody: &mut PayloadCustody, bytes: u64) -> Result<()> {
        if let Some(scope) = self.scope.as_deref_mut() {
            custody.reserve_unit(bytes, scope)?;
        }
        Ok(())
    }

    fn append_custody(&mut self, into: &mut PayloadCustody, from: &mut PayloadCustody) -> Result<()> {
        if let Some(scope) = self.scope.as_deref_mut() {
            into.append_from(from, scope)?;
        }
        Ok(())
    }

    fn clone_atom(&mut self, atom: &Atom) -> Result<RetainedMagicValue<Atom>> {
        let custody = match self.scope.as_deref_mut() {
            Some(scope) => PayloadCustody::for_atom_clone(atom, scope)?,
            None => PayloadCustody::default(),
        };
        Ok(RetainedMagicValue {
            value: atom.clone(),
            custody,
        })
    }

    fn clone_body(&mut self, body: &[BodyLiteral]) -> Result<RetainedMagicValue<Vec<BodyLiteral>>> {
        let custody = match self.scope.as_deref_mut() {
            Some(scope) => PayloadCustody::for_body_clone(body, scope)?,
            None => PayloadCustody::default(),
        };
        Ok(RetainedMagicValue {
            value: body.to_vec(),
            custody,
        })
    }

    fn term_custody(&mut self, term: &Term) -> Result<PayloadCustody> {
        match self.scope.as_deref_mut() {
            Some(scope) => PayloadCustody::for_term_clone(term, scope),
            None => Ok(PayloadCustody::default()),
        }
    }

    fn body_buffer(&mut self, count: usize) -> Result<RetainedMagicValue<Vec<BodyLiteral>>> {
        let mut custody = PayloadCustody::default();
        self.output_unit(&mut custody, magic_storage_bytes::<BodyLiteral>(count)?)?;
        Ok(RetainedMagicValue {
            value: Vec::with_capacity(count),
            custody,
        })
    }

    fn push_literal(
        &mut self,
        body: &mut RetainedMagicValue<Vec<BodyLiteral>>,
        literal: &BodyLiteral,
    ) -> Result<()> {
        let mut custody = match self.scope.as_deref_mut() {
            Some(scope) => PayloadCustody::for_literal_clone(literal, scope)?,
            None => PayloadCustody::default(),
        };
        self.append_custody(&mut body.custody, &mut custody)?;
        body.value.push(literal.clone());
        Ok(())
    }

    fn rules_buffer(&mut self, count: usize) -> Result<RetainedMagicRules> {
        let custody = match self.scope.as_deref_mut() {
            Some(scope) => RetainedVectorCustody::for_capacity::<Rule>(count, scope)?,
            None => RetainedVectorCustody::default(),
        };
        Ok(RetainedMagicRules {
            value: Vec::with_capacity(count),
            custody,
        })
    }

    fn push_rule(
        &mut self,
        out: &mut RetainedMagicRules,
        rule: RetainedMagicValue<Rule>,
    ) -> Result<()> {
        if let Some(scope) = self.scope.as_deref_mut() {
            out.custody.prepare_elements(1, scope)?;
        }
        if self.scope.is_some() && out.value.len() == out.value.capacity() {
            let count = out
                .value
                .len()
                .checked_add(1)
                .ok_or_else(|| XlogError::allocation_extent_overflow("magic rule vector"))?;
            let owner = match self.scope.as_deref_mut() {
                Some(scope) => Some(scope.reserve_allocation(magic_storage_bytes::<Rule>(count)?)?),
                None => None,
            };
            out.value.reserve_exact(1);
            out.custody.backing_owner = owner;
        }
        out.value.push(rule.value);
        if self.scope.is_some() {
            out.custody.elements.push(rule.custody);
        }
        Ok(())
    }

    fn retained_rule(
        &mut self,
        mut head: RetainedMagicValue<Atom>,
        mut body: RetainedMagicValue<Vec<BodyLiteral>>,
    ) -> Result<RetainedMagicValue<Rule>> {
        self.append_custody(&mut head.custody, &mut body.custody)?;
        Ok(RetainedMagicValue {
            value: Rule {
                head: head.value,
                body: body.value,
            },
            custody: head.custody,
        })
    }

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
    fn reserve_vector<T>(&mut self, vector: &mut Vec<T>, additional: usize) -> Result<()> {
        if self.scope.is_some() && vector.capacity() - vector.len() < additional {
            let count = vector
                .len()
                .checked_add(additional)
                .ok_or_else(|| XlogError::allocation_extent_overflow("magic vector"))?;
            self.vector::<T>(count)?;
            vector.reserve_exact(additional);
        }
        Ok(())
    }
    fn format(&mut self, args: std::fmt::Arguments<'_>) -> Result<()> {
        if self.scope.is_some() {
            self.acquire(crate::compile::formatted_text_allocation_bytes(args)?)?;
        }
        Ok(())
    }
    fn atom(&mut self, atom: &Atom) -> Result<()> {
        if self.scope.is_some() {
            self.acquire(atom.owned_allocation_bytes()?)?;
        }
        Ok(())
    }
    fn tree<K, V>(&mut self, length: usize) -> Result<()> {
        if self.scope.is_some() {
            self.acquire(crate::compile::btree_insert_allocation_bytes::<K, V>(
                length,
            )?)?;
        }
        Ok(())
    }
    fn hash<K, V>(&mut self, length: usize, capacity: usize) -> Result<()> {
        if self.scope.is_some() && length == capacity {
            self.acquire(
                crate::compile::hash_allocation_bytes::<K, V>(
                    length
                        .checked_add(1)
                        .ok_or_else(|| XlogError::allocation_extent_overflow("magic table"))?,
                )?
                .checked_mul(2)
                .ok_or_else(|| XlogError::allocation_extent_overflow("magic table"))?,
            )?;
        }
        Ok(())
    }
    fn variables(&mut self, atom: &Atom) -> Result<()> {
        if self.scope.is_some() {
            let mut extent = Some((0usize, 0u64));
            atom.visit_variables(&mut |name| {
                extent = extent
                    .and_then(|(n, b)| Some((n.checked_add(1)?, b.checked_add(name.len() as u64)?)))
            });
            let (count, text) =
                extent.ok_or_else(|| XlogError::allocation_extent_overflow("magic variable"))?;
            self.vector::<&str>(count)?;
            self.acquire(text)?;
            self.acquire(crate::compile::hash_allocation_bytes::<String, ()>(count)?)?;
        }
        Ok(())
    }
}

/// Status of a magic-set rewrite attempt.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MagicSetStatus {
    /// Rewriting was disabled by source configuration or there was no directive.
    Disabled,
    /// A supported bound recursive query was rewritten.
    Applied,
    /// `auto` mode found an unsafe or inapplicable program and left it unchanged.
    Declined,
}

/// Human- and test-readable metadata for a magic-set rewrite attempt.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MagicSetReport {
    /// Final status.
    pub status: MagicSetStatus,
    /// Generated magic predicate names.
    pub generated_predicates: Vec<String>,
    /// Adorned recursive predicates, formatted as `predicate/adornment`.
    pub adorned_predicates: Vec<String>,
    /// Reasons the rewrite declined.
    pub declined_reasons: Vec<String>,
}

/// Rewritten program plus its report.
#[derive(Debug, Clone)]
pub struct MagicSetRewrite {
    /// Program after rewriting, or the original program when disabled/declined.
    pub program: Program,
    /// Rewrite metadata.
    pub report: MagicSetReport,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
struct Adornment {
    pred: String,
    pattern: Vec<bool>,
}

#[derive(Debug, Clone)]
struct Seed {
    pred: String,
    pattern: Vec<bool>,
    terms: Vec<Term>,
}

struct AdornmentText<'a>(&'a [bool]);
impl std::fmt::Display for AdornmentText<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        for bound in self.0 {
            f.write_str(if *bound { "b" } else { "f" })?;
        }
        Ok(())
    }
}

/// Rewrite supported bound recursive queries with magic predicates.
pub fn rewrite_magic_sets(program: &Program) -> Result<MagicSetRewrite> {
    rewrite_magic_sets_owned(program.clone())
}

/// [`rewrite_magic_sets`] taking the program by value: when magic sets are
/// off (the default) the program is returned as is, without a clone.
pub fn rewrite_magic_sets_owned(program: Program) -> Result<MagicSetRewrite> {
    let mode = program.directives.magic_sets;
    if mode.is_none() || mode == Some(MagicSetsMode::Off) {
        return Ok(with_status(program, MagicSetStatus::Disabled));
    }
    let output = rewrite_enabled_with_owner(
        RetainedProgram {
            value: program,
            custody: ProgramAllocationCustody::default(),
        },
        &mut MagicAllocations {
            scope: None,
            retained_input: false,
        },
    )?;
    Ok(MagicSetRewrite {
        program: output.program.value,
        report: output.report,
    })
}

/// Rewrite through the original reachable-adornment closure with custody.
pub fn rewrite_magic_sets_owned_with_allocation_owner(
    program: Program,
    scope: &mut crate::compile::CompilationAllocationScope<'_>,
) -> Result<MagicSetRewrite> {
    let mode = program.directives.magic_sets;
    if mode.is_none() || mode == Some(MagicSetsMode::Off) {
        return Ok(with_status(program, MagicSetStatus::Disabled));
    }
    let output = rewrite_enabled_with_owner(
        RetainedProgram {
            value: program,
            custody: ProgramAllocationCustody::default(),
        },
        &mut MagicAllocations {
            scope: Some(scope),
            retained_input: false,
        },
    )?;
    if output.report.status != MagicSetStatus::Applied {
        return Ok(MagicSetRewrite {
            program: output.program.value,
            report: output.report,
        });
    }
    let bytes = std::mem::size_of::<ScopedMagicCustody>()
        .checked_add(2 * std::mem::size_of::<usize>())
        .and_then(|bytes| u64::try_from(bytes).ok())
        .ok_or_else(|| XlogError::allocation_extent_overflow("magic custody storage"))?;
    let storage_owner = scope.reserve_allocation(bytes)?;
    let output = ScopedMagicRewrite {
        value: MagicSetRewrite {
            program: output.program.value,
            report: output.report,
        },
        owner: Arc::new(ScopedMagicCustody {
            _custody: output.program.custody,
            _storage_owner: storage_owner,
        }),
    };
    scope.retain(Arc::clone(&output.owner) as Arc<dyn Send + Sync>)?;
    Ok(output.value)
}

/// Move the compiler-owned program and its original physical receipts through
/// the same reachable-adornment generator used by the public entry points.
pub(crate) fn rewrite_magic_sets_retained(
    program: RetainedProgram,
    scope: &mut crate::compile::CompilationAllocationScope<'_>,
) -> Result<RetainedMagicSetRewrite> {
    let mode = program.value.directives.magic_sets;
    if mode.is_none() || mode == Some(MagicSetsMode::Off) {
        return Ok(with_retained_status(program, MagicSetStatus::Disabled));
    }
    rewrite_enabled_with_owner(
        program,
        &mut MagicAllocations {
            scope: Some(scope),
            retained_input: true,
        },
    )
}

fn rewrite_enabled_with_owner(
    mut program: RetainedProgram,
    allocations: &mut MagicAllocations<'_, '_>,
) -> Result<RetainedMagicSetRewrite> {
    let mode = program.value.directives.magic_sets;
    let mode = mode.expect("checked above");

    let recursive = recursive_predicates(&program.value, allocations)?;
    let seeds = collect_query_seeds(&program.value, &recursive, allocations)?;
    if seeds.is_empty() {
        let reason = "no bound recursive query eligible for magic_sets";
        allocations.vector::<String>(1)?;
        allocations.acquire(reason.len() as u64)?;
        return decline_or_error(
            program,
            mode,
            vec![reason.to_string()],
            allocations,
        );
    }

    if program.value.is_probabilistic_profile() {
        let reason = "probabilistic profiles are handled outside deterministic magic_sets";
        allocations.vector::<String>(1)?;
        allocations.acquire(reason.len() as u64)?;
        return decline_or_error(
            program,
            mode,
            vec![reason.to_string()],
            allocations,
        );
    }

    let mut target_preds: BTreeSet<String> = BTreeSet::new();
    for seed in &seeds {
        allocations.acquire(seed.pred.len() as u64)?;
        allocations.tree::<String, ()>(target_preds.len())?;
        target_preds.insert(seed.pred.clone());
    }
    let unsafe_reasons =
        unsupported_target_reasons(&program.value, &target_preds, &recursive, allocations)?;
    if !unsafe_reasons.is_empty() {
        return decline_or_error(program, mode, unsafe_reasons, allocations);
    }

    let mut adornments = initial_adornments(&seeds, allocations)?;
    expand_adornments(&program.value, &target_preds, &mut adornments, allocations)?;

    let mut generated_predicates: BTreeSet<String> = BTreeSet::new();
    let mut adorned_predicates: BTreeSet<String> = BTreeSet::new();
    for adornment in &adornments {
        allocations.tree::<String, ()>(generated_predicates.len())?;
        allocations.tree::<String, ()>(adorned_predicates.len())?;
        allocations.format(format_args!(
            "__xlog_magic_{}_{}",
            adornment.pred,
            AdornmentText(&adornment.pattern)
        ))?;
        allocations.vector::<u8>(adornment.pattern.len())?;
        allocations.format(format_args!(
            "{}/{}",
            adornment.pred,
            AdornmentText(&adornment.pattern)
        ))?;
        generated_predicates.insert(magic_predicate(&adornment.pred, &adornment.pattern));
        adorned_predicates.insert(format!(
            "{}/{}",
            adornment.pred,
            adornment_key(&adornment.pattern)
        ));
    }

    allocations.vector::<String>(generated_predicates.len())?;
    allocations.vector::<String>(adorned_predicates.len())?;
    let report = MagicSetReport {
        status: MagicSetStatus::Applied,
        generated_predicates: generated_predicates.into_iter().collect(),
        adorned_predicates: adorned_predicates.into_iter().collect(),
        declined_reasons: Vec::new(),
    };
    rewrite_rules(&mut program, &target_preds, &adornments, &seeds, allocations)?;

    Ok(RetainedMagicSetRewrite {
        program,
        report,
    })
}

fn with_status(program: Program, status: MagicSetStatus) -> MagicSetRewrite {
    MagicSetRewrite {
        program,
        report: MagicSetReport {
            status,
            generated_predicates: Vec::new(),
            adorned_predicates: Vec::new(),
            declined_reasons: Vec::new(),
        },
    }
}

fn with_retained_status(program: RetainedProgram, status: MagicSetStatus) -> RetainedMagicSetRewrite {
    RetainedMagicSetRewrite {
        program,
        report: MagicSetReport {
            status,
            generated_predicates: Vec::new(),
            adorned_predicates: Vec::new(),
            declined_reasons: Vec::new(),
        },
    }
}

fn decline_or_error(
    program: RetainedProgram,
    mode: MagicSetsMode,
    reasons: Vec<String>,
    allocations: &mut MagicAllocations<'_, '_>,
) -> Result<RetainedMagicSetRewrite> {
    if mode == MagicSetsMode::On {
        return Err(magic_error(
            format_args!("{}", MagicReasons(&reasons)),
            allocations,
        ));
    }
    Ok(RetainedMagicSetRewrite {
        program,
        report: MagicSetReport {
            status: MagicSetStatus::Declined,
            generated_predicates: Vec::new(),
            adorned_predicates: Vec::new(),
            declined_reasons: reasons,
        },
    })
}

struct MagicReasons<'a>(&'a [String]);

impl std::fmt::Display for MagicReasons<'_> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        for (index, reason) in self.0.iter().enumerate() {
            if index != 0 {
                formatter.write_str("; ")?;
            }
            formatter.write_str(reason)?;
        }
        Ok(())
    }
}

fn magic_error(
    message: std::fmt::Arguments<'_>,
    allocations: &mut MagicAllocations<'_, '_>,
) -> XlogError {
    let bytes = xlog_core::error::formatted_text_allocation_bytes(format_args!(
        "magic_sets error: {message}"
    ));
    let producer = match allocations.scope.as_deref_mut() {
        Some(scope) => {
            let mut reserve = |bytes| scope.reserve_allocation(bytes);
            xlog_core::CompilationAllocationScope::prepare_diagnostic(bytes, Some(&mut reserve))
        }
        None => xlog_core::CompilationAllocationScope::prepare_diagnostic(bytes, None),
    };
    match producer {
        Ok(producer) => {
            producer.finish(|| XlogError::Compilation(format!("magic_sets error: {message}")))
        }
        Err(error) => error,
    }
}

fn collect_query_seeds(
    program: &Program,
    recursive: &HashSet<String>,
    allocations: &mut MagicAllocations<'_, '_>,
) -> Result<Vec<Seed>> {
    let mut seen = HashSet::new();
    let mut out = Vec::new();
    for query in &program.queries {
        if !recursive.contains(&query.atom.predicate) {
            continue;
        }
        allocations.vector::<bool>(query.atom.terms.len())?;
        let pattern: Vec<bool> = query.atom.terms.iter().map(is_seed_term).collect();
        if !pattern.iter().any(|bound| *bound) {
            continue;
        }
        if !query
            .atom
            .terms
            .iter()
            .zip(&pattern)
            .all(|(term, bound)| !*bound || is_supported_magic_term(term))
        {
            continue;
        }
        allocations.atom(&query.atom)?;
        let terms = bound_terms(&query.atom, &pattern);
        allocations.vector::<u8>(pattern.len())?;
        allocations.format(format_args!(
            "{}:{}:{:?}",
            query.atom.predicate,
            AdornmentText(&pattern),
            terms
        ))?;
        let key = format!(
            "{}:{}:{:?}",
            query.atom.predicate,
            adornment_key(&pattern),
            terms
        );
        allocations.hash::<String, ()>(seen.len(), seen.capacity())?;
        if seen.insert(key) {
            allocations.acquire(query.atom.predicate.len() as u64)?;
            allocations.reserve_vector(&mut out, 1)?;
            out.push(Seed {
                pred: query.atom.predicate.clone(),
                pattern,
                terms,
            });
        }
    }
    Ok(out)
}

fn initial_adornments(
    seeds: &[Seed],
    allocations: &mut MagicAllocations<'_, '_>,
) -> Result<BTreeSet<Adornment>> {
    let mut out = BTreeSet::new();
    for seed in seeds {
        allocations.tree::<Adornment, ()>(out.len())?;
        allocations.acquire(seed.pred.len() as u64)?;
        allocations.vector::<bool>(seed.pattern.len())?;
        out.insert(Adornment {
            pred: seed.pred.clone(),
            pattern: seed.pattern.clone(),
        });
    }
    Ok(out)
}

fn expand_adornments(
    program: &Program,
    target_preds: &BTreeSet<String>,
    adornments: &mut BTreeSet<Adornment>,
    allocations: &mut MagicAllocations<'_, '_>,
) -> Result<()> {
    let mut changed = true;
    while changed {
        changed = false;
        allocations.vector::<Adornment>(adornments.len())?;
        for adornment in adornments.iter() {
            allocations.acquire(adornment.pred.len() as u64)?;
            allocations.vector::<bool>(adornment.pattern.len())?;
        }
        let snapshot: Vec<Adornment> = adornments.iter().cloned().collect();
        for adornment in snapshot {
            for rule in program
                .rules
                .iter()
                .filter(|rule| rule.head.predicate == adornment.pred)
            {
                for discovered in
                    discover_body_adornments(rule, &adornment.pattern, target_preds, allocations)?
                {
                    if !adornments.contains(&discovered) {
                        allocations.tree::<Adornment, ()>(adornments.len())?;
                    }
                    changed |= adornments.insert(discovered);
                }
            }
        }
    }
    Ok(())
}

fn discover_body_adornments(
    rule: &Rule,
    head_pattern: &[bool],
    target_preds: &BTreeSet<String>,
    allocations: &mut MagicAllocations<'_, '_>,
) -> Result<Vec<Adornment>> {
    allocations.variables(&rule.head)?;
    let mut bound = head_bound_variables(&rule.head, head_pattern);
    let mut out = Vec::new();
    for lit in &rule.body {
        match lit {
            BodyLiteral::Positive(atom) => {
                if target_preds.contains(&atom.predicate) {
                    allocations.vector::<bool>(atom.terms.len())?;
                    let pattern = atom_adornment(atom, &bound);
                    if !pattern.iter().any(|is_bound| *is_bound) {
                        return Err(magic_error(
                            format_args!(
                                "recursive call {}/{} has no bound argument under supported SIPS",
                                atom.predicate,
                                atom.arity()
                            ),
                            allocations,
                        ));
                    }
                    allocations.acquire(atom.predicate.len() as u64)?;
                    allocations.reserve_vector(&mut out, 1)?;
                    out.push(Adornment {
                        pred: atom.predicate.clone(),
                        pattern,
                    });
                }
                bind_atom_variables(atom, &mut bound, allocations)?;
            }
            BodyLiteral::Comparison(_)
            | BodyLiteral::Epistemic(_)
            | BodyLiteral::IsExpr(_)
            | BodyLiteral::Negated(_)
            | BodyLiteral::Univ(_) => {}
        }
    }
    Ok(out)
}

fn rewrite_rules(
    program: &mut RetainedProgram,
    target_preds: &BTreeSet<String>,
    adornments: &BTreeSet<Adornment>,
    seeds: &[Seed],
    allocations: &mut MagicAllocations<'_, '_>,
) -> Result<()> {
    let mut out = allocations.rules_buffer(program.value.rules.len())?;

    let mut emitted = HashSet::new();
    for seed in seeds {
        let head = retained_magic_atom(&seed.pred, &seed.pattern, &seed.terms, None, allocations)?;
        let body = RetainedMagicValue {
            value: Vec::new(),
            custody: PayloadCustody::default(),
        };
        let rule = allocations.retained_rule(head, body)?;
        push_unique_rule(&mut out, &mut emitted, rule, allocations)?;
    }

    for adornment in adornments {
        for rule in program
            .value
            .rules
            .iter()
            .filter(|rule| rule.head.predicate == adornment.pred)
        {
            let generated =
                propagation_rules(rule, &adornment.pattern, target_preds, allocations)?;
            let generated_custody = generated.custody;
            let payloads = generated_custody.elements.into_iter();
            let values = generated.value.into_iter();
            if allocations.scope.is_some() {
                for (value, custody) in values.zip(payloads) {
                    push_unique_rule(
                        &mut out,
                        &mut emitted,
                        RetainedMagicValue { value, custody },
                        allocations,
                    )?;
                }
            } else {
                for value in values {
                    push_unique_rule(
                        &mut out,
                        &mut emitted,
                        RetainedMagicValue {
                            value,
                            custody: PayloadCustody::default(),
                        },
                        allocations,
                    )?;
                }
            }
        }
    }

    for adornment in adornments {
        for rule in program
            .value
            .rules
            .iter()
            .filter(|rule| rule.head.predicate == adornment.pred)
        {
            let mut magic = retained_magic_atom_for(&rule.head, &adornment.pattern, allocations)?;
            let count = rule
                .body
                .len()
                .checked_add(1)
                .ok_or_else(|| XlogError::allocation_extent_overflow("magic rule body"))?;
            let mut body = allocations.body_buffer(count)?;
            allocations.append_custody(&mut body.custody, &mut magic.custody)?;
            body.value.push(BodyLiteral::Positive(magic.value));
            for literal in &rule.body {
                allocations.push_literal(&mut body, literal)?;
            }
            let head = allocations.clone_atom(&rule.head)?;
            let rewritten = allocations.retained_rule(head, body)?;
            allocations.push_rule(&mut out, rewritten)?;
        }
    }

    let original_custody = std::mem::take(&mut program.custody.rules);
    let mut payloads = original_custody.elements.into_iter();
    let values = std::mem::take(&mut program.value.rules).into_iter();
    let mut moved = 0;
    for value in values {
        let custody = if allocations.retained_input {
            payloads
                .next()
                .expect("compiler-owned rule has original allocation custody")
        } else {
            PayloadCustody::default()
        };
        let rule = RetainedMagicValue { value, custody };
        if !target_preds.contains(&rule.value.head.predicate) {
            allocations.push_rule(&mut out, rule)?;
            moved += 1;
        }
    }
    // The original generator emits unchanged rules first. Appending them only
    // after the borrowed generation phase lets their original payloads move;
    // rotating both real buffers restores that exact order without allocation.
    out.value.rotate_right(moved);
    if allocations.scope.is_some() {
        out.custody.elements.rotate_right(moved);
    }
    program.value.rules = out.value;
    program.custody.rules = out.custody;
    Ok(())
}

fn propagation_rules(
    rule: &Rule,
    head_pattern: &[bool],
    target_preds: &BTreeSet<String>,
    allocations: &mut MagicAllocations<'_, '_>,
) -> Result<RetainedMagicRules> {
    let mut caller_magic = retained_magic_atom_for(&rule.head, head_pattern, allocations)?;
    let count = rule
        .body
        .len()
        .checked_add(1)
        .ok_or_else(|| XlogError::allocation_extent_overflow("magic prefix body"))?;
    let mut prefix = allocations.body_buffer(count)?;
    allocations.append_custody(&mut prefix.custody, &mut caller_magic.custody)?;
    prefix.value.push(BodyLiteral::Positive(caller_magic.value));
    allocations.variables(&rule.head)?;
    let mut bound = head_bound_variables(&rule.head, head_pattern);
    let mut out = allocations.rules_buffer(0)?;

    for lit in &rule.body {
        let BodyLiteral::Positive(atom) = lit else {
            continue;
        };
        if target_preds.contains(&atom.predicate) {
            allocations.vector::<bool>(atom.terms.len())?;
            let pattern = atom_adornment(atom, &bound);
            if !pattern.iter().any(|is_bound| *is_bound) {
                return Err(magic_error(
                    format_args!(
                        "recursive call {}/{} has no bound argument under supported SIPS",
                        atom.predicate,
                        atom.arity()
                    ),
                    allocations,
                ));
            }
            let head = retained_magic_atom_for(atom, &pattern, allocations)?;
            let is_trivial = prefix.value.len() == 1
                && matches!(&prefix.value[0], BodyLiteral::Positive(prefix_atom) if *prefix_atom == head.value);
            if !is_trivial {
                let body = allocations.clone_body(&prefix.value)?;
                let generated = allocations.retained_rule(head, body)?;
                allocations.push_rule(&mut out, generated)?;
            }
        }
        bind_atom_variables(atom, &mut bound, allocations)?;
        allocations.push_literal(&mut prefix, lit)?;
    }

    Ok(out)
}

fn unsupported_target_reasons(
    program: &Program,
    target_preds: &BTreeSet<String>,
    recursive: &HashSet<String>,
    allocations: &mut MagicAllocations<'_, '_>,
) -> Result<Vec<String>> {
    let mut reasons = BTreeSet::new();
    for rule in &program.rules {
        if !target_preds.contains(&rule.head.predicate) {
            continue;
        }
        if allocations.scope.is_some() {
            let count = rule
                .body
                .len()
                .checked_mul(2)
                .ok_or_else(|| XlogError::allocation_extent_overflow("magic reason"))?
                .checked_add(2)
                .ok_or_else(|| XlogError::allocation_extent_overflow("magic reason"))?;
            let mut bytes = 0u64;
            let mut longest = rule.head.predicate.len();
            for literal in &rule.body {
                if let BodyLiteral::Positive(atom) = literal {
                    longest = longest.max(atom.predicate.len());
                }
            }
            // The original loop emits at most two head reasons and two per
            // reached positive literal. The fixed diagnostic prefix is the longest
            // spelling in this same routine, not a source-size tariff.
            let text=crate::compile::formatted_text_allocation_bytes(format_args!(
                "non-positive literal in recursive rule for {} is outside the supported magic_sets subset",rule.head.predicate))?;
            bytes = bytes
                .checked_add(
                    (longest - rule.head.predicate.len())
                        .try_into()
                        .ok()
                        .and_then(|length: u64| length.checked_mul(6))
                        .and_then(|extra| text.checked_add(extra))
                        .and_then(|n| n.checked_mul(count as u64))
                        .ok_or_else(|| XlogError::allocation_extent_overflow("magic reason"))?,
                )
                .ok_or_else(|| XlogError::allocation_extent_overflow("magic reason"))?;
            for offset in 0..count {
                bytes = bytes
                    .checked_add(crate::compile::btree_insert_allocation_bytes::<String, ()>(
                        reasons.len() + offset,
                    )?)
                    .ok_or_else(|| XlogError::allocation_extent_overflow("magic reason"))?;
            }
            allocations.acquire(bytes)?;
        }
        if rule.has_negation() {
            reasons.insert(format!(
                "negation in recursive rule for {} is outside the supported magic_sets subset",
                rule.head.predicate
            ));
        }
        if rule.has_aggregation() || rule.body.iter().any(body_literal_has_aggregate) {
            reasons.insert(format!(
                "aggregation in recursive rule for {} is outside the supported magic_sets subset",
                rule.head.predicate
            ));
        }
        for lit in &rule.body {
            match lit {
                BodyLiteral::Positive(atom) => {
                    if recursive.contains(&atom.predicate) && atom.predicate != rule.head.predicate
                    {
                        reasons.insert(format!(
                            "mutual recursion through {} is outside the supported magic_sets subset",
                            atom.predicate
                        ));
                    }
                    if atom.predicate.starts_with("__xlog_meta_")
                        || atom.predicate.starts_with("__xlog_list_")
                    {
                        reasons.insert(format!(
                            "meta/list helper {} in recursive rule is outside the supported magic_sets subset",
                            atom.predicate
                        ));
                    }
                }
                BodyLiteral::Negated(_) => {}
                BodyLiteral::Comparison(_)
                | BodyLiteral::Epistemic(_)
                | BodyLiteral::IsExpr(_)
                | BodyLiteral::Univ(_) => {
                    reasons.insert(format!(
                        "non-positive literal in recursive rule for {} is outside the supported magic_sets subset",
                        rule.head.predicate
                    ));
                }
            }
        }
    }
    allocations.vector::<String>(reasons.len())?;
    Ok(reasons.into_iter().collect())
}

fn recursive_predicates(
    program: &Program,
    allocations: &mut MagicAllocations<'_, '_>,
) -> Result<HashSet<String>> {
    let mut deps: HashMap<String, HashSet<String>> = HashMap::new();
    for rule in &program.rules {
        allocations.acquire(rule.head.predicate.len() as u64)?;
        if !deps.contains_key(&rule.head.predicate) {
            allocations.hash::<String, HashSet<String>>(deps.len(), deps.capacity())?;
        }
        allocations.vector::<&str>(rule.body.len())?;
        let entry = deps.entry(rule.head.predicate.clone()).or_default();
        for pred in rule.body_predicates() {
            allocations.acquire(pred.len() as u64)?;
            if !entry.contains(pred) {
                allocations.hash::<String, ()>(entry.len(), entry.capacity())?;
            }
            entry.insert(pred.to_string());
        }
    }
    if allocations.scope.is_some() {
        let count = deps
            .values()
            .try_fold(deps.len(), |n, preds| n.checked_add(preds.len()))
            .ok_or_else(|| XlogError::allocation_extent_overflow("magic dependency"))?;
        let mut bytes = crate::compile::hash_allocation_bytes::<String, ()>(count)?
            .checked_mul(2)
            .ok_or_else(|| XlogError::allocation_extent_overflow("magic dependency"))?;
        for (name, preds) in &deps {
            bytes = bytes
                .checked_add(name.len() as u64)
                .ok_or_else(|| XlogError::allocation_extent_overflow("magic dependency"))?;
            for pred in preds {
                bytes = bytes
                    .checked_add(pred.len() as u64)
                    .ok_or_else(|| XlogError::allocation_extent_overflow("magic dependency"))?;
            }
        }
        // The reachability search has one original seen set at a time; the
        // result set coexists with it. Both use this actual dependency roster.
        allocations.acquire(
            bytes
                .checked_mul(2)
                .ok_or_else(|| XlogError::allocation_extent_overflow("magic dependency"))?,
        )?;
    }
    Ok(deps
        .keys()
        .filter(|pred| reaches(pred, pred, &deps, &mut HashSet::new()))
        .cloned()
        .collect())
}

fn reaches(
    start: &str,
    target: &str,
    deps: &HashMap<String, HashSet<String>>,
    seen: &mut HashSet<String>,
) -> bool {
    let Some(next) = deps.get(start) else {
        return false;
    };
    for pred in next {
        if pred == target {
            return true;
        }
        if seen.insert(pred.clone()) && reaches(pred, target, deps, seen) {
            return true;
        }
    }
    false
}

fn head_bound_variables(atom: &Atom, pattern: &[bool]) -> HashSet<String> {
    atom.terms
        .iter()
        .zip(pattern)
        .filter(|(_, bound)| **bound)
        .flat_map(|(term, _)| term.variables().into_iter().map(str::to_string))
        .collect()
}

fn atom_adornment(atom: &Atom, bound: &HashSet<String>) -> Vec<bool> {
    atom.terms
        .iter()
        .map(|term| term_is_bound(term, bound))
        .collect()
}

fn term_is_bound(term: &Term, bound: &HashSet<String>) -> bool {
    match term {
        Term::Variable(name) => bound.contains(name),
        Term::Anonymous => false,
        Term::List(items) => items.iter().all(|item| term_is_bound(item, bound)),
        Term::Cons { head, tail } => term_is_bound(head, bound) && term_is_bound(tail, bound),
        Term::Compound { args, .. } => args.iter().all(|arg| term_is_bound(arg, bound)),
        Term::Integer(_)
        | Term::Float(_)
        | Term::String(_)
        | Term::Symbol(_)
        | Term::PredRef(_) => true,
        Term::Aggregate(_) => false,
    }
}

fn bind_atom_variables(
    atom: &Atom,
    bound: &mut HashSet<String>,
    allocations: &mut MagicAllocations<'_, '_>,
) -> Result<()> {
    allocations.variables(atom)?;
    for name in atom.variables() {
        if !bound.contains(name) {
            allocations.hash::<String, ()>(bound.len(), bound.capacity())?;
        }
        bound.insert(name.to_string());
    }
    Ok(())
}

fn body_literal_has_aggregate(lit: &BodyLiteral) -> bool {
    match lit {
        BodyLiteral::Positive(atom) | BodyLiteral::Negated(atom) => atom_has_aggregate(atom),
        BodyLiteral::Epistemic(lit) => atom_has_aggregate(&lit.atom),
        BodyLiteral::Comparison(comparison) => {
            term_has_aggregate(&comparison.left) || term_has_aggregate(&comparison.right)
        }
        BodyLiteral::IsExpr(_) => false,
        BodyLiteral::Univ(univ) => {
            term_has_aggregate(&univ.term) || term_has_aggregate(&univ.parts)
        }
    }
}

fn atom_has_aggregate(atom: &Atom) -> bool {
    atom.terms.iter().any(term_has_aggregate)
}

fn term_has_aggregate(term: &Term) -> bool {
    match term {
        Term::Aggregate(_) => true,
        Term::List(items) => items.iter().any(term_has_aggregate),
        Term::Cons { head, tail } => term_has_aggregate(head) || term_has_aggregate(tail),
        Term::Compound { args, .. } => args.iter().any(term_has_aggregate),
        Term::Variable(_)
        | Term::Anonymous
        | Term::Integer(_)
        | Term::Float(_)
        | Term::String(_)
        | Term::Symbol(_)
        | Term::PredRef(_) => false,
    }
}

fn is_seed_term(term: &Term) -> bool {
    is_supported_magic_term(term) && !term.is_any_variable()
}

fn is_supported_magic_term(term: &Term) -> bool {
    matches!(
        term,
        Term::Integer(_) | Term::Float(_) | Term::String(_) | Term::Symbol(_)
    )
}

fn bound_terms(atom: &Atom, pattern: &[bool]) -> Vec<Term> {
    atom.terms
        .iter()
        .zip(pattern)
        .filter(|(_, bound)| **bound)
        .map(|(term, _)| term.clone())
        .collect()
}

fn retained_magic_atom_for(
    atom: &Atom,
    pattern: &[bool],
    allocations: &mut MagicAllocations<'_, '_>,
) -> Result<RetainedMagicValue<Atom>> {
    retained_magic_atom(
        &atom.predicate,
        pattern,
        &atom.terms,
        Some(pattern),
        allocations,
    )
}

fn retained_magic_atom(
    predicate: &str,
    pattern: &[bool],
    terms: &[Term],
    selected: Option<&[bool]>,
    allocations: &mut MagicAllocations<'_, '_>,
) -> Result<RetainedMagicValue<Atom>> {
    let count = match selected {
        Some(selected) => terms
            .iter()
            .zip(selected)
            .filter(|(_, bound)| **bound)
            .count(),
        None => terms.len(),
    };
    let mut custody = PayloadCustody::default();
    allocations.output_unit(&mut custody, magic_storage_bytes::<Term>(count)?)?;
    let mut out = RetainedMagicValue {
        value: Vec::with_capacity(count),
        custody,
    };
    for (index, term) in terms.iter().enumerate() {
        if selected.is_some_and(|selected| !selected.get(index).copied().unwrap_or(false)) {
            continue;
        }
        let mut custody = allocations.term_custody(term)?;
        allocations.append_custody(&mut out.custody, &mut custody)?;
        out.value.push(term.clone());
    }
    allocations.output_unit(
        &mut out.custody,
        crate::compile::formatted_text_allocation_bytes(format_args!(
            "__xlog_magic_{}_{}",
            predicate,
            AdornmentText(pattern)
        ))?,
    )?;
    Ok(RetainedMagicValue {
        value: Atom {
            predicate: magic_predicate(predicate, pattern),
            terms: out.value,
        },
        custody: out.custody,
    })
}

fn magic_predicate(pred: &str, pattern: &[bool]) -> String {
    format!("__xlog_magic_{}_{}", pred, AdornmentText(pattern))
}

fn adornment_key(pattern: &[bool]) -> String {
    pattern
        .iter()
        .map(|bound| if *bound { 'b' } else { 'f' })
        .collect()
}

fn push_unique_rule(
    out: &mut RetainedMagicRules,
    emitted: &mut HashSet<String>,
    rule: RetainedMagicValue<Rule>,
    allocations: &mut MagicAllocations<'_, '_>,
) -> Result<()> {
    allocations.format(format_args!("{:?}", rule.value))?;
    let key = format!("{:?}", rule.value);
    allocations.hash::<String, ()>(emitted.len(), emitted.capacity())?;
    if emitted.insert(key) {
        allocations.push_rule(out, rule)?;
    }
    Ok(())
}
