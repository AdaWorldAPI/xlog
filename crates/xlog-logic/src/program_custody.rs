//! Physical allocation custody for the original compiler-owned AST clone.

use std::sync::Arc;

use xlog_core::{CompilationAllocationScope, Result, XlogError};

use crate::ast::{
    AstAllocationSize, AstHeapUnitSink, Atom, BodyLiteral, FuncDef, NeuralPredDecl, Program, Rule,
    TypeRef, UseDecl,
};

type AllocationOwner = Arc<dyn Send + Sync>;

fn storage_bytes<T>(count: usize) -> Result<u64> {
    count
        .checked_mul(std::mem::size_of::<T>())
        .and_then(|bytes| u64::try_from(bytes).ok())
        .ok_or_else(|| XlogError::allocation_extent_overflow("AST custody storage"))
}

fn reserve_storage<T>(
    count: usize,
    scope: &mut CompilationAllocationScope<'_>,
) -> Result<Option<AllocationOwner>> {
    let bytes = storage_bytes::<T>(count)?;
    if bytes == 0 {
        Ok(None)
    } else {
        scope.reserve_allocation(bytes).map(Some)
    }
}

/// Receipts for independent physical allocations replaced together by a real
/// AST mutator. The receipt roster itself dies before its storage receipt.
#[derive(Default)]
pub(crate) struct PayloadCustody {
    receipts: Vec<AllocationOwner>,
    storage_owner: Option<AllocationOwner>,
}

impl PayloadCustody {
    fn prepare_receipts(
        &mut self,
        additional: usize,
        scope: &mut CompilationAllocationScope<'_>,
    ) -> Result<()> {
        let count = self
            .receipts
            .len()
            .checked_add(additional)
            .ok_or_else(|| XlogError::allocation_extent_overflow("AST custody receipt"))?;
        if count > self.receipts.capacity() {
            let capacity = count
                .max(4)
                .checked_next_power_of_two()
                .ok_or_else(|| XlogError::allocation_extent_overflow("AST custody receipt"))?;
            let owner = reserve_storage::<AllocationOwner>(capacity, scope)?;
            self.receipts.reserve_exact(capacity - self.receipts.len());
            self.storage_owner = owner;
        }
        Ok(())
    }

    /// Reserve one original physical allocation before its data is born.
    pub(crate) fn reserve_unit(
        &mut self,
        bytes: u64,
        scope: &mut CompilationAllocationScope<'_>,
    ) -> Result<()> {
        if bytes != 0 {
            self.prepare_receipts(1, scope)?;
            let receipt = scope.reserve_allocation(bytes)?;
            self.receipts.push(receipt);
        }
        Ok(())
    }

    /// Keep original custody live on roster-growth refusal. Cloning an Arc
    /// transfers no AST data and creates no additional payload allocation.
    pub(crate) fn retain_original(
        &mut self,
        receipt: &AllocationOwner,
        scope: &mut CompilationAllocationScope<'_>,
    ) -> Result<()> {
        self.prepare_receipts(1, scope)?;
        self.receipts.push(Arc::clone(receipt));
        Ok(())
    }

    /// Transfer already-acquired physical receipts without repricing them.
    /// Refusal leaves both original groups intact. Explicit append preserves
    /// the destination allocation; the emptied source buffer is then dropped.
    pub(crate) fn append_from(
        &mut self,
        other: &mut Self,
        scope: &mut CompilationAllocationScope<'_>,
    ) -> Result<()> {
        self.prepare_receipts(other.receipts.len(), scope)?;
        self.receipts.append(&mut other.receipts);
        drop(std::mem::take(&mut other.receipts));
        other.storage_owner = None;
        Ok(())
    }

    pub(crate) fn for_body_clone(
        body: &[BodyLiteral],
        scope: &mut CompilationAllocationScope<'_>,
    ) -> Result<Self> {
        Self::for_clone(ClonePayload::Body(body), scope)
    }

    pub(crate) fn for_atom_clone(
        atom: &Atom,
        scope: &mut CompilationAllocationScope<'_>,
    ) -> Result<Self> {
        Self::for_clone(ClonePayload::Atom(atom), scope)
    }

    fn for_clone(
        payload: ClonePayload<'_>,
        scope: &mut CompilationAllocationScope<'_>,
    ) -> Result<Self> {
        let mut count = CountHeapUnits(0);
        payload.walk(&mut AstAllocationSize::with_sink(&mut count))?;
        let storage_owner = reserve_storage::<AllocationOwner>(count.0, scope)?;
        let mut custody = Self {
            receipts: Vec::with_capacity(count.0),
            storage_owner,
        };
        let mut sink = ReserveHeapUnits {
            receipts: &mut custody.receipts,
            scope,
        };
        payload.walk(&mut AstAllocationSize::with_sink(&mut sink))?;
        Ok(custody)
    }
}

struct CountHeapUnits(usize);

impl AstHeapUnitSink for CountHeapUnits {
    fn heap_unit(&mut self, _: usize) -> Result<()> {
        self.0 = self
            .0
            .checked_add(1)
            .ok_or_else(AstAllocationSize::overflow)?;
        Ok(())
    }
}

struct ReserveHeapUnits<'a, 'pool> {
    receipts: &'a mut Vec<AllocationOwner>,
    scope: &'a mut CompilationAllocationScope<'pool>,
}

impl AstHeapUnitSink for ReserveHeapUnits<'_, '_> {
    fn heap_unit(&mut self, bytes: usize) -> Result<()> {
        let bytes = u64::try_from(bytes).map_err(|_| AstAllocationSize::overflow())?;
        let receipt = self.scope.reserve_allocation(bytes)?;
        self.receipts.push(receipt);
        Ok(())
    }
}

#[derive(Clone, Copy)]
enum ClonePayload<'a> {
    Text(&'a str),
    Type(&'a TypeRef),
    Atom(&'a Atom),
    Body(&'a [BodyLiteral]),
    Import(&'a UseDecl),
    Function(&'a FuncDef),
    Rule(&'a Rule),
    NeuralMetadata(&'a NeuralPredDecl),
}

impl ClonePayload<'_> {
    fn walk(self, size: &mut AstAllocationSize<'_>) -> Result<()> {
        match self {
            Self::Text(text) => size.text(text),
            Self::Type(typ) => size.type_ref(typ),
            Self::Atom(atom) => size.atom(atom),
            Self::Body(body) => size.body(body),
            Self::Import(import) => size.import(import),
            Self::Function(function) => size.function(function),
            Self::Rule(rule) => size.rule(rule),
            Self::NeuralMetadata(neural) => size.neural_metadata(neural),
        }
    }
}

/// Parallel payload custody for one actual AST vector. Payload sidecars die
/// before both the original backing receipt and their own metadata storage.
pub(crate) struct RetainedVectorCustody<C> {
    pub(crate) elements: Vec<C>,
    pub(crate) backing_owner: Option<AllocationOwner>,
    pub(crate) elements_storage_owner: Option<AllocationOwner>,
}

impl<C> Default for RetainedVectorCustody<C> {
    fn default() -> Self {
        Self {
            elements: Vec::new(),
            backing_owner: None,
            elements_storage_owner: None,
        }
    }
}

impl<C> RetainedVectorCustody<C> {
    fn for_clone<T>(items: &[T], scope: &mut CompilationAllocationScope<'_>) -> Result<Self> {
        let backing_owner = reserve_storage::<T>(items.len(), scope)?;
        let elements_storage_owner = reserve_storage::<C>(items.len(), scope)?;
        Ok(Self {
            elements: Vec::with_capacity(items.len()),
            backing_owner,
            elements_storage_owner,
        })
    }

    /// Prepare only the parallel sidecar buffer. The real mutator separately
    /// reserves and replaces the actual AST vector's backing owner around its
    /// original reserve/push, with the old owner live until that operation ends.
    pub(crate) fn prepare_elements(
        &mut self,
        additional: usize,
        scope: &mut CompilationAllocationScope<'_>,
    ) -> Result<()> {
        let count = self
            .elements
            .len()
            .checked_add(additional)
            .ok_or_else(|| XlogError::allocation_extent_overflow("AST custody element"))?;
        if count > self.elements.capacity() {
            let capacity = count
                .max(4)
                .checked_next_power_of_two()
                .ok_or_else(|| XlogError::allocation_extent_overflow("AST custody element"))?;
            let owner = reserve_storage::<C>(capacity, scope)?;
            self.elements.reserve_exact(capacity - self.elements.len());
            self.elements_storage_owner = owner;
        }
        Ok(())
    }
}

#[derive(Default)]
pub(crate) struct ColumnCustody {
    pub(crate) name: PayloadCustody,
    pub(crate) typ: PayloadCustody,
}

#[derive(Default)]
pub(crate) struct PredicateCustody {
    pub(crate) name: PayloadCustody,
    pub(crate) types: RetainedVectorCustody<PayloadCustody>,
    pub(crate) columns: RetainedVectorCustody<ColumnCustody>,
}

#[derive(Default)]
pub(crate) struct AnnotatedDisjunctionCustody {
    pub(crate) choices: RetainedVectorCustody<PayloadCustody>,
}

#[derive(Default)]
pub(crate) struct NeuralPredicateCustody {
    pub(crate) metadata: PayloadCustody,
    pub(crate) predicate: PayloadCustody,
}

#[derive(Default)]
pub(crate) struct LearnableRuleCustody {
    pub(crate) mask_name: PayloadCustody,
    pub(crate) head: PayloadCustody,
    pub(crate) body: PayloadCustody,
}

#[derive(Default)]
pub(crate) struct ProgramAllocationCustody {
    pub(crate) imports: RetainedVectorCustody<PayloadCustody>,
    pub(crate) functions: RetainedVectorCustody<PayloadCustody>,
    pub(crate) domains: RetainedVectorCustody<PayloadCustody>,
    pub(crate) predicates: RetainedVectorCustody<PredicateCustody>,
    pub(crate) rules: RetainedVectorCustody<PayloadCustody>,
    pub(crate) constraints: RetainedVectorCustody<PayloadCustody>,
    pub(crate) queries: RetainedVectorCustody<PayloadCustody>,
    pub(crate) prob_facts: RetainedVectorCustody<PayloadCustody>,
    pub(crate) annotated_disjunctions: RetainedVectorCustody<AnnotatedDisjunctionCustody>,
    pub(crate) evidence: RetainedVectorCustody<PayloadCustody>,
    pub(crate) prob_queries: RetainedVectorCustody<PayloadCustody>,
    pub(crate) neural_predicates: RetainedVectorCustody<NeuralPredicateCustody>,
    pub(crate) learnable_rules: RetainedVectorCustody<LearnableRuleCustody>,
}

pub(crate) struct RetainedProgram {
    pub(crate) value: Program,
    pub(crate) custody: ProgramAllocationCustody,
}

impl ProgramAllocationCustody {
    pub(crate) fn for_clone(
        program: &Program,
        scope: &mut CompilationAllocationScope<'_>,
    ) -> Result<Self> {
        let mut custody = Self::default();
        custody.imports = RetainedVectorCustody::for_clone(&program.imports, scope)?;
        for import in &program.imports {
            custody.imports.elements.push(PayloadCustody::for_clone(
                ClonePayload::Import(import),
                scope,
            )?);
        }
        custody.functions = RetainedVectorCustody::for_clone(&program.functions, scope)?;
        for function in &program.functions {
            custody.functions.elements.push(PayloadCustody::for_clone(
                ClonePayload::Function(function),
                scope,
            )?);
        }
        custody.domains = RetainedVectorCustody::for_clone(&program.domains, scope)?;
        for domain in &program.domains {
            custody.domains.elements.push(PayloadCustody::for_clone(
                ClonePayload::Text(&domain.name),
                scope,
            )?);
        }
        custody.predicates = RetainedVectorCustody::for_clone(&program.predicates, scope)?;
        for predicate in &program.predicates {
            let name = PayloadCustody::for_clone(ClonePayload::Text(&predicate.name), scope)?;
            let mut types = RetainedVectorCustody::for_clone(&predicate.types, scope)?;
            for typ in &predicate.types {
                types
                    .elements
                    .push(PayloadCustody::for_clone(ClonePayload::Type(typ), scope)?);
            }
            let mut columns = RetainedVectorCustody::for_clone(&predicate.columns, scope)?;
            for column in &predicate.columns {
                columns.elements.push(ColumnCustody {
                    name: match &column.name {
                        Some(name) => PayloadCustody::for_clone(ClonePayload::Text(name), scope)?,
                        None => PayloadCustody::default(),
                    },
                    typ: PayloadCustody::for_clone(ClonePayload::Type(&column.typ), scope)?,
                });
            }
            custody.predicates.elements.push(PredicateCustody {
                name,
                types,
                columns,
            });
        }
        custody.rules = RetainedVectorCustody::for_clone(&program.rules, scope)?;
        for rule in &program.rules {
            custody
                .rules
                .elements
                .push(PayloadCustody::for_clone(ClonePayload::Rule(rule), scope)?);
        }
        custody.constraints = RetainedVectorCustody::for_clone(&program.constraints, scope)?;
        for constraint in &program.constraints {
            custody
                .constraints
                .elements
                .push(PayloadCustody::for_body_clone(&constraint.body, scope)?);
        }
        custody.queries = RetainedVectorCustody::for_clone(&program.queries, scope)?;
        for query in &program.queries {
            custody
                .queries
                .elements
                .push(PayloadCustody::for_atom_clone(&query.atom, scope)?);
        }
        custody.prob_facts = RetainedVectorCustody::for_clone(&program.prob_facts, scope)?;
        for fact in &program.prob_facts {
            custody
                .prob_facts
                .elements
                .push(PayloadCustody::for_atom_clone(&fact.atom, scope)?);
        }
        custody.annotated_disjunctions =
            RetainedVectorCustody::for_clone(&program.annotated_disjunctions, scope)?;
        for disjunction in &program.annotated_disjunctions {
            let mut choices = RetainedVectorCustody::for_clone(&disjunction.choices, scope)?;
            for fact in &disjunction.choices {
                choices
                    .elements
                    .push(PayloadCustody::for_atom_clone(&fact.atom, scope)?);
            }
            custody
                .annotated_disjunctions
                .elements
                .push(AnnotatedDisjunctionCustody { choices });
        }
        custody.evidence = RetainedVectorCustody::for_clone(&program.evidence, scope)?;
        for evidence in &program.evidence {
            custody
                .evidence
                .elements
                .push(PayloadCustody::for_atom_clone(&evidence.atom, scope)?);
        }
        custody.prob_queries = RetainedVectorCustody::for_clone(&program.prob_queries, scope)?;
        for query in &program.prob_queries {
            custody
                .prob_queries
                .elements
                .push(PayloadCustody::for_atom_clone(&query.atom, scope)?);
        }
        custody.neural_predicates =
            RetainedVectorCustody::for_clone(&program.neural_predicates, scope)?;
        for neural in &program.neural_predicates {
            custody
                .neural_predicates
                .elements
                .push(NeuralPredicateCustody {
                    metadata: PayloadCustody::for_clone(
                        ClonePayload::NeuralMetadata(neural),
                        scope,
                    )?,
                    predicate: PayloadCustody::for_atom_clone(&neural.predicate, scope)?,
                });
        }
        custody.learnable_rules =
            RetainedVectorCustody::for_clone(&program.learnable_rules, scope)?;
        for rule in &program.learnable_rules {
            custody.learnable_rules.elements.push(LearnableRuleCustody {
                mask_name: PayloadCustody::for_clone(ClonePayload::Text(&rule.mask_name), scope)?,
                head: PayloadCustody::for_atom_clone(&rule.head, scope)?,
                body: PayloadCustody::for_body_clone(&rule.body, scope)?,
            });
        }
        Ok(custody)
    }
}
