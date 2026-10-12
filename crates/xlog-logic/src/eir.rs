//! Frontend construction for the Epistemic Intermediate Representation.

use xlog_core::{Result, XlogError};
use xlog_ir::{
    EirAtom, EirBodyLiteral, EirConstraint, EirEpistemicLiteral, EirEpistemicMode, EirEpistemicOp,
    EirProgram, EirRule, EirTerm,
};

use crate::ast::{
    AggOp, Atom, BodyLiteral, Constraint as AstConstraint, EpistemicMode, EpistemicOp, Program,
    Rule as AstRule, Term,
};

/// Build EIR from parsed frontend AST without lowering to RIR.
pub fn build_eir(program: &Program) -> Result<EirProgram> {
    build_eir_core(program, &mut None)
}

/// Build the same EIR while acquiring the original owner before each output
/// allocation. The caller keeps its scope alive through the EIR's lifetime.
pub fn build_eir_with_allocation_owner(
    program: &Program,
    scope: &mut crate::compile::CompilationAllocationScope<'_>,
) -> Result<EirProgram> {
    build_eir_core(program, &mut Some(scope))
}

fn acquire(
    scope: &mut Option<&mut crate::compile::CompilationAllocationScope<'_>>,
    bytes: usize,
) -> Result<()> {
    if let Some(scope) = scope.as_deref_mut() {
        scope.acquire(u64::try_from(bytes).map_err(|_| {
            XlogError::Compilation("epistemic representation allocation overflow".into())
        })?)?;
    }
    Ok(())
}

fn acquire_vector<T>(
    scope: &mut Option<&mut crate::compile::CompilationAllocationScope<'_>>,
    count: usize,
) -> Result<()> {
    if let Some(scope) = scope.as_deref_mut() {
        scope.acquire(crate::compile::push_vector_allocation_bytes::<T>(count)?)?;
    }
    Ok(())
}

fn build_eir_core(
    program: &Program,
    scope: &mut Option<&mut crate::compile::CompilationAllocationScope<'_>>,
) -> Result<EirProgram> {
    acquire_vector::<EirRule>(scope, program.rules.len())?;
    acquire_vector::<EirConstraint>(scope, program.constraints.len())?;
    Ok(EirProgram {
        mode: convert_mode(program.directives.epistemic_mode_or_default()),
        rules: program
            .rules
            .iter()
            .map(|rule| convert_rule(rule, scope))
            .collect::<Result<_>>()?,
        constraints: program
            .constraints
            .iter()
            .map(|constraint| convert_constraint(constraint, scope))
            .collect::<Result<_>>()?,
    })
}

fn convert_constraint(
    constraint: &AstConstraint,
    scope: &mut Option<&mut crate::compile::CompilationAllocationScope<'_>>,
) -> Result<EirConstraint> {
    acquire_vector::<EirBodyLiteral>(scope, constraint.body.len())?;
    Ok(EirConstraint {
        authored_index: constraint.authored_index,
        body: constraint
            .body
            .iter()
            .map(|literal| convert_body_literal(literal, scope))
            .collect::<Result<_>>()?,
    })
}

fn convert_rule(
    rule: &AstRule,
    scope: &mut Option<&mut crate::compile::CompilationAllocationScope<'_>>,
) -> Result<EirRule> {
    acquire_vector::<EirBodyLiteral>(scope, rule.body.len())?;
    Ok(EirRule {
        head: convert_atom(&rule.head, scope)?,
        body: rule
            .body
            .iter()
            .map(|literal| convert_body_literal(literal, scope))
            .collect::<Result<_>>()?,
    })
}

fn convert_body_literal(
    lit: &BodyLiteral,
    scope: &mut Option<&mut crate::compile::CompilationAllocationScope<'_>>,
) -> Result<EirBodyLiteral> {
    Ok(match lit {
        BodyLiteral::Positive(atom) => EirBodyLiteral::Relational {
            negated: false,
            atom: convert_atom(atom, scope)?,
        },
        BodyLiteral::Negated(atom) => EirBodyLiteral::Relational {
            negated: true,
            atom: convert_atom(atom, scope)?,
        },
        BodyLiteral::Epistemic(lit) => EirBodyLiteral::Epistemic(EirEpistemicLiteral {
            op: convert_op(lit.op),
            negated: lit.negated,
            atom: convert_atom(&lit.atom, scope)?,
        }),
        BodyLiteral::Comparison(_) => EirBodyLiteral::Constraint,
        BodyLiteral::IsExpr(_) => EirBodyLiteral::Binding,
        BodyLiteral::Univ(_) => EirBodyLiteral::Binding,
    })
}

fn convert_atom(
    atom: &Atom,
    scope: &mut Option<&mut crate::compile::CompilationAllocationScope<'_>>,
) -> Result<EirAtom> {
    acquire(scope, atom.predicate.len())?;
    acquire_vector::<EirTerm>(scope, atom.terms.len())?;
    Ok(EirAtom {
        predicate: atom.predicate.clone(),
        arity: atom.arity(),
        terms: atom
            .terms
            .iter()
            .map(|term| convert_term_core(term, scope))
            .collect::<Result<_>>()?,
    })
}

pub(crate) fn convert_term(term: &Term) -> EirTerm {
    convert_term_core(term, &mut None).expect("unbounded EIR conversion has no owner refusal")
}

pub(crate) fn convert_term_with_allocation_owner(
    term: &Term,
    scope: &mut crate::compile::CompilationAllocationScope<'_>,
) -> Result<EirTerm> {
    convert_term_core(term, &mut Some(scope))
}

/// Heap geometry of an existing EIR term before the original term clone.
pub fn term_clone_allocation_bytes(term: &EirTerm) -> Result<u64> {
    let overflow = || XlogError::Compilation("epistemic term clone allocation overflow".into());
    let text = |text: &str| u64::try_from(text.len()).map_err(|_| overflow());
    let terms = |terms: &[EirTerm]| -> Result<u64> {
        let mut bytes = u64::try_from(
            terms
                .len()
                .checked_mul(std::mem::size_of::<EirTerm>())
                .ok_or_else(overflow)?,
        )
        .map_err(|_| overflow())?;
        for term in terms {
            bytes = bytes
                .checked_add(term_clone_allocation_bytes(term)?)
                .ok_or_else(overflow)?;
        }
        Ok(bytes)
    };
    match term {
        EirTerm::Variable(value) | EirTerm::String(value) | EirTerm::PredRef(value) => text(value),
        EirTerm::List(items) => terms(items),
        EirTerm::Cons { head, tail } => (2 * std::mem::size_of::<EirTerm>() as u64)
            .checked_add(term_clone_allocation_bytes(head)?)
            .and_then(|bytes| bytes.checked_add(term_clone_allocation_bytes(tail).ok()?))
            .ok_or_else(overflow),
        EirTerm::Compound { functor, args } => text(functor)?
            .checked_add(terms(args)?)
            .ok_or_else(overflow),
        EirTerm::Aggregate { op, variable } => {
            text(op)?.checked_add(text(variable)?).ok_or_else(overflow)
        }
        EirTerm::Anonymous | EirTerm::Integer(_) | EirTerm::FloatBits(_) | EirTerm::Symbol(_) => {
            Ok(0)
        }
    }
}

fn convert_term_core(
    term: &Term,
    scope: &mut Option<&mut crate::compile::CompilationAllocationScope<'_>>,
) -> Result<EirTerm> {
    Ok(match term {
        Term::Variable(name) => {
            acquire(scope, name.len())?;
            EirTerm::Variable(name.clone())
        }
        Term::Anonymous => EirTerm::Anonymous,
        Term::Integer(value) => EirTerm::Integer(*value),
        Term::Float(value) => EirTerm::FloatBits(value.to_bits()),
        Term::String(value) => {
            acquire(scope, value.len())?;
            EirTerm::String(value.clone())
        }
        Term::Symbol(id) => EirTerm::Symbol(*id),
        Term::List(items) => {
            acquire_vector::<EirTerm>(scope, items.len())?;
            EirTerm::List(
                items
                    .iter()
                    .map(|term| convert_term_core(term, scope))
                    .collect::<Result<_>>()?,
            )
        }
        Term::Cons { head, tail } => {
            acquire_vector::<EirTerm>(scope, 2)?;
            EirTerm::Cons {
                head: Box::new(convert_term_core(head, scope)?),
                tail: Box::new(convert_term_core(tail, scope)?),
            }
        }
        Term::Compound { functor, args } => {
            acquire(scope, functor.len())?;
            acquire_vector::<EirTerm>(scope, args.len())?;
            EirTerm::Compound {
                functor: functor.clone(),
                args: args
                    .iter()
                    .map(|term| convert_term_core(term, scope))
                    .collect::<Result<_>>()?,
            }
        }
        Term::PredRef(name) => {
            acquire(scope, name.len())?;
            EirTerm::PredRef(name.clone())
        }
        Term::Aggregate(agg) => {
            acquire(scope, convert_agg_op(agg.op).len())?;
            acquire(scope, agg.variable.len())?;
            EirTerm::Aggregate {
                op: convert_agg_op(agg.op).to_string(),
                variable: agg.variable.clone(),
            }
        }
    })
}

fn convert_agg_op(op: AggOp) -> &'static str {
    match op {
        AggOp::Count => "count",
        AggOp::Sum => "sum",
        AggOp::Min => "min",
        AggOp::Max => "max",
        AggOp::LogSumExp => "logsumexp",
    }
}

fn convert_mode(mode: EpistemicMode) -> EirEpistemicMode {
    match mode {
        EpistemicMode::G91 => EirEpistemicMode::G91,
        EpistemicMode::Faeel => EirEpistemicMode::Faeel,
    }
}

fn convert_op(op: EpistemicOp) -> EirEpistemicOp {
    match op {
        EpistemicOp::Know => EirEpistemicOp::Know,
        EpistemicOp::Possible => EirEpistemicOp::Possible,
    }
}
