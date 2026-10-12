//! Device-resident execution for bounded ordinary Datalog plans.
//!
//! The resident route is deliberately fail-closed.  Plan inspection records
//! every physical route occurrence before setup allocates workspace or
//! enqueues CUDA work.  Unsupported shapes remain on the existing executor.

use std::collections::{BTreeSet, HashMap};
use std::fmt::{self, Write};
use std::sync::Arc;

#[cfg(all(test, feature = "resident-graph-tests"))]
use std::cell::Cell;

use xlog_core::{CompilationAllocationScope, RelId, Result, ScalarType, Schema, XlogError};
use xlog_ir::{ConstValue, ExecutionPlan, Expr, JoinType, ProjectExpr, RirNode};

const RESIDENT_GRAPH_MAX_INTERMEDIATE_ARITY: usize = 17;

#[derive(Clone, Default)]
struct ResidentAllocationOwner(Option<Arc<dyn Send + Sync>>);

impl fmt::Debug for ResidentAllocationOwner {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(if self.0.is_some() {
            "retained"
        } else {
            "ordinary"
        })
    }
}

struct ResidentText {
    value: String,
    owner: Option<Arc<dyn Send + Sync>>,
}

impl fmt::Display for ResidentText {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.value)
    }
}

impl ResidentText {
    fn empty() -> Self {
        Self {
            value: String::new(),
            owner: None,
        }
    }

    fn append(
        &mut self,
        scope: &mut Option<&mut CompilationAllocationScope<'_>>,
        args: fmt::Arguments<'_>,
    ) -> Result<()> {
        if let Some(scope) = scope.as_deref_mut() {
            let extent = xlog_core::error::formatted_text_extent(args)?;
            let length =
                self.value
                    .len()
                    .checked_add(usize::try_from(extent).map_err(|_| {
                        XlogError::allocation_extent_overflow("resident descriptor")
                    })?)
                    .ok_or_else(|| XlogError::allocation_extent_overflow("resident descriptor"))?;
            if length > self.value.capacity() {
                let owner = scope
                    .reserve_allocation(xlog_core::push_vector_allocation_bytes::<u8>(length)?)?;
                self.value.reserve(length - self.value.len());
                self.owner = Some(owner);
            }
        }
        self.value
            .write_fmt(args)
            .expect("String formatter cannot fail");
        Ok(())
    }

    fn into_retained(
        self,
        scope: &mut Option<&mut CompilationAllocationScope<'_>>,
    ) -> Result<String> {
        if let (Some(scope), Some(owner)) = (scope.as_deref_mut(), self.owner.as_ref()) {
            scope.retain(Arc::clone(owner))?;
        }
        Ok(self.value)
    }
}

fn resident_text(
    scope: &mut Option<&mut CompilationAllocationScope<'_>>,
    args: fmt::Arguments<'_>,
) -> Result<ResidentText> {
    let mut text = ResidentText::empty();
    text.append(scope, args)?;
    Ok(text)
}

fn reserve_resident_vector<T>(
    scope: &mut Option<&mut CompilationAllocationScope<'_>>,
    count: usize,
) -> Result<Option<Arc<dyn Send + Sync>>> {
    scope
        .as_deref_mut()
        .map(|scope| scope.reserve_allocation(xlog_core::push_vector_allocation_bytes::<T>(count)?))
        .transpose()
}

/// Schemas keyed by the relation identities carried by compiled scan nodes.
#[derive(Debug, Clone, Default)]
pub struct ResidentGraphSchemaCatalog {
    by_relation: HashMap<RelId, ResidentSchemaAliases>,
    _table_owner: ResidentAllocationOwner,
    _allocation_owner: ResidentAllocationOwner,
}

#[derive(Debug, Clone)]
struct ResidentSchemaAlias {
    name: String,
    schema: Schema,
    _allocation_owner: ResidentAllocationOwner,
}

#[derive(Debug, Clone, Default)]
struct ResidentSchemaAliases {
    entries: Vec<ResidentSchemaAlias>,
    _storage_owner: ResidentAllocationOwner,
}

enum ResidentSchemaEntry<'a> {
    Owned(String, RelId, Schema),
    Borrowed(&'a str, RelId, &'a Schema),
}

impl ResidentGraphSchemaCatalog {
    /// Builds a catalog from compiler-assigned names, relation ids, and schemas.
    pub fn from_named_schemas(entries: impl IntoIterator<Item = (String, RelId, Schema)>) -> Self {
        Self::from_entries(
            entries
                .into_iter()
                .map(|(name, relation, schema)| ResidentSchemaEntry::Owned(name, relation, schema)),
            &mut None,
        )
        .expect("ordinary schema catalog has no allocation callback")
    }

    /// Copy the original compiler schema roster under its allocation authority.
    pub fn from_named_schema_references_with_allocation_owner<'a>(
        entries: impl IntoIterator<Item = (&'a str, RelId, &'a Schema)>,
        scope: &mut CompilationAllocationScope<'_>,
    ) -> Result<Self> {
        let mut owned = scope.child_scope();
        owned.prepare_owner()?;
        match Self::from_entries(
            entries.into_iter().map(|(name, relation, schema)| {
                ResidentSchemaEntry::Borrowed(name, relation, schema)
            }),
            &mut Some(&mut owned),
        ) {
            Ok(mut catalog) => {
                catalog._allocation_owner = ResidentAllocationOwner(Some(owned.into_owner()?));
                Ok(catalog)
            }
            Err(error) => Err(owned.into_failure(error)),
        }
    }

    fn from_entries<'a>(
        entries: impl IntoIterator<Item = ResidentSchemaEntry<'a>>,
        scope: &mut Option<&mut CompilationAllocationScope<'_>>,
    ) -> Result<Self> {
        let mut table_owner = ResidentAllocationOwner::default();
        let mut by_relation: HashMap<RelId, ResidentSchemaAliases> = HashMap::new();
        for entry in entries {
            let (alias, relation) = match entry {
                ResidentSchemaEntry::Owned(name, relation, schema) => (
                    ResidentSchemaAlias {
                        name,
                        schema,
                        _allocation_owner: ResidentAllocationOwner::default(),
                    },
                    relation,
                ),
                ResidentSchemaEntry::Borrowed(name, relation, schema) => {
                    let owner = scope
                        .as_deref_mut()
                        .map(|scope| {
                            let bytes = (name.len() as u64)
                                .checked_add(schema.owned_allocation_bytes()?)
                                .ok_or_else(|| {
                                    XlogError::allocation_extent_overflow("resident schema clone")
                                })?;
                            scope.reserve_allocation(bytes)
                        })
                        .transpose()?;
                    (
                        ResidentSchemaAlias {
                            name: name.to_owned(),
                            schema: schema.clone(),
                            _allocation_owner: ResidentAllocationOwner(owner),
                        },
                        relation,
                    )
                }
            };
            if !by_relation.contains_key(&relation) && by_relation.len() == by_relation.capacity() {
                let next_owner = scope
                    .as_deref_mut()
                    .map(|scope| {
                        let count = by_relation.len().checked_add(1).ok_or_else(|| {
                            XlogError::allocation_extent_overflow("resident schema catalog")
                        })?;
                        scope.reserve_allocation(xlog_core::hash_allocation_bytes::<
                            RelId,
                            ResidentSchemaAliases,
                        >(count)?)
                    })
                    .transpose()?;
                by_relation.reserve(1);
                table_owner = ResidentAllocationOwner(next_owner);
            }
            let aliases = by_relation.entry(relation).or_default();
            if aliases.entries.len() == aliases.entries.capacity() {
                let count = aliases.entries.len().checked_add(1).ok_or_else(|| {
                    XlogError::allocation_extent_overflow("resident schema aliases")
                })?;
                let next_owner = reserve_resident_vector::<ResidentSchemaAlias>(scope, count)?;
                aliases.entries.reserve(1);
                aliases._storage_owner = ResidentAllocationOwner(next_owner);
            }
            aliases.entries.push(alias);
        }
        for aliases in by_relation.values_mut() {
            let _sort_receipt =
                reserve_resident_vector::<ResidentSchemaAlias>(scope, aliases.entries.len())?;
            aliases
                .entries
                .sort_by(|left, right| left.name.cmp(&right.name));
            aliases
                .entries
                .dedup_by(|left, right| left.name == right.name && left.schema == right.schema);
        }
        Ok(Self {
            by_relation,
            _table_owner: table_owner,
            _allocation_owner: ResidentAllocationOwner::default(),
        })
    }

    fn clone_with_allocation_owner(
        &self,
        scope: &mut Option<&mut CompilationAllocationScope<'_>>,
    ) -> Result<Self> {
        match scope.as_deref_mut() {
            Some(scope) => Self::from_named_schema_references_with_allocation_owner(
                self.by_relation.iter().flat_map(|(&relation, aliases)| {
                    aliases
                        .entries
                        .iter()
                        .map(move |alias| (alias.name.as_str(), relation, &alias.schema))
                }),
                scope,
            ),
            None => Ok(self.clone()),
        }
    }

    fn descriptor(
        &self,
        relation: RelId,
        scope: &mut Option<&mut CompilationAllocationScope<'_>>,
    ) -> Result<Option<ResidentText>> {
        let Some(aliases) = self.by_relation.get(&relation) else {
            return Ok(None);
        };
        let _parts_receipt = reserve_resident_vector::<ResidentText>(scope, aliases.entries.len())?;
        let mut parts = Vec::with_capacity(aliases.entries.len());
        for alias in &aliases.entries {
            parts.push(resident_text(
                scope,
                format_args!("{}={:#?}", alias.name, alias.schema),
            )?);
        }
        let mut text = ResidentText::empty();
        for (index, part) in parts.iter().enumerate() {
            if index != 0 {
                text.append(scope, format_args!("|"))?;
            }
            text.append(scope, format_args!("{part}"))?;
        }
        Ok(Some(text))
    }

    pub(crate) fn schema(&self, relation: RelId) -> Option<&Schema> {
        let aliases = self.by_relation.get(&relation)?;
        let first = &aliases.entries.first()?.schema;
        aliases
            .entries
            .iter()
            .all(|alias| &alias.schema == first)
            .then_some(first)
    }
}

/// A reason why a complete plan cannot use the resident conditional graph.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ResidentGraphDeclineReason {
    /// A scan relation has no compiler schema identity.
    MissingScanSchema {
        /// Relation whose scan lacks a compiler schema identity.
        relation: RelId,
    },
    /// A physical node has no resident implementation.
    UnsupportedNode {
        /// Stable path to the unsupported node within the physical plan.
        path: String,
        /// Physical node kind that has no resident implementation.
        node: &'static str,
    },
    /// The resident route supports only inner and semi joins.
    UnsupportedJoin {
        /// Stable path to the unsupported join within the physical plan.
        path: String,
        /// Join semantics that the resident route cannot execute.
        join_type: JoinType,
    },
    /// The caller requested a complete relation store, which cannot be staged
    /// as a query-only transaction.
    FullStoreRequested,
    /// The compiled program uses epistemic or another non-ordinary semantics.
    NonOrdinaryPlan,
    /// A caller input is imported or belongs to a different memory manager, so
    /// its lifetime cannot be bound to this resident transaction.
    ImportedInputUnsupported {
        /// Imported relation whose ownership cannot join the transaction.
        relation: String,
    },
    /// A scanned source lacks a current provider-produced full-row set proof.
    SourceSetUncertified {
        /// Source relation without a current full-row set certificate.
        relation: String,
    },
    /// The CUDA driver cannot construct a conditional WHILE graph.
    ConditionalGraphUnavailable {
        /// Driver capability failure reported during conditional-graph setup.
        detail: String,
    },
    /// Setup cannot reserve a bounded workspace without exceeding the budget.
    WorkspaceUnbounded {
        /// Workspace bound or reservation failure reported during preflight.
        detail: String,
    },
}

/// Complete prelaunch proof of the physical routes selected for a plan.
#[derive(Debug, Clone)]
pub struct ResidentGraphRouteCertificate {
    covered_route_descriptors: BTreeSet<String>,
    declines: Vec<ResidentGraphDeclineReason>,
    schema_catalog: ResidentGraphSchemaCatalog,
    plan_fingerprint: u64,
    _decline_storage: ResidentAllocationOwner,
    _allocation_owner: ResidentAllocationOwner,
}

/// A route certificate sealed to the exact immutable plan allocation it inspected.
///
/// Preparation through this value cannot accidentally pair a certificate with a
/// different plan. The owned [`Arc`] also prevents safe mutation of the plan while
/// the seal remains live.
#[derive(Debug, Clone)]
pub struct ResidentGraphCertifiedPlan {
    plan: Arc<ExecutionPlan>,
    certificate: ResidentGraphRouteCertificate,
    _allocation_owner: ResidentAllocationOwner,
}

fn append_resident_expression_descriptor(
    descriptor: &mut ResidentText,
    expression: &Expr,
    scope: &mut Option<&mut CompilationAllocationScope<'_>>,
) -> Result<()> {
    fn grow_pending<'a>(
        pending: &mut Vec<&'a Expr>,
        extra: usize,
        scope: &mut Option<&mut CompilationAllocationScope<'_>>,
        receipt: &mut Option<Arc<dyn Send + Sync>>,
    ) -> Result<()> {
        let count = pending
            .len()
            .checked_add(extra)
            .ok_or_else(|| XlogError::allocation_extent_overflow("resident expression roster"))?;
        if count > pending.capacity() {
            let next = reserve_resident_vector::<&Expr>(scope, count)?;
            pending.reserve(extra);
            *receipt = next;
        }
        Ok(())
    }
    fn schedule_binary<'a>(
        pending: &mut Vec<&'a Expr>,
        left: &'a Expr,
        right: &'a Expr,
        scope: &mut Option<&mut CompilationAllocationScope<'_>>,
        receipt: &mut Option<Arc<dyn Send + Sync>>,
    ) -> Result<()> {
        grow_pending(pending, 2, scope, receipt)?;
        pending.push(right);
        pending.push(left);
        Ok(())
    }
    let mut pending_receipt = reserve_resident_vector::<&Expr>(scope, 1)?;
    let mut pending = vec![expression];
    while let Some(expression) = pending.pop() {
        match expression {
            Expr::Column(index) => descriptor.append(scope, format_args!("column:{index};"))?,
            Expr::Const(value) => {
                descriptor.append(scope, format_args!("{}", "constant:"))?;
                match value {
                    ConstValue::U32(value) => {
                        descriptor.append(scope, format_args!("u32:{value};"))?
                    }
                    ConstValue::U64(value) => {
                        descriptor.append(scope, format_args!("u64:{value};"))?
                    }
                    ConstValue::I32(value) => {
                        descriptor.append(scope, format_args!("i32:{value};"))?
                    }
                    ConstValue::I64(value) => {
                        descriptor.append(scope, format_args!("i64:{value};"))?
                    }
                    ConstValue::F32(value) => {
                        descriptor.append(scope, format_args!("f32:{:08x};", value.to_bits()))?
                    }
                    ConstValue::F64(value) => {
                        descriptor.append(scope, format_args!("f64:{:016x};", value.to_bits()))?
                    }
                    ConstValue::Bool(value) => {
                        descriptor.append(scope, format_args!("bool:{value};"))?
                    }
                    ConstValue::Symbol(value) => {
                        descriptor.append(scope, format_args!("symbol:{}:", value.len()))?;
                        descriptor.append(scope, format_args!("{}", value))?;
                        descriptor.append(scope, format_args!("{}", ";"))?;
                    }
                }
            }
            Expr::Compare { left, op, right } => {
                descriptor.append(scope, format_args!("compare:{op:?};"))?;
                schedule_binary(&mut pending, left, right, scope, &mut pending_receipt)?;
            }
            Expr::And(items) => {
                descriptor.append(scope, format_args!("and:{};", items.len()))?;
                grow_pending(&mut pending, items.len(), scope, &mut pending_receipt)?;
                pending.extend(items.iter().rev());
            }
            Expr::Or(items) => {
                descriptor.append(scope, format_args!("or:{};", items.len()))?;
                grow_pending(&mut pending, items.len(), scope, &mut pending_receipt)?;
                pending.extend(items.iter().rev());
            }
            Expr::Not(inner) => {
                descriptor.append(scope, format_args!("{}", "not;"))?;
                grow_pending(&mut pending, 1, scope, &mut pending_receipt)?;
                pending.push(inner);
            }
            Expr::Add(left, right) => {
                descriptor.append(scope, format_args!("{}", "add;"))?;
                schedule_binary(&mut pending, left, right, scope, &mut pending_receipt)?;
            }
            Expr::Sub(left, right) => {
                descriptor.append(scope, format_args!("{}", "subtract;"))?;
                schedule_binary(&mut pending, left, right, scope, &mut pending_receipt)?;
            }
            Expr::Mul(left, right) => {
                descriptor.append(scope, format_args!("{}", "multiply;"))?;
                schedule_binary(&mut pending, left, right, scope, &mut pending_receipt)?;
            }
            Expr::Div(left, right) => {
                descriptor.append(scope, format_args!("{}", "divide;"))?;
                schedule_binary(&mut pending, left, right, scope, &mut pending_receipt)?;
            }
            Expr::Mod(left, right) => {
                descriptor.append(scope, format_args!("{}", "modulo;"))?;
                schedule_binary(&mut pending, left, right, scope, &mut pending_receipt)?;
            }
            Expr::Abs(inner) => {
                descriptor.append(scope, format_args!("{}", "absolute_value;"))?;
                grow_pending(&mut pending, 1, scope, &mut pending_receipt)?;
                pending.push(inner);
            }
            Expr::Min(left, right) => {
                descriptor.append(scope, format_args!("{}", "minimum;"))?;
                schedule_binary(&mut pending, left, right, scope, &mut pending_receipt)?;
            }
            Expr::Max(left, right) => {
                descriptor.append(scope, format_args!("{}", "maximum;"))?;
                schedule_binary(&mut pending, left, right, scope, &mut pending_receipt)?;
            }
            Expr::Pow(left, right) => {
                descriptor.append(scope, format_args!("{}", "power;"))?;
                schedule_binary(&mut pending, left, right, scope, &mut pending_receipt)?;
            }
            Expr::Cast(inner, target) => {
                descriptor.append(scope, format_args!("cast:{target:?};"))?;
                grow_pending(&mut pending, 1, scope, &mut pending_receipt)?;
                pending.push(inner);
            }
            Expr::Conditional {
                condition,
                then_expr,
                else_expr,
            } => {
                descriptor.append(scope, format_args!("{}", "conditional;"))?;
                grow_pending(&mut pending, 3, scope, &mut pending_receipt)?;
                pending.push(else_expr);
                pending.push(then_expr);
                pending.push(condition);
            }
        }
    }
    Ok(())
}

fn resident_project_columns_descriptor(
    columns: &[ProjectExpr],
    scope: &mut Option<&mut CompilationAllocationScope<'_>>,
) -> Result<ResidentText> {
    let mut descriptor = resident_text(scope, format_args!("column_count={};", columns.len()))?;
    for column in columns {
        match column {
            ProjectExpr::Column(index) => {
                descriptor.append(scope, format_args!("column:{index};"))?
            }
            ProjectExpr::Computed(expression, scalar_type) => {
                descriptor.append(scope, format_args!("computed:{scalar_type:?};"))?;
                append_resident_expression_descriptor(&mut descriptor, expression, scope)?;
            }
        }
    }
    Ok(descriptor)
}

fn resident_node_descriptor(
    node: &RirNode,
    scope: &mut Option<&mut CompilationAllocationScope<'_>>,
) -> Result<ResidentText> {
    match node {
        RirNode::Unit => resident_text(scope, format_args!("unit")),
        RirNode::Scan { rel } => resident_text(scope, format_args!("scan;relation={}", rel.0)),
        RirNode::Filter { predicate, .. } => {
            let mut descriptor = resident_text(scope, format_args!("filter;predicate="))?;
            append_resident_expression_descriptor(&mut descriptor, predicate, scope)?;
            Ok(descriptor)
        }
        RirNode::Project { columns, .. } => {
            let columns = resident_project_columns_descriptor(columns, scope)?;
            resident_text(scope, format_args!("project;{columns}"))
        }
        RirNode::Join { left_keys, right_keys, join_type, .. } => resident_text(scope, format_args!(
            "join;left_keys={left_keys:?};right_keys={right_keys:?};join_type={join_type:?}"
        )),
        RirNode::ChainJoin { left_key, right_key, output_columns, .. } => {
            let output = resident_project_columns_descriptor(output_columns, scope)?;
            resident_text(scope, format_args!("chain_join;left_key={left_key};right_key={right_key};output_{output}"))
        }
        RirNode::GroupBy { key_cols, aggs, .. } => resident_text(scope, format_args!("group_by;key_cols={key_cols:?};aggregates={aggs:?}")),
        RirNode::Union { inputs } => resident_text(scope, format_args!("union;input_count={}", inputs.len())),
        RirNode::Distinct { key_cols, .. } => resident_text(scope, format_args!("distinct;key_cols={key_cols:?}")),
        RirNode::Diff { .. } => resident_text(scope, format_args!("difference")),
        RirNode::Fixpoint { scc_id, delta_rel, full_rel, .. } => resident_text(scope, format_args!(
            "fixpoint;scc_id={scc_id};delta_relation={};full_relation={}", delta_rel.0, full_rel.0
        )),
        RirNode::MultiWayJoin { inputs, slot_vars, output_columns, plan, var_order, .. } => {
            let output = resident_project_columns_descriptor(output_columns, scope)?;
            resident_text(scope, format_args!(
                "multi_way_join;input_count={};slot_variables={slot_vars:?};output_{output};plan={plan:?};variable_order={var_order:?}", inputs.len()
            ))
        }
        RirNode::TensorMaskedJoin { mask_name, schema_size, left_keys, right_keys, rel_index, head_rel_name, head_rel_id, max_active_rules, head_projection } => resident_text(scope, format_args!(
            "tensor_masked_join;mask={mask_name:?};schema_size={schema_size};left_keys={left_keys:?};right_keys={right_keys:?};relation_index={rel_index:?};head_relation_name={head_rel_name:?};head_relation_id={};max_active_rules={max_active_rules};head_projection={head_projection:?}", head_rel_id.0
        )),
    }
}

impl ResidentGraphCertifiedPlan {
    /// Inspect and seal one immutable plan allocation.
    pub fn inspect(plan: Arc<ExecutionPlan>, catalog: &ResidentGraphSchemaCatalog) -> Result<Self> {
        let certificate = ResidentGraphRouteCertificate::inspect(&plan, catalog)?;
        Ok(Self {
            plan,
            certificate,
            _allocation_owner: ResidentAllocationOwner::default(),
        })
    }

    /// Seal the same immutable plan with the original allocation receipts.
    pub fn inspect_with_allocation_owner(
        plan: Arc<ExecutionPlan>,
        catalog: &ResidentGraphSchemaCatalog,
        scope: &mut CompilationAllocationScope<'_>,
        plan_allocation_owner: Arc<dyn Send + Sync>,
    ) -> Result<Self> {
        let plan_custody = (plan, plan_allocation_owner);
        let certificate = ResidentGraphRouteCertificate::inspect_with_allocation_owner(
            &plan_custody.0,
            catalog,
            scope,
        )?;
        Ok(Self {
            plan: plan_custody.0,
            certificate,
            _allocation_owner: ResidentAllocationOwner(Some(plan_custody.1)),
        })
    }

    /// Return the exact plan inspected by this seal.
    pub fn plan(&self) -> &ExecutionPlan {
        &self.plan
    }

    /// Return the certificate produced for the sealed plan.
    pub fn certificate(&self) -> &ResidentGraphRouteCertificate {
        &self.certificate
    }
}

#[cfg(all(test, feature = "resident-graph-tests"))]
thread_local! {
    static RESIDENT_ROUTE_INSPECTION_COUNT: Cell<usize> = const { Cell::new(0) };
}

#[cfg(all(test, feature = "resident-graph-tests"))]
pub(crate) fn reset_resident_route_inspection_count() {
    RESIDENT_ROUTE_INSPECTION_COUNT.with(|count| count.set(0));
}

#[cfg(all(test, feature = "resident-graph-tests"))]
pub(crate) fn resident_route_inspection_count() -> usize {
    RESIDENT_ROUTE_INSPECTION_COUNT.with(Cell::get)
}

impl ResidentGraphRouteCertificate {
    fn insert_descriptor(
        &mut self,
        descriptor: ResidentText,
        scope: &mut Option<&mut CompilationAllocationScope<'_>>,
    ) -> Result<()> {
        if self.covered_route_descriptors.contains(&descriptor.value) {
            return Ok(());
        }
        if let Some(scope) = scope.as_deref_mut() {
            scope.acquire(xlog_core::btree_insert_allocation_bytes::<String, ()>(
                self.covered_route_descriptors.len(),
            )?)?;
        }
        let descriptor = descriptor.into_retained(scope)?;
        self.covered_route_descriptors.insert(descriptor);
        Ok(())
    }

    fn push_decline(
        &mut self,
        decline: ResidentGraphDeclineReason,
        scope: &mut Option<&mut CompilationAllocationScope<'_>>,
    ) -> Result<()> {
        if self.declines.len() == self.declines.capacity() {
            let next_owner = reserve_resident_vector::<ResidentGraphDeclineReason>(
                scope,
                self.declines.len().checked_add(1).ok_or_else(|| {
                    XlogError::allocation_extent_overflow("resident decline roster")
                })?,
            )?;
            self.declines.reserve(1);
            self._decline_storage = ResidentAllocationOwner(next_owner);
        }
        self.declines.push(decline);
        Ok(())
    }
    /// Inspects every explicit and implicit route in deterministic plan order.
    pub fn inspect(plan: &ExecutionPlan, catalog: &ResidentGraphSchemaCatalog) -> Result<Self> {
        Self::inspect_core(plan, catalog, &mut None)
    }

    /// Inspect the same complete route roster with intrinsic allocation custody.
    pub fn inspect_with_allocation_owner(
        plan: &ExecutionPlan,
        catalog: &ResidentGraphSchemaCatalog,
        scope: &mut CompilationAllocationScope<'_>,
    ) -> Result<Self> {
        let mut owned = scope.child_scope();
        owned.prepare_owner()?;
        match Self::inspect_core(plan, catalog, &mut Some(&mut owned)) {
            Ok(mut certificate) => {
                certificate._allocation_owner = ResidentAllocationOwner(Some(owned.into_owner()?));
                Ok(certificate)
            }
            Err(error) => Err(owned.into_failure(error)),
        }
    }

    fn inspect_core(
        plan: &ExecutionPlan,
        catalog: &ResidentGraphSchemaCatalog,
        scope: &mut Option<&mut CompilationAllocationScope<'_>>,
    ) -> Result<Self> {
        #[cfg(all(test, feature = "resident-graph-tests"))]
        RESIDENT_ROUTE_INSPECTION_COUNT.with(|count| count.set(count.get() + 1));
        let mut certificate = Self {
            covered_route_descriptors: BTreeSet::new(),
            declines: Vec::new(),
            schema_catalog: catalog.clone_with_allocation_owner(scope)?,
            plan_fingerprint: 0,
            _decline_storage: ResidentAllocationOwner::default(),
            _allocation_owner: ResidentAllocationOwner::default(),
        };
        certificate.insert_descriptor(
            resident_text(scope, format_args!("plan;scc_count={}", plan.sccs.len()))?,
            scope,
        )?;
        for (scc_index, scc) in plan.sccs.iter().enumerate() {
            certificate.insert_descriptor(
                resident_text(
                    scope,
                    format_args!(
                        "plan;scc={scc_index};id={};recursive={};predicate_count={}",
                        scc.id,
                        scc.is_recursive,
                        scc.predicates.len()
                    ),
                )?,
                scope,
            )?;
            for (predicate_index, predicate) in scc.predicates.iter().enumerate() {
                certificate.insert_descriptor(
                    resident_text(
                        scope,
                        format_args!(
                            "plan;scc={scc_index};predicate={predicate_index};name={predicate:?}"
                        ),
                    )?,
                    scope,
                )?;
            }
        }
        certificate.insert_descriptor(
            resident_text(
                scope,
                format_args!("plan;stratum_count={}", plan.strata.len()),
            )?,
            scope,
        )?;
        for (stratum_index, stratum) in plan.strata.iter().enumerate() {
            certificate.insert_descriptor(
                resident_text(
                    scope,
                    format_args!(
                        "plan;stratum={stratum_index};id={};scc_count={}",
                        stratum.id,
                        stratum.sccs.len()
                    ),
                )?,
                scope,
            )?;
            for (scc_position, scc_id) in stratum.sccs.iter().enumerate() {
                certificate.insert_descriptor(
                    resident_text(
                        scope,
                        format_args!("plan;stratum={stratum_index};scc={scc_position};id={scc_id}"),
                    )?,
                    scope,
                )?;
            }
        }
        certificate.insert_descriptor(
            resident_text(
                scope,
                format_args!("plan;rules_by_scc_count={}", plan.rules_by_scc.len()),
            )?,
            scope,
        )?;
        certificate.insert_descriptor(
            resident_text(
                scope,
                format_args!(
                    "plan;generated_query_rule_count={}",
                    plan.generated_query_rules.len()
                ),
            )?,
            scope,
        )?;
        for (position, provenance) in plan.generated_query_rules.iter().enumerate() {
            certificate.insert_descriptor(
                resident_text(
                    scope,
                    format_args!(
                        "plan;generated_query_rule={position};query_index={};scc={};rule={}",
                        provenance.query_index, provenance.scc_index, provenance.rule_index
                    ),
                )?,
                scope,
            )?;
        }
        for (scc_index, rules) in plan.rules_by_scc.iter().enumerate() {
            certificate.insert_descriptor(
                resident_text(
                    scope,
                    format_args!("plan;rules_by_scc={scc_index};rule_count={}", rules.len()),
                )?,
                scope,
            )?;
            for (rule_index, rule) in rules.iter().enumerate() {
                certificate.insert_descriptor(
                    resident_text(
                        scope,
                        format_args!(
                            "plan;rules_by_scc={scc_index};rule={rule_index};head={:?};meta={:?}",
                            rule.head, rule.meta
                        ),
                    )?,
                    scope,
                )?;
            }
        }
        certificate.insert_descriptor(
            resident_text(
                scope,
                format_args!("plan;est_memory_peak={}", plan.est_memory_peak),
            )?,
            scope,
        )?;
        certificate.insert_descriptor(
            resident_text(
                scope,
                format_args!("plan;rel_arity_count={}", plan.rel_arities.len()),
            )?,
            scope,
        )?;
        let _rel_arities_receipt =
            reserve_resident_vector::<(RelId, usize)>(scope, plan.rel_arities.len())?;
        let mut rel_arities = plan
            .rel_arities
            .iter()
            .map(|(&relation, &arity)| (relation, arity))
            .collect::<Vec<_>>();
        rel_arities.sort_unstable_by_key(|(relation, _)| *relation);
        for (position, (relation, arity)) in rel_arities.into_iter().enumerate() {
            certificate.insert_descriptor(
                resident_text(
                    scope,
                    format_args!(
                        "plan;rel_arity={position};relation={};arity={arity}",
                        relation.0
                    ),
                )?,
                scope,
            )?;
        }
        for (scc_index, scc) in plan.sccs.iter().enumerate() {
            let Some(rules) = plan.rules_by_scc.get(scc_index) else {
                certificate.push_decline(
                    ResidentGraphDeclineReason::UnsupportedNode {
                        path: resident_text(scope, format_args!("scc={scc_index}"))?
                            .into_retained(scope)?,
                        node: "missing_rule_vector",
                    },
                    scope,
                )?;
                continue;
            };
            for (rule_index, rule) in rules.iter().enumerate() {
                certificate.visit(
                    catalog,
                    scc_index,
                    rule_index,
                    scc.is_recursive,
                    &rule.body,
                    "primary/root",
                    scope,
                )?;
                let identity = resident_text(
                    scope,
                    format_args!(
                        "scc={scc_index};rule={rule_index};head={};schema={:#?}",
                        rule.head, rule.meta.schema
                    ),
                )?;
                certificate.insert_descriptor(
                    resident_text(scope, format_args!("{identity};implicit=rule_result_union"))?,
                    scope,
                )?;
                certificate.insert_descriptor(
                    resident_text(scope, format_args!("{identity};implicit=full_row_dedup"))?,
                    scope,
                )?;
                if scc.is_recursive {
                    certificate.insert_descriptor(
                        resident_text(
                            scope,
                            format_args!("{identity};implicit=novel_tuple_difference"),
                        )?,
                        scope,
                    )?;
                    certificate.insert_descriptor(
                        resident_text(
                            scope,
                            format_args!("{identity};implicit=device_convergence"),
                        )?,
                        scope,
                    )?;
                }
            }
        }
        certificate.plan_fingerprint = stable_descriptor_fingerprint(
            certificate
                .covered_route_descriptors
                .iter()
                .map(String::as_bytes),
        );
        Ok(certificate)
    }

    /// Whether every inspected route has a resident implementation.
    pub fn is_supported(&self) -> bool {
        self.declines.is_empty()
    }

    /// Deterministically ordered route occurrences covered by this proof.
    pub fn covered_route_descriptors(&self) -> &BTreeSet<String> {
        &self.covered_route_descriptors
    }

    /// Fail-closed reasons collected during inspection.
    pub fn declines(&self) -> &[ResidentGraphDeclineReason] {
        &self.declines
    }

    /// Stable binding between this certificate and every inspected physical
    /// route, expression, key position, schema, and implicit set operation.
    pub fn plan_fingerprint(&self) -> u64 {
        self.plan_fingerprint
    }

    /// Re-inspects a plan and proves that it is the exact plan certified by
    /// this value. Preparation performs this check before workspace allocation.
    pub fn matches_plan(&self, plan: &ExecutionPlan) -> Result<bool> {
        let inspected = Self::inspect(plan, &self.schema_catalog)?;
        Ok(inspected.plan_fingerprint == self.plan_fingerprint
            && inspected.covered_route_descriptors == self.covered_route_descriptors
            && inspected.declines == self.declines)
    }

    pub(crate) fn schema_for(&self, relation: RelId) -> Option<&Schema> {
        self.schema_catalog.schema(relation)
    }

    pub(crate) fn node_schema(&self, node: &RirNode) -> Option<Schema> {
        node_schema(&self.schema_catalog, node)
    }

    fn visit(
        &mut self,
        catalog: &ResidentGraphSchemaCatalog,
        scc_index: usize,
        rule_index: usize,
        recursive: bool,
        node: &RirNode,
        path: &str,
        scope: &mut Option<&mut CompilationAllocationScope<'_>>,
    ) -> Result<()> {
        self.validate_node_route(catalog, node, path, scope)?;
        let scan_schema = match node {
            RirNode::Scan { rel } => match catalog.descriptor(*rel, scope)? {
                Some(descriptor) => descriptor,
                None => {
                    self.push_decline(
                        ResidentGraphDeclineReason::MissingScanSchema { relation: *rel },
                        scope,
                    )?;
                    ResidentText::empty()
                }
            },
            _ => ResidentText::empty(),
        };
        let node_descriptor = resident_node_descriptor(node, scope)?;
        self.insert_descriptor(resident_text(scope, format_args!(
            "scc={scc_index};rule={rule_index};recursive={recursive};path={path};node={node_descriptor};scan_schema={scan_schema}"
        ))?, scope)?;
        match node {
            RirNode::Unit | RirNode::Scan { .. } => {}
            RirNode::Filter { input, .. }
            | RirNode::Project { input, .. }
            | RirNode::Distinct { input, .. } => self.visit(
                catalog,
                scc_index,
                rule_index,
                recursive,
                input,
                &resident_text(scope, format_args!("{path}/input"))?.value,
                scope,
            )?,
            RirNode::Join {
                left,
                right,
                join_type,
                ..
            } => {
                if !matches!(join_type, JoinType::Inner | JoinType::Semi) {
                    self.push_decline(
                        ResidentGraphDeclineReason::UnsupportedJoin {
                            path: resident_text(scope, format_args!("{path}"))?
                                .into_retained(scope)?,
                            join_type: *join_type,
                        },
                        scope,
                    )?;
                }
                self.visit(
                    catalog,
                    scc_index,
                    rule_index,
                    recursive,
                    left,
                    &resident_text(scope, format_args!("{path}/left"))?.value,
                    scope,
                )?;
                self.visit(
                    catalog,
                    scc_index,
                    rule_index,
                    recursive,
                    right,
                    &resident_text(scope, format_args!("{path}/right"))?.value,
                    scope,
                )?;
            }
            RirNode::ChainJoin {
                left,
                right,
                fallback,
                ..
            } => {
                self.visit(
                    catalog,
                    scc_index,
                    rule_index,
                    recursive,
                    left,
                    &resident_text(scope, format_args!("{path}/primary/left"))?.value,
                    scope,
                )?;
                self.visit(
                    catalog,
                    scc_index,
                    rule_index,
                    recursive,
                    right,
                    &resident_text(scope, format_args!("{path}/primary/right"))?.value,
                    scope,
                )?;
                self.visit(
                    catalog,
                    scc_index,
                    rule_index,
                    recursive,
                    fallback,
                    &resident_text(scope, format_args!("{path}/alternative/captured_fallback"))?
                        .value,
                    scope,
                )?;
            }
            RirNode::Union { inputs } => {
                for (index, input) in inputs.iter().enumerate() {
                    self.visit(
                        catalog,
                        scc_index,
                        rule_index,
                        recursive,
                        input,
                        &resident_text(scope, format_args!("{path}/input[{index}]"))?.value,
                        scope,
                    )?;
                }
            }
            RirNode::Diff { left, right } => {
                self.visit(
                    catalog,
                    scc_index,
                    rule_index,
                    recursive,
                    left,
                    &resident_text(scope, format_args!("{path}/left"))?.value,
                    scope,
                )?;
                self.visit(
                    catalog,
                    scc_index,
                    rule_index,
                    recursive,
                    right,
                    &resident_text(scope, format_args!("{path}/right"))?.value,
                    scope,
                )?;
            }
            RirNode::Fixpoint {
                base,
                recursive: step,
                ..
            } => {
                self.visit(
                    catalog,
                    scc_index,
                    rule_index,
                    recursive,
                    base,
                    &resident_text(scope, format_args!("{path}/base"))?.value,
                    scope,
                )?;
                self.visit(
                    catalog,
                    scc_index,
                    rule_index,
                    recursive,
                    step,
                    &resident_text(scope, format_args!("{path}/recursive"))?.value,
                    scope,
                )?;
            }
            RirNode::MultiWayJoin {
                inputs, fallback, ..
            } => {
                self.push_decline(
                    ResidentGraphDeclineReason::UnsupportedNode {
                        path: resident_text(scope, format_args!("{path}"))?.into_retained(scope)?,
                        node: "multi_way_join",
                    },
                    scope,
                )?;
                for (index, input) in inputs.iter().enumerate() {
                    self.visit(
                        catalog,
                        scc_index,
                        rule_index,
                        recursive,
                        input,
                        &resident_text(scope, format_args!("{path}/primary/input[{index}]"))?.value,
                        scope,
                    )?;
                }
                self.visit(
                    catalog,
                    scc_index,
                    rule_index,
                    recursive,
                    fallback,
                    &resident_text(scope, format_args!("{path}/alternative/captured_fallback"))?
                        .value,
                    scope,
                )?;
            }
            RirNode::GroupBy { input, .. } => {
                self.push_decline(
                    ResidentGraphDeclineReason::UnsupportedNode {
                        path: resident_text(scope, format_args!("{path}"))?.into_retained(scope)?,
                        node: "group_by",
                    },
                    scope,
                )?;
                self.visit(
                    catalog,
                    scc_index,
                    rule_index,
                    recursive,
                    input,
                    &resident_text(scope, format_args!("{path}/input"))?.value,
                    scope,
                )?;
            }
            RirNode::TensorMaskedJoin { .. } => self.push_decline(
                ResidentGraphDeclineReason::UnsupportedNode {
                    path: resident_text(scope, format_args!("{path}"))?.into_retained(scope)?,
                    node: "tensor_masked_join",
                },
                scope,
            )?,
        }
        Ok(())
    }

    fn validate_node_route(
        &mut self,
        catalog: &ResidentGraphSchemaCatalog,
        node: &RirNode,
        path: &str,
        scope: &mut Option<&mut CompilationAllocationScope<'_>>,
    ) -> Result<()> {
        let schema = node_schema_core(catalog, node, scope)?;
        let mut unsupported = schema
            .as_ref()
            .is_some_and(|schema| schema.schema.arity() > RESIDENT_GRAPH_MAX_INTERMEDIATE_ARITY)
            .then_some("intermediate_arity")
            .or_else(|| {
                schema
                    .as_ref()
                    .is_some_and(|schema| {
                        schema
                            .schema
                            .columns
                            .iter()
                            .any(|(_, scalar)| !resident_scalar_supported(*scalar))
                    })
                    .then_some("scalar_type")
            });
        if unsupported.is_none() {
            unsupported = match node {
                RirNode::Filter { input, predicate } => node_schema_core(catalog, input, scope)?
                    .filter(|schema| resident_predicate_supported(predicate, &schema.schema))
                    .is_none()
                    .then_some("filter_expression"),
                RirNode::Project { input, columns } => node_schema_core(catalog, input, scope)?
                    .filter(|schema| resident_projection_supported(columns, &schema.schema))
                    .is_none()
                    .then_some("project_expression"),
                RirNode::Join {
                    left,
                    right,
                    left_keys,
                    right_keys,
                    join_type: JoinType::Inner | JoinType::Semi,
                } => {
                    let compatible = (|| -> Result<Option<()>> {
                        let Some(left_schema) = node_schema_core(catalog, left, scope)? else {
                            return Ok(None);
                        };
                        let Some(right_schema) = node_schema_core(catalog, right, scope)? else {
                            return Ok(None);
                        };
                        if left_keys.len() != 1 || right_keys.len() != 1 {
                            return Ok(None);
                        }
                        let Some(left_ty) = left_schema.schema.column_type(left_keys[0]) else {
                            return Ok(None);
                        };
                        let Some(right_ty) = right_schema.schema.column_type(right_keys[0]) else {
                            return Ok(None);
                        };
                        Ok((left_ty == right_ty && resident_scalar_supported(left_ty))
                            .then_some(()))
                    })()?
                    .is_some();
                    (!compatible).then_some("join_key_layout")
                }
                RirNode::ChainJoin {
                    left,
                    right,
                    left_key,
                    right_key,
                    output_columns,
                    ..
                } => {
                    let compatible = (|| -> Result<Option<()>> {
                        let Some(left_schema) = node_schema_core(catalog, left, scope)? else {
                            return Ok(None);
                        };
                        let Some(right_schema) = node_schema_core(catalog, right, scope)? else {
                            return Ok(None);
                        };
                        let Some(left_ty) = left_schema.schema.column_type(*left_key) else {
                            return Ok(None);
                        };
                        let Some(right_ty) = right_schema.schema.column_type(*right_key) else {
                            return Ok(None);
                        };
                        if left_ty != right_ty || !resident_scalar_supported(left_ty) {
                            return Ok(None);
                        }
                        let joined =
                            joined_schema_core(&left_schema.schema, &right_schema.schema, scope)?;
                        Ok(
                            resident_projection_supported(output_columns, &joined.schema)
                                .then_some(()),
                        )
                    })()?
                    .is_some();
                    (!compatible).then_some("chain_join_layout")
                }
                RirNode::Distinct { input, key_cols } => node_schema_core(catalog, input, scope)?
                    .filter(|schema| key_cols.iter().copied().eq(0..schema.schema.arity()))
                    .is_none()
                    .then_some("partial_row_distinct"),
                _ => None,
            };
        }
        if let Some(node) = unsupported {
            self.push_decline(
                ResidentGraphDeclineReason::UnsupportedNode {
                    path: resident_text(scope, format_args!("{path}"))?.into_retained(scope)?,
                    node,
                },
                scope,
            )?;
        }
        Ok(())
    }
}

fn resident_scalar_supported(scalar: ScalarType) -> bool {
    matches!(
        scalar,
        ScalarType::Symbol | ScalarType::U32 | ScalarType::U64
    )
}

fn constant_scalar(value: &ConstValue) -> ScalarType {
    match value {
        ConstValue::U32(_) => ScalarType::U32,
        ConstValue::U64(_) => ScalarType::U64,
        ConstValue::I32(_) => ScalarType::I32,
        ConstValue::I64(_) => ScalarType::I64,
        ConstValue::F32(_) => ScalarType::F32,
        ConstValue::F64(_) => ScalarType::F64,
        ConstValue::Bool(_) => ScalarType::Bool,
        ConstValue::Symbol(_) => ScalarType::Symbol,
    }
}

fn scalar_expression_type(expression: &Expr, schema: &Schema) -> Option<ScalarType> {
    match expression {
        Expr::Column(index) => schema.column_type(*index),
        Expr::Const(value) => Some(constant_scalar(value)),
        _ => None,
    }
}

fn resident_predicate_supported(expression: &Expr, schema: &Schema) -> bool {
    match expression {
        Expr::Compare { left, right, .. } => {
            let Some(left_ty) = scalar_expression_type(left, schema) else {
                return false;
            };
            let Some(right_ty) = scalar_expression_type(right, schema) else {
                return false;
            };
            left_ty == right_ty && resident_scalar_supported(left_ty)
        }
        Expr::And(expressions) => {
            !expressions.is_empty()
                && expressions
                    .iter()
                    .all(|expression| resident_predicate_supported(expression, schema))
        }
        _ => false,
    }
}

fn resident_projection_supported(expressions: &[ProjectExpr], schema: &Schema) -> bool {
    expressions.iter().all(|expression| match expression {
        ProjectExpr::Column(index) => schema
            .column_type(*index)
            .is_some_and(resident_scalar_supported),
        ProjectExpr::Computed(Expr::Const(value), declared) => {
            *declared == constant_scalar(value) && resident_scalar_supported(*declared)
        }
        _ => false,
    })
}

struct ResidentSchema {
    schema: Schema,
    _allocation_owner: ResidentAllocationOwner,
}

fn generated_schema(
    count: usize,
    mut scalar: impl FnMut(usize) -> Option<ScalarType>,
    scope: &mut Option<&mut CompilationAllocationScope<'_>>,
) -> Result<Option<ResidentSchema>> {
    let owner = if let Some(scope) = scope.as_deref_mut() {
        let mut bytes = xlog_core::push_vector_allocation_bytes::<(String, ScalarType)>(count)?
            .checked_add(xlog_core::push_vector_allocation_bytes::<usize>(count)?)
            .and_then(|bytes| {
                bytes.checked_add(xlog_core::push_vector_allocation_bytes::<String>(count).ok()?)
            })
            .ok_or_else(|| XlogError::allocation_extent_overflow("resident schema"))?;
        for index in 0..count {
            let name =
                xlog_core::error::formatted_text_allocation_bytes(format_args!("column_{index}"))?;
            // Schema::new keeps both the column name and its original sort-label clone.
            bytes = bytes
                .checked_add(name)
                .and_then(|bytes| bytes.checked_add(name))
                .ok_or_else(|| XlogError::allocation_extent_overflow("resident schema names"))?;
        }
        Some(scope.reserve_allocation(bytes)?)
    } else {
        None
    };
    let output = (0..count)
        .map(|index| scalar(index).map(|scalar| (format!("column_{index}"), scalar)))
        .collect::<Option<Vec<_>>>();
    Ok(output.map(|output| ResidentSchema {
        schema: Schema::new(output),
        _allocation_owner: ResidentAllocationOwner(owner),
    }))
}

fn joined_schema_core(
    left: &Schema,
    right: &Schema,
    scope: &mut Option<&mut CompilationAllocationScope<'_>>,
) -> Result<ResidentSchema> {
    let count = left
        .columns
        .len()
        .checked_add(right.columns.len())
        .ok_or_else(|| XlogError::allocation_extent_overflow("resident joined schema"))?;
    generated_schema(
        count,
        |index| {
            if index < left.columns.len() {
                Some(left.columns[index].1)
            } else {
                Some(right.columns[index - left.columns.len()].1)
            }
        },
        scope,
    )
    .map(|schema| schema.expect("joined source columns have scalar types"))
}

pub(crate) fn node_schema(catalog: &ResidentGraphSchemaCatalog, node: &RirNode) -> Option<Schema> {
    node_schema_core(catalog, node, &mut None)
        .expect("ordinary schema has no allocation callback")
        .map(|schema| schema.schema)
}

fn node_schema_core(
    catalog: &ResidentGraphSchemaCatalog,
    node: &RirNode,
    scope: &mut Option<&mut CompilationAllocationScope<'_>>,
) -> Result<Option<ResidentSchema>> {
    match node {
        RirNode::Unit => Ok(Some(ResidentSchema {
            schema: Schema::new(Vec::new()),
            _allocation_owner: ResidentAllocationOwner::default(),
        })),
        RirNode::Scan { rel } => {
            let Some(schema) = catalog.schema(*rel) else {
                return Ok(None);
            };
            let owner = scope
                .as_deref_mut()
                .map(|scope| scope.reserve_allocation(schema.owned_allocation_bytes()?))
                .transpose()?;
            Ok(Some(ResidentSchema {
                schema: schema.clone(),
                _allocation_owner: ResidentAllocationOwner(owner),
            }))
        }
        RirNode::Filter { input, .. } | RirNode::Distinct { input, .. } => {
            node_schema_core(catalog, input, scope)
        }
        RirNode::Project { input, columns } => {
            let Some(input) = node_schema_core(catalog, input, scope)? else {
                return Ok(None);
            };
            generated_schema(
                columns.len(),
                |index| match &columns[index] {
                    ProjectExpr::Column(column) => input.schema.column_type(*column),
                    ProjectExpr::Computed(_, scalar) => Some(*scalar),
                },
                scope,
            )
        }
        RirNode::Join {
            left,
            right,
            join_type,
            ..
        } => {
            let Some(left) = node_schema_core(catalog, left, scope)? else {
                return Ok(None);
            };
            if matches!(join_type, JoinType::Semi | JoinType::Anti) {
                Ok(Some(left))
            } else {
                let Some(right) = node_schema_core(catalog, right, scope)? else {
                    return Ok(None);
                };
                Ok(Some(joined_schema_core(
                    &left.schema,
                    &right.schema,
                    scope,
                )?))
            }
        }
        RirNode::ChainJoin { fallback, .. } | RirNode::MultiWayJoin { fallback, .. } => {
            node_schema_core(catalog, fallback, scope)
        }
        RirNode::Union { inputs } => match inputs.first() {
            Some(input) => node_schema_core(catalog, input, scope),
            None => Ok(None),
        },
        RirNode::Diff { left, .. } => node_schema_core(catalog, left, scope),
        RirNode::Fixpoint { base, .. } => node_schema_core(catalog, base, scope),
        RirNode::GroupBy { .. } | RirNode::TensorMaskedJoin { .. } => Ok(None),
    }
}

fn stable_descriptor_fingerprint<'a>(descriptors: impl IntoIterator<Item = &'a [u8]>) -> u64 {
    let mut hash = 0xcbf2_9ce4_8422_2325u64;
    for descriptor in descriptors {
        for byte in (descriptor.len() as u64)
            .to_le_bytes()
            .iter()
            .chain(descriptor)
        {
            hash ^= u64::from(*byte);
            hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
        }
    }
    hash
}

/// Device-written terminal status for a resident graph transaction.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ResidentGraphDeviceStatus {
    /// The graph converged and staged output is valid.
    Success {
        /// Device-observed recursive iterations completed before convergence.
        iterations: u32,
    },
    /// The exact configured iteration limit was exhausted.
    IterationLimit {
        /// Configured maximum recursive iterations.
        limit: u32,
        /// Iterations completed when the device stopped the graph.
        completed: u32,
    },
    /// An operator's exact output exceeded its reserved row capacity.
    CapacityOverflow {
        /// Physical operator that exceeded its reserved row count.
        op_id: u32,
        /// Exact row count required by the operator.
        required: u64,
        /// Reserved row capacity available to the operator.
        capacity: u64,
    },
    /// A bounded device resource was insufficient.
    ResourceExhausted {
        /// Physical operator that exhausted the bounded resource.
        op_id: u32,
        /// Stable name of the exhausted device resource.
        resource: &'static str,
        /// Exact resource quantity required by the operator.
        required: u64,
        /// Reserved resource quantity available to the operator.
        capacity: u64,
    },
}

/// Typed error decoded from the graph's single terminal receipt.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ResidentGraphExecutionError {
    /// A complete prelaunch inspection selected the existing GPU route instead.
    Declined(ResidentGraphDeclineReason),
    /// The exact configured iteration limit was exhausted.
    IterationLimit {
        /// Configured maximum recursive iterations.
        limit: u32,
        /// Iterations completed before the device reported exhaustion.
        completed: u32,
    },
    /// An operator's exact output exceeded its reserved row capacity.
    CapacityOverflow {
        /// Physical operator that exceeded its reserved row count.
        op_id: u32,
        /// Exact row count required by the operator.
        required: u64,
        /// Reserved row capacity available to the operator.
        capacity: u64,
    },
    /// A bounded device resource was insufficient.
    ResourceExhausted {
        /// Physical operator that exhausted the bounded resource.
        op_id: u32,
        /// Stable name of the exhausted device resource.
        resource: &'static str,
        /// Exact resource quantity required by the operator.
        required: u64,
        /// Reserved resource quantity available to the operator.
        capacity: u64,
    },
    /// Setup or execution failed before a valid terminal status existed.
    Runtime(String),
}

impl fmt::Display for ResidentGraphExecutionError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "{self:?}")
    }
}

impl std::error::Error for ResidentGraphExecutionError {}

/// Test-only device-kernel status request.  Production execution never writes
/// terminal status from the host.
#[derive(Debug, Clone)]
pub struct ResidentGraphDeviceStatusTestInjection {
    pub(crate) after_op: u32,
    pub(crate) status: ResidentGraphDeviceStatus,
}

impl ResidentGraphDeviceStatusTestInjection {
    /// Requests a device status-writer kernel after the indexed physical op.
    pub fn device_kernel_after_op(after_op: u32, status: ResidentGraphDeviceStatus) -> Self {
        Self { after_op, status }
    }
}

/// Prelaunch options for a resident transaction.
#[derive(Debug, Clone, Default)]
pub struct ResidentGraphPrepareOptions {
    pub(crate) test_device_status: Option<ResidentGraphDeviceStatusTestInjection>,
    pub(crate) latency_diagnostic_sample: Option<u64>,
}

impl ResidentGraphPrepareOptions {
    /// Adds a device-written test status without permitting host injection.
    pub fn with_test_device_status(
        mut self,
        injection: ResidentGraphDeviceStatusTestInjection,
    ) -> Self {
        self.test_device_status = Some(injection);
        self
    }

    /// Enables cold-path prepare timing for one explicitly numbered diagnostic sample.
    #[doc(hidden)]
    pub fn with_latency_diagnostic_sample(mut self, sample: u64) -> Self {
        self.latency_diagnostic_sample = Some(sample);
        self
    }
}

/// Copyable, opt-in cold-path timings captured before a resident graph launches.
#[doc(hidden)]
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct ResidentGraphPrepareDiagnosticSnapshot {
    pub(crate) sample: u64,
    pub(crate) total_ns: u64,
    pub(crate) admission_and_source_snapshot_ns: u64,
    pub(crate) execution_domain_and_build_setup_ns: u64,
    pub(crate) logical_schedule_planning_ns: u64,
    pub(crate) manifest_compact_construction_ns: u64,
    pub(crate) schedule_lowering_ns: u64,
    pub(crate) reservation_ns: u64,
    pub(crate) relation_preparation_ns: u64,
    pub(crate) count_initialization_ns: u64,
    pub(crate) workspace_preparation_ns: u64,
    pub(crate) metadata_binding_construction_ns: u64,
    pub(crate) metadata_preparation_ns: u64,
    pub(crate) reservation_validation_and_release_ns: u64,
    pub(crate) pinned_receipt_ns: u64,
    pub(crate) graph_body_capture_ns: u64,
    pub(crate) graph_instantiate_ns: u64,
    pub(crate) validation_owner_assembly_ns: u64,
    pub(crate) unattributed_ns: u64,
    pub(crate) required_reservation_bytes: u64,
    pub(crate) logical_relation_values: u64,
    pub(crate) physical_relation_slots: u64,
    pub(crate) relation_device_allocation_calls: u64,
    pub(crate) relation_reserved_bytes: u64,
    pub(crate) relation_slot_preparation_ns_max: u64,
    pub(crate) count_memset_calls: u64,
    pub(crate) count_memset_bytes: u64,
    pub(crate) count_initialization_ns_max: u64,
    pub(crate) workspace_provider_calls: u64,
    pub(crate) workspace_reserved_bytes: u64,
    pub(crate) filter_scratch_preparation_ns: u64,
    pub(crate) filter_scratch_reserved_bytes: u64,
    pub(crate) set_workspace_preparation_ns: u64,
    pub(crate) set_workspace_reserved_bytes: u64,
    pub(crate) join_workspace_preparation_ns: u64,
    pub(crate) join_workspace_reserved_bytes: u64,
    pub(crate) control_preparation_ns: u64,
    pub(crate) control_reserved_bytes: u64,
    pub(crate) metadata_provider_calls: u64,
    pub(crate) metadata_reserved_bytes: u64,
    pub(crate) metadata_initial_htod_calls: u64,
    pub(crate) metadata_initial_htod_bytes: u64,
    pub(crate) device_trace_preparation_ns: u64,
    pub(crate) device_trace_reserved_bytes: u64,
    pub(crate) device_trace_initial_htod_calls: u64,
    pub(crate) device_trace_initial_htod_bytes: u64,
    pub(crate) schema_winners_preparation_ns: u64,
    pub(crate) schema_winners_reserved_bytes: u64,
    pub(crate) schema_winners_initial_htod_calls: u64,
    pub(crate) schema_winners_initial_htod_bytes: u64,
    pub(crate) receipt_preparation_ns: u64,
    pub(crate) receipt_reserved_bytes: u64,
    pub(crate) receipt_initial_htod_calls: u64,
    pub(crate) receipt_initial_htod_bytes: u64,
    pub(crate) schedule_program_preparation_ns: u64,
    pub(crate) schedule_program_reserved_bytes: u64,
    pub(crate) schedule_program_initial_htod_calls: u64,
    pub(crate) schedule_program_initial_htod_bytes: u64,
    pub(crate) compact_ops: u64,
    pub(crate) compact_waves: u64,
    pub(crate) compact_regions: u64,
    pub(crate) conditional_regions: u64,
    pub(crate) parent_graph_nodes: u64,
    pub(crate) conditional_body_nodes: u64,
}

impl ResidentGraphPrepareDiagnosticSnapshot {
    /// Format one stable, machine-tokenizable diagnostic line without emitting it.
    #[doc(hidden)]
    pub fn format_line(self) -> String {
        let Self {
            sample,
            total_ns,
            admission_and_source_snapshot_ns,
            execution_domain_and_build_setup_ns,
            logical_schedule_planning_ns,
            manifest_compact_construction_ns,
            schedule_lowering_ns,
            reservation_ns,
            relation_preparation_ns,
            count_initialization_ns,
            workspace_preparation_ns,
            metadata_binding_construction_ns,
            metadata_preparation_ns,
            reservation_validation_and_release_ns,
            pinned_receipt_ns,
            graph_body_capture_ns,
            graph_instantiate_ns,
            validation_owner_assembly_ns,
            unattributed_ns,
            required_reservation_bytes,
            logical_relation_values,
            physical_relation_slots,
            relation_device_allocation_calls,
            relation_reserved_bytes,
            relation_slot_preparation_ns_max,
            count_memset_calls,
            count_memset_bytes,
            count_initialization_ns_max,
            workspace_provider_calls,
            workspace_reserved_bytes,
            filter_scratch_preparation_ns,
            filter_scratch_reserved_bytes,
            set_workspace_preparation_ns,
            set_workspace_reserved_bytes,
            join_workspace_preparation_ns,
            join_workspace_reserved_bytes,
            control_preparation_ns,
            control_reserved_bytes,
            metadata_provider_calls,
            metadata_reserved_bytes,
            metadata_initial_htod_calls,
            metadata_initial_htod_bytes,
            device_trace_preparation_ns,
            device_trace_reserved_bytes,
            device_trace_initial_htod_calls,
            device_trace_initial_htod_bytes,
            schema_winners_preparation_ns,
            schema_winners_reserved_bytes,
            schema_winners_initial_htod_calls,
            schema_winners_initial_htod_bytes,
            receipt_preparation_ns,
            receipt_reserved_bytes,
            receipt_initial_htod_calls,
            receipt_initial_htod_bytes,
            schedule_program_preparation_ns,
            schedule_program_reserved_bytes,
            schedule_program_initial_htod_calls,
            schedule_program_initial_htod_bytes,
            compact_ops,
            compact_waves,
            compact_regions,
            conditional_regions,
            parent_graph_nodes,
            conditional_body_nodes,
        } = self;
        format!(
            "resident prepare phases: sample={sample} total_ns={total_ns} admission_and_source_snapshot_ns={admission_and_source_snapshot_ns} execution_domain_and_build_setup_ns={execution_domain_and_build_setup_ns} logical_schedule_planning_ns={logical_schedule_planning_ns} manifest_compact_construction_ns={manifest_compact_construction_ns} schedule_lowering_ns={schedule_lowering_ns} reservation_ns={reservation_ns} relation_preparation_ns={relation_preparation_ns} count_initialization_ns={count_initialization_ns} workspace_preparation_ns={workspace_preparation_ns} metadata_binding_construction_ns={metadata_binding_construction_ns} metadata_preparation_ns={metadata_preparation_ns} reservation_validation_and_release_ns={reservation_validation_and_release_ns} pinned_receipt_ns={pinned_receipt_ns} graph_body_capture_ns={graph_body_capture_ns} graph_instantiate_ns={graph_instantiate_ns} validation_owner_assembly_ns={validation_owner_assembly_ns} unattributed_ns={unattributed_ns} required_reservation_bytes={required_reservation_bytes} logical_relation_values={logical_relation_values} physical_relation_slots={physical_relation_slots} relation_device_allocation_calls={relation_device_allocation_calls} relation_reserved_bytes={relation_reserved_bytes} relation_slot_preparation_ns_max={relation_slot_preparation_ns_max} count_memset_calls={count_memset_calls} count_memset_bytes={count_memset_bytes} count_initialization_ns_max={count_initialization_ns_max} workspace_provider_calls={workspace_provider_calls} workspace_reserved_bytes={workspace_reserved_bytes} filter_scratch_preparation_ns={filter_scratch_preparation_ns} filter_scratch_reserved_bytes={filter_scratch_reserved_bytes} set_workspace_preparation_ns={set_workspace_preparation_ns} set_workspace_reserved_bytes={set_workspace_reserved_bytes} join_workspace_preparation_ns={join_workspace_preparation_ns} join_workspace_reserved_bytes={join_workspace_reserved_bytes} control_preparation_ns={control_preparation_ns} control_reserved_bytes={control_reserved_bytes} metadata_provider_calls={metadata_provider_calls} metadata_reserved_bytes={metadata_reserved_bytes} metadata_initial_htod_calls={metadata_initial_htod_calls} metadata_initial_htod_bytes={metadata_initial_htod_bytes} device_trace_preparation_ns={device_trace_preparation_ns} device_trace_reserved_bytes={device_trace_reserved_bytes} device_trace_initial_htod_calls={device_trace_initial_htod_calls} device_trace_initial_htod_bytes={device_trace_initial_htod_bytes} schema_winners_preparation_ns={schema_winners_preparation_ns} schema_winners_reserved_bytes={schema_winners_reserved_bytes} schema_winners_initial_htod_calls={schema_winners_initial_htod_calls} schema_winners_initial_htod_bytes={schema_winners_initial_htod_bytes} receipt_preparation_ns={receipt_preparation_ns} receipt_reserved_bytes={receipt_reserved_bytes} receipt_initial_htod_calls={receipt_initial_htod_calls} receipt_initial_htod_bytes={receipt_initial_htod_bytes} schedule_program_preparation_ns={schedule_program_preparation_ns} schedule_program_reserved_bytes={schedule_program_reserved_bytes} schedule_program_initial_htod_calls={schedule_program_initial_htod_calls} schedule_program_initial_htod_bytes={schedule_program_initial_htod_bytes} compact_ops={compact_ops} compact_waves={compact_waves} compact_regions={compact_regions} conditional_regions={conditional_regions} parent_graph_nodes={parent_graph_nodes} conditional_body_nodes={conditional_body_nodes} deallocation_calls=unavailable"
        )
    }
}

#[cfg(test)]
fn resident_graph_prepare_diagnostic_line(
    snapshot: Option<ResidentGraphPrepareDiagnosticSnapshot>,
) -> Option<String> {
    snapshot.map(ResidentGraphPrepareDiagnosticSnapshot::format_line)
}

/// Runtime route selected for one ordinary evaluation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ResidentGraphSelectionKind {
    /// The existing host-dispatched GPU executor ran the plan.
    ExistingGpu,
    /// A device-controlled conditional CUDA graph ran the plan.
    ResidentConditionalGraph,
}

/// Core-loop host transfer counters.
#[derive(Debug, Clone, Default)]
pub struct ResidentGraphCoreTransferStats {
    /// Host-to-device calls observed by the tracked memory runtime.
    pub tracked_htod_calls: u64,
    /// Host-to-device bytes observed by the tracked memory runtime.
    pub tracked_htod_bytes: u64,
    /// Device-to-host calls observed by the tracked memory runtime.
    pub tracked_dtoh_calls: u64,
    /// Device-to-host bytes observed by the tracked memory runtime.
    pub tracked_dtoh_bytes: u64,
    /// Device-to-host calls issued through provider metadata operations.
    pub provider_dtoh_calls: u64,
    /// Metadata device-to-host calls not attributable to the tracked runtime.
    pub untracked_metadata_dtoh_calls: u64,
}

/// The one bounded observation after the terminal synchronization.
#[derive(Debug, Clone, Default)]
pub struct ResidentGraphFinalObservationStats {
    /// Device-to-host calls used to read the terminal observation.
    pub dtoh_calls: u64,
    /// Device-to-host bytes used to read the terminal observation.
    pub dtoh_bytes: u64,
    /// Pinned host receipts used for the terminal observation.
    pub pinned_receipts: u64,
}

/// CUDA-event timing resolved after the graph completes.
#[derive(Debug, Clone, Default)]
pub struct ResidentGraphDeferredProfile {
    /// Resident scan and filter invocations covered by deferred CUDA timing.
    pub timed_scan_filter_invocations: u64,
    /// Device elapsed time resolved from CUDA events after completion.
    pub device_elapsed_ns: u64,
    /// Host synchronization time excluded from device execution timing.
    pub final_sync_misattributed_ns: u64,
}

/// Truthful telemetry for resident selection, execution, and decline.
#[derive(Debug, Clone)]
pub struct ResidentGraphExecutionStats {
    /// Runtime route selected for the evaluation.
    pub selection: ResidentGraphSelectionKind,
    /// Preflight decline reason when the existing GPU route was selected.
    pub decline: Option<ResidentGraphDeclineReason>,
    /// Conditional resident graph launches performed by this evaluation.
    pub conditional_graph_launches: u64,
    /// Host synchronizations performed to obtain terminal status.
    pub terminal_synchronizations: u64,
    /// Fixpoint iterations controlled by the host during the core loop.
    pub host_iterations: u64,
    /// Host allocations performed during the resident core loop.
    pub host_allocations: u64,
    /// Terminal statuses injected by the host rather than written by a device kernel.
    pub host_status_injections: u64,
    /// Deterministic device-to-host transfer contract violations.
    pub deterministic_d2h_violations: u64,
    /// Scan operations dispatched individually by the host.
    pub host_dispatched_scan_ops: u64,
    /// Filter operations dispatched individually by the host.
    pub host_dispatched_filter_ops: u64,
    /// Physical Scan nodes executed by the resident device graph.
    pub device_scan_invocations: u64,
    /// Physical Filter nodes executed by the resident device graph.
    pub device_filter_invocations: u64,
    /// Logical Scan count for the selected dependency-closed plan after
    /// excluding recursive variants whose input delta was empty.
    pub semantic_scan_invocations: u64,
    /// Logical Filter count for the selected dependency-closed plan after
    /// excluding recursive variants whose input delta was empty.
    pub semantic_filter_invocations: u64,
    /// Relation-store mutations staged until terminal success is authoritative.
    pub staged_store_mutations: u64,
    /// CUDA-event measurements resolved after the graph completes.
    pub deferred_profile: ResidentGraphDeferredProfile,
    /// Host/device transfers observed inside the resident core loop.
    pub core_transfers: ResidentGraphCoreTransferStats,
    /// Single bounded terminal observation made after synchronization.
    pub final_observation: ResidentGraphFinalObservationStats,
}

impl ResidentGraphExecutionStats {
    /// Telemetry for a call that deliberately remained on the existing GPU path.
    pub fn declined(reason: ResidentGraphDeclineReason) -> Self {
        Self {
            selection: ResidentGraphSelectionKind::ExistingGpu,
            decline: Some(reason),
            conditional_graph_launches: 0,
            terminal_synchronizations: 0,
            host_iterations: 0,
            host_allocations: 0,
            host_status_injections: 0,
            deterministic_d2h_violations: 0,
            host_dispatched_scan_ops: 0,
            host_dispatched_filter_ops: 0,
            device_scan_invocations: 0,
            device_filter_invocations: 0,
            semantic_scan_invocations: 0,
            semantic_filter_invocations: 0,
            staged_store_mutations: 0,
            deferred_profile: ResidentGraphDeferredProfile::default(),
            core_transfers: ResidentGraphCoreTransferStats::default(),
            final_observation: ResidentGraphFinalObservationStats::default(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use xlog_ir::{CompiledRule, GeneratedQueryRuleProvenance, RirMeta, Scc, Stratum};

    fn schema(columns: &[(&str, ScalarType)]) -> Schema {
        Schema::new(
            columns
                .iter()
                .map(|(name, scalar)| ((*name).to_owned(), *scalar))
                .collect(),
        )
    }

    fn rule(head: &str, body: RirNode, columns: &[(&str, ScalarType)]) -> CompiledRule {
        CompiledRule {
            head: head.to_owned(),
            body,
            meta: RirMeta {
                schema: schema(columns),
                ..RirMeta::default()
            },
        }
    }

    fn chain_join() -> RirNode {
        RirNode::ChainJoin {
            left: Box::new(RirNode::Scan { rel: RelId(1) }),
            right: Box::new(RirNode::Scan { rel: RelId(1) }),
            left_key: 0,
            right_key: 0,
            output_columns: vec![ProjectExpr::Column(0)],
            fallback: Box::new(RirNode::Join {
                left: Box::new(RirNode::Scan { rel: RelId(1) }),
                right: Box::new(RirNode::Scan { rel: RelId(1) }),
                left_keys: vec![0],
                right_keys: vec![0],
                join_type: JoinType::Inner,
            }),
        }
    }

    fn representative_plan() -> ExecutionPlan {
        let mut rel_arities = HashMap::new();
        rel_arities.insert(RelId(9), 0);
        rel_arities.insert(RelId(1), 2);
        ExecutionPlan {
            sccs: vec![
                Scc {
                    id: 10,
                    predicates: vec!["alpha".to_owned(), "beta".to_owned()],
                    is_recursive: true,
                },
                Scc {
                    id: 20,
                    predicates: Vec::new(),
                    is_recursive: false,
                },
                Scc {
                    id: 30,
                    predicates: vec!["gamma".to_owned()],
                    is_recursive: false,
                },
                Scc {
                    id: 40,
                    predicates: Vec::new(),
                    is_recursive: false,
                },
            ],
            strata: vec![
                Stratum {
                    id: 0,
                    sccs: vec![10],
                },
                Stratum {
                    id: 1,
                    sccs: Vec::new(),
                },
                Stratum {
                    id: 2,
                    sccs: vec![20, 30, 40],
                },
            ],
            rules_by_scc: vec![
                vec![
                    rule(
                        "alpha",
                        RirNode::Scan { rel: RelId(1) },
                        &[("left", ScalarType::U32), ("right", ScalarType::Symbol)],
                    ),
                    rule("beta", chain_join(), &[("value", ScalarType::U64)]),
                ],
                Vec::new(),
                vec![rule("gamma", RirNode::Unit, &[])],
                Vec::new(),
            ],
            generated_query_rules: vec![],
            est_memory_peak: 4_096,
            rel_arities,
        }
    }

    fn catalog() -> ResidentGraphSchemaCatalog {
        ResidentGraphSchemaCatalog::from_named_schemas([(
            "input".to_owned(),
            RelId(1),
            schema(&[("left", ScalarType::U32), ("right", ScalarType::Symbol)]),
        )])
    }

    fn assert_mutation_rejected(mutate: impl FnOnce(&mut ExecutionPlan)) {
        let plan = representative_plan();
        let certificate = ResidentGraphRouteCertificate::inspect(&plan, &catalog()).unwrap();
        let mut mutated = plan.clone();
        mutate(&mut mutated);
        assert!(!certificate.matches_plan(&mutated).unwrap());
    }

    fn assert_generated_query_provenance_mutation_rejected(
        mutate: impl FnOnce(&mut GeneratedQueryRuleProvenance),
    ) {
        let mut plan = representative_plan();
        plan.sccs[0].predicates[0] = "__xlog_query_0".to_owned();
        plan.rules_by_scc[0][0].head = "__xlog_query_0".to_owned();
        plan.generated_query_rules = vec![GeneratedQueryRuleProvenance {
            query_index: 0,
            scc_index: 0,
            rule_index: 0,
        }];
        let certificate = ResidentGraphRouteCertificate::inspect(&plan, &catalog()).unwrap();
        mutate(&mut plan.generated_query_rules[0]);
        assert!(!certificate.matches_plan(&plan).unwrap());
    }

    #[test]
    fn resident_node_descriptors_encode_only_local_fields() {
        let descendant = RirNode::Project {
            input: Box::new(RirNode::Scan { rel: RelId(9) }),
            columns: vec![ProjectExpr::Column(0)],
        };
        let project = RirNode::Project {
            input: Box::new(descendant),
            columns: vec![ProjectExpr::Computed(
                Expr::Const(ConstValue::U32(7)),
                ScalarType::U32,
            )],
        };

        let descriptor = resident_node_descriptor(&project);
        assert_eq!(
            descriptor,
            "project;column_count=1;computed:U32;constant:u32:7;"
        );
        assert!(!descriptor.contains("Scan"));
        assert!(!descriptor.contains("input"));
    }

    #[test]
    fn resident_node_descriptors_distinguish_local_semantics() {
        let left = resident_node_descriptor(&RirNode::Join {
            left: Box::new(RirNode::Unit),
            right: Box::new(RirNode::Unit),
            left_keys: vec![0],
            right_keys: vec![1],
            join_type: JoinType::Inner,
        });
        let right = resident_node_descriptor(&RirNode::Join {
            left: Box::new(RirNode::Unit),
            right: Box::new(RirNode::Unit),
            left_keys: vec![1],
            right_keys: vec![0],
            join_type: JoinType::Inner,
        });
        assert_ne!(left, right);

        let negative_zero = resident_node_descriptor(&RirNode::Filter {
            input: Box::new(RirNode::Unit),
            predicate: Expr::Const(ConstValue::F64(-0.0)),
        });
        let positive_zero = resident_node_descriptor(&RirNode::Filter {
            input: Box::new(RirNode::Unit),
            predicate: Expr::Const(ConstValue::F64(0.0)),
        });
        assert_ne!(negative_zero, positive_zero);
    }

    #[test]
    fn certificate_descriptors_remain_bounded_for_deep_unary_plans() {
        let mut body = RirNode::Scan { rel: RelId(1) };
        for _ in 0..128 {
            body = RirNode::Project {
                input: Box::new(body),
                columns: vec![ProjectExpr::Column(0)],
            };
        }
        let mut plan = representative_plan();
        plan.rules_by_scc[0][0].body = body;

        let certificate = ResidentGraphRouteCertificate::inspect(&plan, &catalog()).unwrap();
        let largest_descriptor = certificate
            .covered_route_descriptors
            .iter()
            .map(String::len)
            .max()
            .unwrap();
        assert!(largest_descriptor < 8_192, "{largest_descriptor}");
    }

    #[test]
    fn certified_plan_seals_the_exact_arc_and_certificate() {
        let plan = Arc::new(representative_plan());
        let certified = ResidentGraphCertifiedPlan::inspect(Arc::clone(&plan), &catalog()).unwrap();

        assert!(std::ptr::eq(Arc::as_ptr(&plan), certified.plan()));
        assert!(certified
            .certificate()
            .matches_plan(certified.plan())
            .unwrap());

        let mut independent_mutation = (*plan).clone();
        independent_mutation.est_memory_peak += 1;
        assert!(!certified
            .certificate()
            .matches_plan(&independent_mutation)
            .unwrap());
    }

    #[test]
    fn prepare_latency_diagnostics_are_explicit_and_sample_bound() {
        let disabled = ResidentGraphPrepareOptions::default();
        assert_eq!(disabled.latency_diagnostic_sample, None);

        let enabled = disabled.with_latency_diagnostic_sample(7);
        assert_eq!(enabled.latency_diagnostic_sample, Some(7));
    }

    #[test]
    fn prepare_latency_diagnostic_format_is_aligned_and_default_off_is_silent() {
        let mut snapshot = ResidentGraphPrepareDiagnosticSnapshot {
            sample: 17,
            total_ns: 101,
            admission_and_source_snapshot_ns: 102,
            execution_domain_and_build_setup_ns: 109,
            metadata_binding_construction_ns: 110,
            reservation_validation_and_release_ns: 111,
            relation_device_allocation_calls: 103,
            workspace_provider_calls: 104,
            metadata_initial_htod_calls: 105,
            metadata_initial_htod_bytes: 106,
            graph_instantiate_ns: 107,
            unattributed_ns: 108,
            ..ResidentGraphPrepareDiagnosticSnapshot::default()
        };
        snapshot.schema_winners_initial_htod_calls = 1;
        snapshot.schema_winners_initial_htod_bytes = 12;

        let line = resident_graph_prepare_diagnostic_line(Some(snapshot))
            .expect("enabled diagnostics produce one line");
        for expected in [
            "sample=17",
            "total_ns=101",
            "admission_and_source_snapshot_ns=102",
            "execution_domain_and_build_setup_ns=109",
            "metadata_binding_construction_ns=110",
            "reservation_validation_and_release_ns=111",
            "relation_device_allocation_calls=103",
            "workspace_provider_calls=104",
            "metadata_initial_htod_calls=105",
            "metadata_initial_htod_bytes=106",
            "schema_winners_initial_htod_calls=1",
            "schema_winners_initial_htod_bytes=12",
            "graph_instantiate_ns=107",
            "unattributed_ns=108",
        ] {
            assert!(line.split_ascii_whitespace().any(|field| field == expected));
        }
        assert_eq!(line.matches("sample=17").count(), 1);
        assert!(!line.contains("admission_source_validation_ns="));
        assert_eq!(resident_graph_prepare_diagnostic_line(None), None);
    }

    #[test]
    fn certificate_binds_strata_vector_length() {
        assert_mutation_rejected(|plan| {
            plan.strata.push(Stratum {
                id: 3,
                sccs: Vec::new(),
            });
        });
    }

    #[test]
    fn certificate_binds_empty_stratum_position() {
        assert_mutation_rejected(|plan| plan.strata.swap(1, 2));
    }

    #[test]
    fn certificate_binds_stratum_id() {
        assert_mutation_rejected(|plan| plan.strata[1].id = 99);
    }

    #[test]
    fn certificate_binds_ordered_stratum_scc_membership() {
        assert_mutation_rejected(|plan| plan.strata[2].sccs.swap(0, 1));
    }

    #[test]
    fn certificate_binds_scc_vector_length_including_empty_sccs() {
        assert_mutation_rejected(|plan| {
            plan.sccs.pop();
        });
    }

    #[test]
    fn certificate_binds_scc_id() {
        assert_mutation_rejected(|plan| plan.sccs[1].id = 99);
    }

    #[test]
    fn certificate_binds_ordered_scc_predicate_membership() {
        assert_mutation_rejected(|plan| plan.sccs[0].predicates.swap(0, 1));
    }

    #[test]
    fn certificate_binds_scc_recursive_flag() {
        assert_mutation_rejected(|plan| plan.sccs[1].is_recursive = true);
    }

    #[test]
    fn certificate_binds_rules_by_scc_vector_length() {
        assert_mutation_rejected(|plan| plan.rules_by_scc.push(Vec::new()));
    }

    #[test]
    fn certificate_binds_rule_occurrence_count() {
        assert_mutation_rejected(|plan| {
            plan.rules_by_scc[0].pop();
        });
    }

    #[test]
    fn certificate_binds_rule_order() {
        assert_mutation_rejected(|plan| plan.rules_by_scc[0].swap(0, 1));
    }

    #[test]
    fn certificate_binds_generated_query_provenance_omission() {
        let mut plan = representative_plan();
        plan.generated_query_rules = vec![GeneratedQueryRuleProvenance {
            query_index: 0,
            scc_index: 0,
            rule_index: 0,
        }];
        let certificate = ResidentGraphRouteCertificate::inspect(&plan, &catalog()).unwrap();
        plan.generated_query_rules.clear();
        assert!(!certificate.matches_plan(&plan).unwrap());
    }

    #[test]
    fn certificate_binds_generated_query_index() {
        assert_generated_query_provenance_mutation_rejected(|provenance| {
            provenance.query_index = 1;
        });
    }

    #[test]
    fn certificate_binds_generated_query_rule_position() {
        assert_generated_query_provenance_mutation_rejected(|provenance| {
            provenance.rule_index = 1;
        });
    }

    #[test]
    fn certificate_binds_rule_schema_type() {
        assert_mutation_rejected(|plan| {
            plan.rules_by_scc[0][0].meta.schema =
                schema(&[("left", ScalarType::U64), ("right", ScalarType::Symbol)]);
        });
    }

    #[test]
    fn certificate_binds_rule_schema_arity() {
        assert_mutation_rejected(|plan| {
            plan.rules_by_scc[0][0].meta.schema = schema(&[("left", ScalarType::U32)]);
        });
    }

    #[test]
    fn certificate_binds_primary_rir() {
        assert_mutation_rejected(|plan| plan.rules_by_scc[0][0].body = RirNode::Unit);
    }

    #[test]
    fn certificate_binds_captured_alternative_rir() {
        assert_mutation_rejected(|plan| {
            let RirNode::ChainJoin { fallback, .. } = &mut plan.rules_by_scc[0][1].body else {
                panic!("representative rule must carry a captured alternative");
            };
            **fallback = RirNode::Unit;
        });
    }

    #[test]
    fn certificate_binds_implicit_device_convergence_route() {
        let plan = representative_plan();
        let mut certificate = ResidentGraphRouteCertificate::inspect(&plan, &catalog()).unwrap();
        let convergence = certificate
            .covered_route_descriptors
            .iter()
            .find(|descriptor| {
                descriptor.contains("scc=0;rule=0;head=alpha;")
                    && descriptor.ends_with(";implicit=device_convergence")
            })
            .cloned()
            .expect("recursive rule must certify implicit device convergence");
        assert!(certificate.covered_route_descriptors.remove(&convergence));
        certificate
            .covered_route_descriptors
            .insert(format!("{convergence};mutated"));
        certificate.plan_fingerprint = stable_descriptor_fingerprint(
            certificate
                .covered_route_descriptors
                .iter()
                .map(String::as_bytes),
        );
        assert!(!certificate.matches_plan(&plan).unwrap());
    }

    #[test]
    fn certificate_binds_rel_arities_vector_length() {
        assert_mutation_rejected(|plan| {
            plan.rel_arities.insert(RelId(7), 3);
        });
    }

    #[test]
    fn certificate_binds_relation_arity() {
        assert_mutation_rejected(|plan| {
            plan.rel_arities.insert(RelId(1), 3);
        });
    }

    #[test]
    fn certificate_rel_arities_are_order_independent() {
        let plan = representative_plan();
        let certificate = ResidentGraphRouteCertificate::inspect(&plan, &catalog()).unwrap();
        let mut reordered = plan.clone();
        reordered.rel_arities.clear();
        reordered.rel_arities.insert(RelId(1), 2);
        reordered.rel_arities.insert(RelId(9), 0);
        assert!(certificate.matches_plan(&reordered).unwrap());
    }

    #[test]
    fn certificate_binds_estimated_memory_peak() {
        assert_mutation_rejected(|plan| plan.est_memory_peak += 1);
    }

    #[test]
    fn certificate_rejects_intermediate_arity_above_fixed_metadata_envelope() {
        let columns = (0..18)
            .map(|index| (format!("column_{index}"), ScalarType::U32))
            .collect::<Vec<_>>();
        let wide_schema = Schema::new(columns);
        let plan = ExecutionPlan {
            sccs: vec![Scc {
                id: 1,
                predicates: vec!["wide".to_owned()],
                is_recursive: false,
            }],
            strata: vec![Stratum {
                id: 0,
                sccs: vec![1],
            }],
            rules_by_scc: vec![vec![CompiledRule {
                head: "wide".to_owned(),
                body: RirNode::Scan { rel: RelId(1) },
                meta: RirMeta {
                    schema: wide_schema.clone(),
                    ..RirMeta::default()
                },
            }]],
            generated_query_rules: vec![],
            est_memory_peak: 0,
            rel_arities: HashMap::from([(RelId(1), 18)]),
        };
        let catalog = ResidentGraphSchemaCatalog::from_named_schemas([(
            "wide".to_owned(),
            RelId(1),
            wide_schema,
        )]);
        let certificate = ResidentGraphRouteCertificate::inspect(&plan, &catalog).unwrap();
        assert!(!certificate.is_supported());
        assert!(certificate.declines().iter().any(|decline| matches!(
            decline,
            ResidentGraphDeclineReason::UnsupportedNode {
                node: "intermediate_arity",
                ..
            }
        )));
    }

    #[test]
    fn certificate_accepts_intermediate_arity_at_fixed_metadata_envelope() {
        let columns = (0..17)
            .map(|index| (format!("column_{index}"), ScalarType::U32))
            .collect::<Vec<_>>();
        let wide_schema = Schema::new(columns);
        let plan = ExecutionPlan {
            sccs: vec![Scc {
                id: 1,
                predicates: vec!["wide".to_owned()],
                is_recursive: false,
            }],
            strata: vec![Stratum {
                id: 0,
                sccs: vec![1],
            }],
            rules_by_scc: vec![vec![CompiledRule {
                head: "wide".to_owned(),
                body: RirNode::Scan { rel: RelId(1) },
                meta: RirMeta {
                    schema: wide_schema.clone(),
                    ..RirMeta::default()
                },
            }]],
            generated_query_rules: vec![],
            est_memory_peak: 0,
            rel_arities: HashMap::from([(RelId(1), 17)]),
        };
        let catalog = ResidentGraphSchemaCatalog::from_named_schemas([(
            "wide".to_owned(),
            RelId(1),
            wide_schema,
        )]);
        let certificate = ResidentGraphRouteCertificate::inspect(&plan, &catalog).unwrap();
        assert!(certificate.is_supported(), "{:#?}", certificate.declines());
    }
}
