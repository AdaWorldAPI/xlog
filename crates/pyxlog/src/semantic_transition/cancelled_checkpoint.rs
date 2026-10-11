//! Original checkpoint custody for a positively retired, non-submitted instruction.

use super::*;

pyo3::create_exception!(
    pyxlog._native,
    SemanticTransitionCheckpointCancelled,
    PyRuntimeError,
    "The original instruction was cancelled without submission; its charged terminal checkpoint cannot resume execution."
);

pub(super) struct OriginalCancelledCheckpointCustody {
    manifest: SemanticCheckpointManifest,
    snapshot: AuthoritySnapshot,
    phase: CheckpointTaskPhase,
    binding: TaskCheckpointBinding,
    current_binding: TaskCheckpointBinding,
    issuance: TaskIssuance,
    parent: xlog_cuda::SemanticPublishedIdentity,
    source_material_digest: Identity256,
    source_checkpoint_digest: Identity256,
    completion: Mutex<CancelledCheckpointCompletion>,
}

#[derive(Default)]
struct CancelledCheckpointCompletion {
    assignments: Option<Arc<[u8]>>,
    sealed: Option<Arc<[u8]>>,
}

pub(super) struct CancelledCheckpointTerminal {
    pub(super) first_program_ordinal: u64,
    pub(super) end_program_ordinal: u64,
    pub(super) parent: xlog_cuda::SemanticPublishedIdentity,
    pub(super) proposal_spent: u64,
    pub(super) task_identity: [u8; 32],
    source_checkpoint_digest: Identity256,
}

fn material_digest(bytes: &[u8]) -> Identity256 {
    Identity256::from_bytes(Sha256::digest(bytes).into())
}

fn initial_index(admission: &OriginalActorRefreshAdmission) -> PyResult<Identity256> {
    let kit = admission.initial_kit.fields(2)?;
    let ColdValue::Bytes(index) = &kit[0] else {
        return Err(invalid("original replay index bytes are absent"));
    };
    Ok(material_digest(index))
}

fn require_seed_fields(original: &ColdValue, current: &ColdValue, terminal: bool) -> PyResult<()> {
    let original = original.fields(19)?;
    let current = current.fields(19)?;
    if original
        .iter()
        .zip(current)
        .enumerate()
        .any(|(index, (left, right))| !(terminal && matches!(index, 12 | 18)) && left != right)
    {
        return Err(invalid(
            "cancelled checkpoint differs from its original task capsule",
        ));
    }
    Ok(())
}

fn require_initial_prefill_source(
    manifest: &SemanticCheckpointManifest,
    phase: &CheckpointTaskPhase,
    binding: TaskCheckpointBinding,
) -> PyResult<()> {
    let has_initial_prefill = SemanticTransitionSession::verify_checkpoint_initial_prefill_source(
        &manifest.initial_prefill,
        &manifest.native,
        Identity256::from_bytes(binding.identity),
        binding.epoch,
        binding.content_identity(),
        *phase == CheckpointTaskPhase::InitialPrefillBound,
    )
    .map_err(xlog_err)?;
    if has_initial_prefill != (*phase == CheckpointTaskPhase::InitialPrefillBound)
        && !matches!(phase, CheckpointTaskPhase::Segment(_))
    {
        return Err(invalid(
            "checkpoint initial prefill differs from its task phase",
        ));
    }
    Ok(())
}

fn require_session_checkpoint_custody(
    session: &PySemanticTransitionSession,
    manifest: &SemanticCheckpointManifest,
    bytes: &[u8],
) -> PyResult<()> {
    let retained = session.checkpoint_custody()?;
    let custody = retained
        .as_ref()
        .ok_or_else(|| invalid("checkpoint source lost its original Session signing owner"))?;
    let digest: [u8; 32] = Sha256::digest(bytes).into();
    if Some(digest) != custody.origin && custody.seal(manifest)? != manifest.phase_seal {
        return Err(invalid(
            "checkpoint source differs from this Session's authenticated custody",
        ));
    }
    Ok(())
}

impl OriginalCancelledCheckpointCustody {
    /// Validate the pre-request capsule before the instruction can charge or hand off.
    pub(super) fn prepare(
        py: Python<'_>,
        session: &PySemanticTransitionSession,
        task: &PySemanticTransitionTaskUse,
        parent: &PySemanticPublishedParent,
        bytes: &[u8],
    ) -> PyResult<Self> {
        session.require_creator()?;
        parent.require_task(py, task)?;
        let manifest = SemanticCheckpointManifest::decode(bytes)?;
        require_session_checkpoint_custody(session, &manifest, bytes)?;
        let (seed, snapshot, phase, binding) = TaskCheckpointSeed::decode(&manifest.task)?;
        require_initial_prefill_source(&manifest, &phase, binding)?;
        let config = SemanticCheckpointSessionConfig::decode(&manifest.session)?;
        if config.capacities != session.capacities
            || config.admission_limits != session.admission_limits
            || config.memory_bytes != session.memory_bytes
        {
            return Err(invalid(
                "cancelled checkpoint has another original Session reservation",
            ));
        }
        let expense = seed.proposal_expense()?;
        if expense.original_admission.is_none() {
            return Err(invalid(
                "cancelled checkpoint requires the original complete Program",
            ));
        }
        if terminal_checkpoint(&manifest, &seed, &phase, binding)?.is_some() {
            return Err(invalid(
                "a terminal cancelled checkpoint cannot admit another instruction",
            ));
        }
        let state = task.state()?;
        if checkpoint_task_phase(&state)? != phase {
            return Err(invalid(
                "cancelled checkpoint has another original task phase",
            ));
        }
        if state.snapshot.canonical != snapshot.canonical {
            state.snapshot.newer_than(&snapshot)?;
        }
        drop(state);
        let owner = session.owner()?;
        task.require_current(&owner)?;
        let current_binding = TaskCheckpointBinding::from_owner(&owner)?;
        if !binding.same_task_content(current_binding) {
            return Err(invalid(
                "cancelled checkpoint has another native task binding",
            ));
        }
        // The original restore owner, not Python digest equivalence, authenticates
        // a saved instance when the selected parent was created by fresh restore.
        let lease = parent.lease()?;
        let (source_material_digest, restored_from_exact_source) = owner
            .verify_checkpoint_source_material(&lease, &manifest.native)
            .map_err(xlog_err)?;
        if binding.epoch != current_binding.epoch && !restored_from_exact_source {
            return Err(invalid(
                "cancelled checkpoint has another original native task epoch",
            ));
        }
        let selected = owner.published_identity(&lease).map_err(xlog_err)?;
        let current_task = task
            .checkpoint
            .encode(&owner, &snapshot, &phase, &binding)?;
        let seed_value = checkpoint_cold_value(&manifest.task, manifest.task.len())?;
        require_seed_fields(
            &seed_value,
            &checkpoint_cold_value(&current_task, current_task.len())?,
            false,
        )?;
        Ok(Self {
            manifest,
            snapshot,
            phase,
            binding,
            current_binding,
            issuance: task.issuance.clone(),
            parent: selected,
            source_material_digest,
            source_checkpoint_digest: material_digest(bytes),
            completion: Mutex::new(CancelledCheckpointCompletion::default()),
        })
    }

    /// Attach the already validated source to this native issuance before any charge.
    pub(super) fn bind_source(
        &self,
        owner: &mut SemanticTransitionSession,
        admission: &SemanticSegmentInstructionAdmission,
        parent: &SemanticPublishedLease,
    ) -> PyResult<()> {
        if owner.published_identity(parent).map_err(xlog_err)? != self.parent {
            return Err(invalid(
                "cancelled checkpoint source parent changed before admission",
            ));
        }
        let (source, _, _, _) = TaskCheckpointSeed::decode(&self.manifest.task)?;
        let source_expense = source.proposal_expense()?;
        let source_assignments = source.actor_refresh_assignments.as_deref().ok_or_else(|| {
            invalid("cancelled checkpoint lost its original native source assignments")
        })?;
        owner
            .bind_segment_checkpoint_source(
                admission,
                parent,
                &self.manifest.native,
                self.source_checkpoint_digest,
                source_expense.spent,
                source_assignments,
            )
            .map_err(xlog_err)
    }

    /// Freeze only after the original native cancellation and all final-use reports.
    pub(super) fn seal(
        &self,
        session: &PySemanticTransitionSession,
        task: &PySemanticTransitionTaskUse,
        owner: &mut SemanticTransitionSession,
        admission: &SemanticSegmentInstructionAdmission,
    ) -> PyResult<Arc<[u8]>> {
        session.require_creator()?;
        let mut completion = self
            .completion
            .lock()
            .map_err(|_| invalid("cancelled checkpoint custody mutex is poisoned"))?;
        if let Some(sealed) = &completion.sealed {
            return Ok(Arc::clone(sealed));
        }
        if !self.issuance.same_as(&task.issuance) {
            return Err(invalid(
                "cancelled checkpoint requires its original issued task",
            ));
        }
        task.require_original_identity(owner)?;
        if TaskCheckpointBinding::from_owner(owner)? != self.current_binding {
            return Err(invalid(
                "cancelled checkpoint native task changed during retirement",
            ));
        }
        if completion.assignments.is_none() {
            let assignments = owner
                .cancelled_segment_assignment_material(admission)
                .map_err(xlog_err)?;
            // Retain the native terminal disposition before any fallible encoding.
            completion.assignments = Some(Arc::from(assignments));
        }
        let task_bytes =
            task.checkpoint
                .encode(owner, &self.snapshot, &self.phase, &self.binding)?;
        let task_value = checkpoint_cold_value(&task_bytes, task_bytes.len())?;
        require_seed_fields(
            &checkpoint_cold_value(&self.manifest.task, self.manifest.task.len())?,
            &task_value,
            true,
        )?;
        let fields = task_value.fields(19)?;
        let ColdValue::Bytes(assignments) = &fields[18] else {
            return Err(invalid(
                "cancelled checkpoint has no original native assignment material",
            ));
        };
        if completion.assignments.as_deref() != Some(assignments.as_ref()) {
            return Err(invalid(
                "cancelled checkpoint native assignment material changed while sealing",
            ));
        }
        let (seed, _, _, binding) = TaskCheckpointSeed::decode(&task_bytes)?;
        let terminal_manifest = SemanticCheckpointManifest {
            native: self.manifest.native.clone(),
            model: self.manifest.model.clone(),
            task: task_bytes,
            session: self.manifest.session.clone(),
            initial_prefill: self.manifest.initial_prefill.clone(),
            phase_seal: Vec::new(),
        };
        let terminal = terminal_checkpoint(&terminal_manifest, &seed, &self.phase, binding)?
            .ok_or_else(|| invalid("native cancellation has no complete terminal disposition"))?;
        if terminal.parent != self.parent
            || material_digest(&terminal_manifest.native) != self.source_material_digest
            || terminal.source_checkpoint_digest != self.source_checkpoint_digest
        {
            return Err(invalid(
                "cancelled checkpoint lost its original native source custody",
            ));
        }
        let sealed: Arc<[u8]> = Arc::from(session.seal_owned_checkpoint(terminal_manifest)?);
        completion.sealed = Some(Arc::clone(&sealed));
        Ok(sealed)
    }

    pub(super) fn sealed_bytes(&self) -> PyResult<Option<Arc<[u8]>>> {
        self.completion
            .lock()
            .map(|completion| completion.sealed.as_ref().map(Arc::clone))
            .map_err(|_| invalid("cancelled checkpoint custody mutex is poisoned"))
    }
}

/// The normal and lightweight restore entries use this same metadata-only check.
pub(super) fn terminal_checkpoint(
    manifest: &SemanticCheckpointManifest,
    seed: &TaskCheckpointSeed,
    phase: &CheckpointTaskPhase,
    binding: TaskCheckpointBinding,
) -> PyResult<Option<CancelledCheckpointTerminal>> {
    let expense = seed.proposal_expense()?;
    let Some(original) = expense.original_admission else {
        return Ok(None);
    };
    let assignments = seed
        .actor_refresh_assignments
        .as_deref()
        .ok_or_else(|| invalid("checkpoint original native assignment material is absent"))?;
    let terminal = SemanticTransitionSession::verify_cancelled_segment_assignment_material(
        &original.program,
        initial_index(&original)?,
        assignments,
    )
    .map_err(xlog_err)?;
    let Some((first, end, parent, proposal_spent, source_digest, source_checkpoint_digest)) =
        terminal
    else {
        return Ok(None);
    };
    if proposal_spent != expense.spent || source_digest != material_digest(&manifest.native) {
        return Err(invalid(
            "cancelled checkpoint source or charged Proposal expense is inconsistent",
        ));
    }
    SemanticTransitionSession::state_material_projection(&manifest.native).map_err(xlog_err)?;
    SemanticCheckpointSessionConfig::decode(&manifest.session)?;
    require_initial_prefill_source(manifest, phase, binding)?;
    Ok(Some(CancelledCheckpointTerminal {
        first_program_ordinal: first,
        end_program_ordinal: end,
        parent,
        proposal_spent,
        task_identity: binding.identity,
        source_checkpoint_digest,
    }))
}

pub(super) fn terminal_error(
    py: Python<'_>,
    checkpoint: &[u8],
    terminal: &CancelledCheckpointTerminal,
) -> PyResult<PyErr> {
    let error = SemanticTransitionCheckpointCancelled::new_err(
        "the original non-submitted instruction is durably cancelled; no further request can be admitted",
    );
    let value = error.value(py);
    value.setattr("checkpoint", PyBytes::new(py, checkpoint))?;
    value.setattr("task_identity", PyBytes::new(py, &terminal.task_identity))?;
    value.setattr("first_program_ordinal", terminal.first_program_ordinal)?;
    value.setattr("end_program_ordinal", terminal.end_program_ordinal)?;
    value.setattr("proposal_spent", terminal.proposal_spent)?;
    value.setattr(
        "parent_identity",
        (
            PyBytes::new(py, terminal.parent.instance.as_bytes()),
            terminal.parent.word,
            PyBytes::new(py, terminal.parent.logical_digest.as_bytes()),
            PyBytes::new(py, terminal.parent.state_digest.as_bytes()),
        ),
    )?;
    Ok(error)
}

#[pyfunction]
#[pyo3(signature = (checkpoint, *, program_bytes, task_identity, snapshot=None, learning_phase_issuer=None, learning_phase_records=None, learning_phase_record_limits=None, learning_phase_fence=None, learning_phase_checkpoint_issuer=None))]
#[expect(
    clippy::too_many_arguments,
    reason = "terminal recovery authenticates the original complete lifecycle without native execution"
)]
pub(crate) fn restore_cancelled_checkpoint(
    py: Python<'_>,
    checkpoint: &Bound<'_, PyAny>,
    program_bytes: &Bound<'_, PyAny>,
    task_identity: &Bound<'_, PyAny>,
    snapshot: Option<&Bound<'_, PyAny>>,
    learning_phase_issuer: Option<&Bound<'_, PyAny>>,
    learning_phase_records: Option<&Bound<'_, PyAny>>,
    learning_phase_record_limits: Option<&Bound<'_, PyAny>>,
    learning_phase_fence: Option<&Bound<'_, PyAny>>,
    learning_phase_checkpoint_issuer: Option<&Bound<'_, PyAny>>,
) -> PyResult<()> {
    let raw = exact_replay_bytes(checkpoint)?;
    let program = exact_replay_bytes(program_bytes)?;
    let identity: [u8; 32] = exact_replay_bytes(task_identity)?.try_into().map_err(|_| {
        invalid("cancelled checkpoint requires its original native task identity32")
    })?;
    let manifest = SemanticCheckpointManifest::decode(raw)?;
    let (seed, _, phase, binding) = TaskCheckpointSeed::decode(&manifest.task)?;
    let expense = seed.proposal_expense()?;
    if expense
        .original_admission
        .as_ref()
        .map(|original| original.program.as_ref())
        != Some(program)
        || binding.identity != identity
    {
        return Err(invalid(
            "cancelled checkpoint differs from the expected original Program or native task",
        ));
    }
    let _closure = learning_phase::cold_restore::read_checkpoint_restore_closure(
        raw,
        snapshot,
        learning_phase_issuer,
        learning_phase_records,
        learning_phase_record_limits,
        learning_phase_fence,
        learning_phase_checkpoint_issuer,
    )?;
    let terminal = terminal_checkpoint(&manifest, &seed, &phase, binding)?
        .ok_or_else(|| invalid("checkpoint is not a positively completed cancelled terminal"))?;
    Err(terminal_error(py, raw, &terminal)?)
}
