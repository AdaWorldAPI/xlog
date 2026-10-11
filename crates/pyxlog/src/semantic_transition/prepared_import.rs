use super::*;

#[derive(Clone, Copy, PartialEq, Eq)]
enum ImportPreparationStage {
    MetadataEntering,
    Prepared,
    ExecutionEntering,
    Complete,
    CancellationEntering,
    Cancelled,
}

/// Immutable metadata and the actual Python producer of an original tensor port.
pub(super) struct OriginalImportTensorPort {
    layout: SemanticTensorLayout,
    logical_begin: u64,
    logical_end: u64,
    native_allocation: Option<DeviceAllocationProvenance>,
    producer: Py<PyAny>,
}

impl OriginalImportTensorPort {
    pub(super) fn retain(
        py: Python<'_>,
        input: &SemanticTensorInput,
        producer: &Py<PyAny>,
    ) -> Self {
        Self::retain_geometry(
            py,
            &(
                input.layout,
                input.logical_begin,
                input.logical_end,
                input.native_allocation.clone(),
            ),
            producer,
        )
    }

    pub(super) fn retain_geometry(
        py: Python<'_>,
        geometry: &(
            SemanticTensorLayout,
            u64,
            u64,
            Option<DeviceAllocationProvenance>,
        ),
        producer: &Py<PyAny>,
    ) -> Self {
        Self {
            layout: geometry.0,
            logical_begin: geometry.1,
            logical_end: geometry.2,
            native_allocation: geometry.3.clone(),
            producer: producer.clone_ref(py),
        }
    }

    fn clone_ref(&self, py: Python<'_>) -> Self {
        Self {
            layout: self.layout,
            logical_begin: self.logical_begin,
            logical_end: self.logical_end,
            native_allocation: self.native_allocation.clone(),
            producer: self.producer.clone_ref(py),
        }
    }

    fn export(
        &self,
        py: Python<'_>,
        device: usize,
        stream: u64,
        check: &dyn Fn() -> PyResult<()>,
    ) -> PyResult<SemanticTensorInput> {
        let producer = self.producer.bind(py);
        validate_producer_device_guarded(producer, device, check)?;
        let tensor = crate::dlpack_from_py_for_stream_guarded(
            producer,
            i64::try_from(stream)
                .map_err(|_| invalid("consumer stream exceeds DLPack address space"))?,
            check,
        )?;
        Ok(SemanticTensorInput {
            layout: self.layout,
            logical_begin: self.logical_begin,
            logical_end: self.logical_end,
            tensor,
            native_allocation: self.native_allocation.clone(),
        })
    }
}

/// The original SOURCE import suspended before its numerical invocation.
#[pyclass(name = "SemanticPreparedTaskImport", module = "pyxlog._native", frozen)]
pub(crate) struct PySemanticPreparedTaskImport {
    pub(super) session: Py<PySemanticTransitionSession>,
    pub(super) task_use: Py<PySemanticTransitionTaskUse>,
    pub(super) parent: Py<PySemanticPublishedParent>,
    controller_identity: Arc<()>,
    pub(super) ordinal: usize,
    pub(super) material: Arc<xlog_cuda::SemanticReplayMaterial>,
    original_decisions: Arc<[u8]>,
    pub(super) replay_row: SemanticTrainingViewRow,
    expected: (String, Vec<u8>),
    prepare_invocation: Py<PyAny>,
    restore_invocation: Py<PyAny>,
    pack_policy: Py<PyAny>,
    finish_invocation: Py<PyAny>,
    refresh_snapshot: Py<PyAny>,
    stage: Mutex<ImportPreparationStage>,
    member: Mutex<Option<Py<PySemanticRetainedReplayMember>>>,
    completed: Mutex<Option<Py<PyAny>>>,
    cancelled_final_use: Mutex<Option<ReplayImportFinalization>>,
    source_ports: Mutex<Option<Vec<OriginalImportTensorPort>>>,
    source_witness: Mutex<Option<Py<PySemanticTensorContentWitness>>>,
    pub(super) group_use: Arc<Mutex<RetainedGroupUse>>,
}

impl PySemanticPreparedTaskImport {
    pub(super) fn prepare(
        controller: &PySemanticTransitionController,
        py: Python<'_>,
        task_use: &Py<PySemanticTransitionTaskUse>,
        ordinal: usize,
        material: NativeReplayBinding,
        prepare_invocation: &Bound<'_, PyAny>,
        restore_invocation: &Bound<'_, PyAny>,
        pack_policy: &Bound<'_, PyAny>,
        finish_invocation: &Bound<'_, PyAny>,
        refresh_snapshot: &Bound<'_, PyAny>,
        import: &mut ColdImportGuard<'_>,
    ) -> PyResult<Py<Self>> {
        let issued = task_use.borrow(py);
        let replay_row = controller.original_replay_import_row(py, &issued, ordinal, import)?;
        let (parent, expected) = controller.acquire_replay_import_parent(
            py,
            task_use,
            SemanticTransitionKind::Proposal,
            import,
        )?;
        let prepared = Py::new(
            py,
            Self {
                session: controller.session.clone_ref(py),
                task_use: task_use.clone_ref(py),
                parent,
                controller_identity: Arc::clone(&controller.identity),
                ordinal,
                material: Arc::new(material.material),
                original_decisions: material.original_decisions,
                replay_row: replay_row.training_view_row()?,
                expected,
                prepare_invocation: prepare_invocation.clone().unbind(),
                restore_invocation: restore_invocation.clone().unbind(),
                pack_policy: pack_policy.clone().unbind(),
                finish_invocation: finish_invocation.clone().unbind(),
                refresh_snapshot: refresh_snapshot.clone().unbind(),
                stage: Mutex::new(ImportPreparationStage::MetadataEntering),
                member: Mutex::new(None),
                completed: Mutex::new(None),
                cancelled_final_use: Mutex::new(Some(ReplayImportFinalization::default())),
                source_ports: Mutex::new(None),
                source_witness: Mutex::new(None),
                group_use: Arc::new(Mutex::new(RetainedGroupUse::new())),
            },
        )?;
        {
            let mut retained = import
                .session
                .prepared_task_import
                .lock()
                .map_err(|_| invalid("prepared task import custody is poisoned"))?;
            if retained.is_some() {
                return Err(invalid("prepared import cannot replace an original owner"));
            }
            *retained = Some(prepared.clone_ref(py));
        }
        if let Some(child) = import.session.private_replay_custody()? {
            child.retain_prepared_import(py, &prepared)?;
        }
        // Retain the original issuance and reader even if the callback or its
        // Python return is lost. Preparation is not a completed import.
        import.retained_preparation = true;
        import.deferred = true;
        let original = prepared.borrow(py);
        let acquired = original.parent.borrow(py);
        // Allocate the original future domain owners only after recovery custody
        // is installed, before metadata preparation freezes their real geometry.
        import
            .session
            .owner()?
            .prepare_policy_numerical_domain_storage(&*acquired.lease()?)
            .map_err(xlog_err)?;
        {
            let _reads = ImportReadScope::enter(&issued)?;
            if !original
                .prepare_invocation
                .bind(py)
                .call1((task_use.clone_ref(py), original.parent.clone_ref(py)))?
                .is_none()
            {
                return Err(invalid(
                    "import metadata preparation must retain original ports and return None",
                ));
            }
        }
        controller.check_import_callback(py, &issued, &acquired, import, &original.expected)?;
        *original
            .stage
            .lock()
            .map_err(|_| invalid("prepared import stage is poisoned"))? =
            ImportPreparationStage::Prepared;
        drop(acquired);
        drop(original);
        Ok(prepared)
    }

    fn require_controller(
        &self,
        py: Python<'_>,
        controller: &PySemanticTransitionController,
    ) -> PyResult<()> {
        let session = self.session.borrow(py);
        session.require_creator()?;
        if controller.session.as_ptr() != self.session.as_ptr()
            || !Arc::ptr_eq(&controller.identity, &self.controller_identity)
            || self.parent.borrow(py).task_use.as_ptr() != self.task_use.as_ptr()
        {
            return Err(invalid(
                "prepared import requires its original controller, task and SOURCE reader",
            ));
        }
        Ok(())
    }

    pub(super) fn require_original(
        &self,
        py: Python<'_>,
        controller: &PySemanticTransitionController,
    ) -> PyResult<()> {
        self.require_controller(py, controller)?;
        let session = self.session.borrow(py);
        if !session.importing.load(Ordering::Acquire) {
            return Err(invalid(
                "unexecuted preparation requires its original active SOURCE import",
            ));
        }
        let owner = session.owner()?;
        self.task_use.borrow(py).require_original_identity(&owner)?;
        Ok(())
    }

    pub(super) fn require_completed(&self) -> PyResult<()> {
        if *self
            .stage
            .lock()
            .map_err(|_| invalid("prepared import stage is poisoned"))?
            != ImportPreparationStage::Complete
        {
            return Err(invalid(
                "future historical consumer requires its genuine completed original import",
            ));
        }
        Ok(())
    }

    pub(super) fn require_ready(&self, py: Python<'_>) -> PyResult<()> {
        self.require_metadata(py)?;
        if *self
            .stage
            .lock()
            .map_err(|_| invalid("prepared import stage is poisoned"))?
            != ImportPreparationStage::Prepared
        {
            return Err(invalid(
                "original metadata preparation has not genuinely returned",
            ));
        }
        Ok(())
    }

    pub(super) fn require_cancellation_preparation(&self, py: Python<'_>) -> PyResult<()> {
        let session = self.session.borrow(py);
        session.require_creator()?;
        if !matches!(
            *self
                .stage
                .lock()
                .map_err(|_| invalid("prepared import stage is poisoned"))?,
            ImportPreparationStage::Prepared | ImportPreparationStage::CancellationEntering
        ) || self
            .member
            .lock()
            .map_err(|_| invalid("prepared import member is poisoned"))?
            .is_some()
            || self.parent.borrow(py).task_use.as_ptr() != self.task_use.as_ptr()
            || self.parent.borrow(py).session.as_ptr() != self.session.as_ptr()
            || !Arc::ptr_eq(
                &self.task_use.borrow(py).controller,
                &self.controller_identity,
            )
            || session
                .prepared_task_import
                .lock()
                .map_err(|_| invalid("prepared import custody is poisoned"))?
                .as_ref()
                .is_none_or(|original| !std::ptr::eq(self, &*original.borrow(py)))
        {
            return Err(invalid(
                "unexecuted cancellation requires its original genuinely prepared SOURCE import",
            ));
        }
        Ok(())
    }

    pub(super) fn require_cancelled_ack(&self, py: Python<'_>) -> PyResult<()> {
        self.session.borrow(py).require_creator()?;
        let progress = self
            .cancelled_final_use
            .try_lock()
            .map_err(|_| invalid("original unexecuted cancellation is active or poisoned"))?;
        if *self
            .stage
            .lock()
            .map_err(|_| invalid("prepared import stage is poisoned"))?
            != ImportPreparationStage::Cancelled
            || progress.as_ref().is_none_or(|progress| {
                !progress.predecessor_released
                    || progress.streams.is_none()
                    || progress.native_pending
            })
            || self
                .member
                .lock()
                .map_err(|_| invalid("prepared import member is poisoned"))?
                .is_some()
            || self.parent.borrow(py).task_use.as_ptr() != self.task_use.as_ptr()
            || self.parent.borrow(py).session.as_ptr() != self.session.as_ptr()
            || !Arc::ptr_eq(
                &self.task_use.borrow(py).controller,
                &self.controller_identity,
            )
        {
            return Err(invalid(
                "unexecuted cancellation has no positive original reader-release acknowledgment",
            ));
        }
        Ok(())
    }

    pub(super) fn finish_original_cancellation(&self, py: Python<'_>) -> PyResult<()> {
        if self.require_cancelled_ack(py).is_ok() {
            return Ok(());
        }
        self.require_cancellation_preparation(py)?;
        let session = self.session.borrow(py);
        let child = session.private_replay_custody()?.ok_or_else(|| {
            invalid("unexecuted cancellation lost its original instruction child")
        })?;
        child.require_cancelled_prepared_import(py, self)?;
        let deliveries = {
            let group = self
                .group_use
                .lock()
                .map_err(|_| invalid("prepared import group use is poisoned"))?;
            if group.completion.is_some() || group.banks != [false; 2] {
                return Err(invalid(
                    "unexecuted cancellation cannot replace an actual target contribution",
                ));
            }
            group
                .deliveries
                .iter()
                .flatten()
                .map(|delivery| delivery.clone_ref(py))
                .collect::<Vec<_>>()
        };
        for delivery in &deliveries {
            let delivery = delivery.borrow(py);
            delivery.require_removed_hooks()?;
            if delivery.activated_once.load(Ordering::Acquire)
                || delivery
                    .handles
                    .lock()
                    .map_err(|_| invalid("cancelled gradient hook custody is poisoned"))?
                    .is_some()
                || !delivery
                    .restores
                    .lock()
                    .map_err(|_| invalid("cancelled gradient restoration custody is poisoned"))?
                    .is_empty()
            {
                return Err(invalid(
                    "unexecuted cancellation still owns active original gradient delivery",
                ));
            }
        }
        let mut progress = self
            .cancelled_final_use
            .try_lock()
            .map_err(|_| invalid("original unexecuted cancellation is active or poisoned"))?
            .take()
            .ok_or_else(|| invalid("original unexecuted cancellation is already active"))?;
        *self
            .stage
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner) =
            ImportPreparationStage::CancellationEntering;
        let issued = self.task_use.borrow(py);
        let parent = self.parent.borrow(py);
        let mut import = ColdImportGuard {
            session: &session,
            issuance: issued.issuance.clone(),
            original_decisions: Some(Arc::clone(&self.original_decisions)),
            committed: false,
            deferred: false,
            retained_final_use: true,
            retained_preparation: true,
            native_pending: std::cell::Cell::new(false),
        };
        let result = (|| -> PyResult<()> {
            let streams = PySemanticTransitionController::finish_original_import_invocation(
                py,
                &self.task_use,
                &self.parent,
                &issued,
                self.finish_invocation.bind(py),
                &mut progress,
            )?;
            // Target final use and the one original cleanup have returned.
            // The retired witness must leave before native reader retirement;
            // real producer backings and pending release custody remain held.
            self.retire_source_witness();
            if !progress.predecessor_released {
                let mut owner = session.owner()?;
                issued.require_original_identity(&owner)?;
                let mut lease = parent.lease()?;
                let result = owner.release(&mut lease, &streams);
                let pending = result.as_ref().is_err_and(|error| {
                    matches!(error, xlog_cuda::SemanticTransitionError::Runtime { .. })
                }) && owner
                    .published_reader_release_pending(&lease)
                    .map_err(xlog_err)?;
                drop(lease);
                drop(owner);
                PySemanticTransitionController::finish_import_native_stage(
                    py,
                    &import,
                    result,
                    pending,
                    &mut progress.native_pending,
                )?;
                parent.release_continuation_producers();
                progress.predecessor_released = true;
            }
            issued.state()?.phase = TaskUsePhase::Refused;
            session.importing.store(false, Ordering::Release);
            Ok(())
        })();
        *self
            .cancelled_final_use
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner) = Some(progress);
        if result.is_ok() {
            import.committed = true;
            let retired_group = {
                let mut group = self
                    .group_use
                    .lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner);
                (
                    group.update.take(),
                    group.target.take(),
                    std::mem::replace(&mut group.deliveries, std::array::from_fn(|_| None)),
                )
            };
            self.retire_source_port_custody(py, &retired_group.2);
            *self
                .stage
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner) =
                ImportPreparationStage::Cancelled;
            drop(retired_group);
        }
        result
    }

    pub(super) fn require_future_roster(&self, py: Python<'_>) -> PyResult<()> {
        if self
            .source_ports
            .lock()
            .map_err(|_| invalid("SOURCE model ports are poisoned"))?
            .is_none()
            || self
                .source_witness
                .lock()
                .map_err(|_| invalid("SOURCE model witness is poisoned"))?
                .is_none()
        {
            return Err(invalid("historical replay preparation requires its complete original SOURCE model ports and witness"));
        }
        let group = self
            .group_use
            .lock()
            .map_err(|_| invalid("prepared import group use is poisoned"))?;
        if group.gradient_banks != [true; 2]
            || group.edit_only != Some(true)
            || group.banks != [false; 2]
            || group.target.is_none()
        {
            return Err(invalid("historical replay requires both original future Edit bank destinations before any effect"));
        }
        let target = group.target.as_ref().expect("validated original target");
        for (bank, delivery) in group.deliveries.iter().enumerate() {
            let delivery = delivery
                .as_ref()
                .ok_or_else(|| invalid("future Edit roster lost its original bank delivery"))?
                .borrow(py);
            let original = delivery
                .future_import
                .as_ref()
                .ok_or_else(|| invalid("future Edit bank requires its original prepared import"))?;
            let (_, actual, actual_bank) = delivery
                .future_target
                .as_ref()
                .ok_or_else(|| invalid("future Edit bank lost its original target"))?;
            if !std::ptr::eq(self, &*original.borrow(py))
                || *actual_bank != bank
                || !actual.same_handle(target)
                || delivery.activated_once.load(Ordering::Acquire)
            {
                return Err(invalid(
                    "future Edit roster changed its original import, Update or bank",
                ));
            }
        }
        Ok(())
    }

    pub(super) fn expected_binding(&self) -> &(String, Vec<u8>) {
        &self.expected
    }

    pub(super) fn owns_member(
        &self,
        py: Python<'_>,
        member: &PySemanticRetainedReplayMember,
    ) -> bool {
        self.member
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .as_ref()
            .is_some_and(|original| std::ptr::eq(member, &*original.borrow(py)))
    }

    pub(super) fn require_future_consumer(
        &self,
        py: Python<'_>,
        step: &PySemanticPreparedStep,
    ) -> PyResult<()> {
        let session = self.session.borrow(py);
        session.require_creator()?;
        if !session.importing.load(Ordering::Acquire)
            || step.session.as_ptr() == self.session.as_ptr()
            || *self
                .stage
                .lock()
                .map_err(|_| invalid("prepared import stage is poisoned"))?
                != ImportPreparationStage::Prepared
        {
            return Err(invalid(
                "future Edit delivery requires the original unexecuted distinct SOURCE import",
            ));
        }
        let retained = session
            .prepared_task_import
            .lock()
            .map_err(|_| invalid("prepared import custody is poisoned"))?;
        if retained
            .as_ref()
            .is_none_or(|original| !std::ptr::eq(self, &*original.borrow(py)))
        {
            return Err(invalid(
                "future Edit delivery changed its retained original import owner",
            ));
        }
        drop(retained);
        let child = session.private_replay_custody()?.ok_or_else(|| {
            invalid("future Edit delivery requires its original instruction-bound SOURCE import")
        })?;
        child.require_future_import_structure(py)?;
        child.require_target(py, step)?;
        let owner = session.owner()?;
        self.task_use.borrow(py).require_original_identity(&owner)?;
        owner
            .published_identity(&*self.parent.borrow(py).lease()?)
            .map_err(xlog_err)?;
        Ok(())
    }

    pub(super) fn require_source_port_owner(
        &self,
        py: Python<'_>,
        task_use: &PySemanticTransitionTaskUse,
        parent: &PySemanticPublishedParent,
    ) -> PyResult<()> {
        if !matches!(
            *self
                .stage
                .lock()
                .map_err(|_| invalid("prepared import stage is poisoned"))?,
            ImportPreparationStage::MetadataEntering | ImportPreparationStage::ExecutionEntering
        ) || !std::ptr::eq(task_use, &*self.task_use.borrow(py))
            || !std::ptr::eq(parent, &*self.parent.borrow(py))
            || parent.session.as_ptr() != self.session.as_ptr()
        {
            return Err(invalid(
                "SOURCE model ports changed their original import owner or stage",
            ));
        }
        Ok(())
    }

    pub(super) fn retain_source_ports(
        &self,
        py: Python<'_>,
        task_use: &PySemanticTransitionTaskUse,
        parent: &PySemanticPublishedParent,
        ports: Vec<OriginalImportTensorPort>,
    ) -> PyResult<()> {
        self.require_source_port_owner(py, task_use, parent)?;
        let mut original = self
            .source_ports
            .lock()
            .map_err(|_| invalid("SOURCE model ports are poisoned"))?;
        if original.is_some() {
            return Err(invalid("original SOURCE model ports were already retained"));
        }
        *original = Some(ports);
        Ok(())
    }

    pub(super) fn retain_source_witness(
        &self,
        py: Python<'_>,
        witness: Py<PySemanticTensorContentWitness>,
    ) -> PyResult<()> {
        let original = witness.borrow(py);
        if original.session.as_ptr() != self.session.as_ptr()
            || !matches!(&original.parent, ContentStepOwner::Published(parent) if parent.as_ptr() == self.parent.as_ptr())
            || !original.model_binding
            || !matches!(
                *self
                    .stage
                    .lock()
                    .map_err(|_| invalid("prepared import stage is poisoned"))?,
                ImportPreparationStage::MetadataEntering
                    | ImportPreparationStage::ExecutionEntering
            )
            || self
                .source_ports
                .lock()
                .map_err(|_| invalid("SOURCE model ports are poisoned"))?
                .is_none()
        {
            return Err(invalid(
                "SOURCE model witness changed its original authenticated owner",
            ));
        }
        drop(original);
        let mut retained = self
            .source_witness
            .lock()
            .map_err(|_| invalid("SOURCE model witness is poisoned"))?;
        if retained.is_some() {
            return Err(invalid(
                "original SOURCE model witness was already retained",
            ));
        }
        *retained = Some(witness);
        Ok(())
    }

    fn complete_future_deliveries(
        &self,
        py: Python<'_>,
        member: &PySemanticRetainedReplayMember,
        check: &dyn Fn() -> PyResult<()>,
    ) -> PyResult<()> {
        let deliveries = self
            .group_use
            .lock()
            .map_err(|_| invalid("prepared import group use is poisoned"))?
            .deliveries
            .iter()
            .flatten()
            .map(|delivery| delivery.clone_ref(py))
            .collect::<Vec<_>>();
        if deliveries.is_empty() {
            return Ok(());
        }
        let _witness = self
            .source_witness
            .lock()
            .map_err(|_| invalid("SOURCE model witness is poisoned"))?
            .as_ref()
            .map(|witness| witness.clone_ref(py))
            .ok_or_else(|| {
                invalid("future Edit completion lacks its original authenticated SOURCE witness")
            })?;
        let ports = self
            .source_ports
            .lock()
            .map_err(|_| invalid("SOURCE model ports are poisoned"))?
            .as_ref()
            .ok_or_else(|| invalid("future Edit completion lost its original SOURCE ports"))?
            .iter()
            .map(|port| port.clone_ref(py))
            .collect::<Vec<_>>();
        let leaves = deliveries
            .iter()
            .map(|delivery| {
                let delivery = delivery.borrow(py);
                let leaves = delivery
                    .future_leaves
                    .lock()
                    .map_err(|_| invalid("future Edit physical leaf custody is poisoned"))?;
                Ok(leaves
                    .iter()
                    .map(|leaf| leaf.clone_ref(py))
                    .collect::<Vec<_>>())
            })
            .collect::<PyResult<Vec<_>>>()?;
        // Capture each shared physical leaf's restored slot once, including
        // frozen leaves with no AccumulateGrad hook. Both banks use this same
        // snapshot; SOURCE layout/provenance comes only from its real witness.
        let mut snapshots = BTreeMap::<usize, (SemanticTensorLayout, Py<PyAny>)>::new();
        let mut original_leaves = None;
        for leaves in &leaves {
            let mut bank_leaves = BTreeSet::new();
            for leaf in leaves {
                let key = leaf.producer.as_ptr() as usize;
                if !bank_leaves.insert(key)
                    || !ports.iter().any(|port| {
                        port.layout == leaf.layout
                            && port.producer.as_ptr() == leaf.producer.as_ptr()
                    })
                {
                    return Err(invalid(
                        "future Edit leaf is not its original authenticated SOURCE producer",
                    ));
                }
                if let Some((layout, _)) = snapshots.get(&key) {
                    if *layout != leaf.layout {
                        return Err(invalid(
                            "future Edit banks changed the original physical leaf metadata",
                        ));
                    }
                    continue;
                }
                check()?;
                let gradient = leaf.producer.bind(py).getattr("grad")?.unbind();
                check()?;
                snapshots.insert(key, (leaf.layout, gradient));
            }
            if original_leaves
                .as_ref()
                .is_some_and(|original| original != &bank_leaves)
            {
                return Err(invalid(
                    "future Edit banks changed the full original physical leaf roster",
                ));
            }
            original_leaves = Some(bank_leaves);
        }
        let device = self.session.borrow(py).device_ordinal;
        let mut restored_rosters = TensorHandoff(Vec::with_capacity(deliveries.len()));
        for (delivery, leaves) in deliveries.iter().zip(&leaves) {
            let delivery = delivery.borrow(py);
            let mut slots = Vec::with_capacity(leaves.len());
            for leaf in leaves {
                let gradient = &snapshots
                    .get(&(leaf.producer.as_ptr() as usize))
                    .expect("captured original physical leaf")
                    .1;
                let restored = if gradient.bind(py).is_none() {
                    None
                } else {
                    let port = ports.iter().find(|port| port.layout.role == SemanticStateRole::Gradients as u64
                        && port.producer.as_ptr() == gradient.as_ptr())
                        .ok_or_else(|| invalid("restored SOURCE gradient is not its original authenticated witness producer"))?;
                    Some(port.export(py, device, delivery.future_stream, check)?)
                };
                slots.push((
                    leaf.export(py, device, delivery.future_stream, check)?,
                    restored,
                ));
            }
            restored_rosters.0.push(slots);
        }
        check()?;
        let source = self.session.borrow(py);
        let source_owner = source.owner()?;
        let parent = self.parent.borrow(py);
        let lease = parent.lease()?;
        let invocation = member.invocation.borrow(py);
        for (delivery, slots) in deliveries.iter().zip(restored_rosters.into_native()) {
            let delivery = delivery.borrow(py);
            let original = delivery
                .future_import
                .as_ref()
                .ok_or_else(|| invalid("future delivery lost its original prepared import"))?;
            if !std::ptr::eq(self, &*original.borrow(py)) {
                return Err(invalid(
                    "future delivery completion changed its original import owner",
                ));
            }
            let (target, step, bank) = delivery
                .future_target
                .as_ref()
                .ok_or_else(|| invalid("future delivery lost its original target bank"))?;
            let target = target.borrow(py);
            target.require_creator()?;
            target
                .owner()?
                .complete_imported_edit_gradient_delivery(
                    step,
                    *bank,
                    &source_owner,
                    &lease,
                    &self.material,
                    invocation.rng,
                    self.ordinal as u64,
                    &self.replay_row,
                    &delivery._binding,
                    slots,
                )
                .map_err(xlog_err)?;
        }
        drop(invocation);
        drop(lease);
        drop(parent);
        drop(source_owner);
        drop(source);
        // Authenticate the same original edges before changing either bank's
        // saved restoration intent. No slot, destination or hook is replaced.
        for delivery in &deliveries {
            let delivery = delivery.borrow(py);
            let entries = delivery
                .deferred_hooks
                .lock()
                .map_err(|_| invalid("gradient-delivery hook roster is poisoned"))?
                .as_ref()
                .ok_or_else(|| invalid("future Edit hooks entered before SOURCE completion"))?
                .iter()
                .map(|entry| (entry.node.clone_ref(py), entry.effective.clone_ref(py)))
                .collect::<Vec<_>>();
            for (node, effective) in entries {
                check()?;
                if !node.bind(py).getattr("variable")?.is(effective.bind(py))
                    || !snapshots.contains_key(&(effective.as_ptr() as usize))
                {
                    return Err(invalid(
                        "SOURCE restoration changed its original AccumulateGrad leaf",
                    ));
                }
                check()?;
            }
        }
        // Revalidate identity without replacing the once-captured snapshot.
        // A metadata callback cannot turn a later arbitrary slot mutation into
        // the authentic numerical restore transition.
        let mut checked = BTreeSet::new();
        for leaves in &leaves {
            for leaf in leaves {
                let key = leaf.producer.as_ptr() as usize;
                if checked.insert(key) {
                    check()?;
                    let current = leaf.producer.bind(py).getattr("grad")?;
                    check()?;
                    if !current.is(snapshots
                        .get(&key)
                        .expect("captured original physical leaf")
                        .1
                        .bind(py))
                    {
                        return Err(invalid(
                            "SOURCE gradient slot changed after its original restored snapshot",
                        ));
                    }
                }
            }
        }
        for delivery in &deliveries {
            let delivery = delivery.borrow(py);
            let mut hooks = delivery
                .deferred_hooks
                .lock()
                .map_err(|_| invalid("gradient-delivery hook roster is poisoned"))?;
            for entry in hooks
                .as_mut()
                .ok_or_else(|| invalid("future Edit hooks lost their original deferred roster"))?
            {
                entry.original_grad = snapshots
                    .get(&(entry.effective.as_ptr() as usize))
                    .expect("authenticated original AccumulateGrad leaf")
                    .1
                    .clone_ref(py);
            }
        }
        Ok(())
    }

    pub(super) fn retire_source_witness(&self) {
        let witness = self
            .source_witness
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .take();
        drop(witness);
    }

    /// Called only after original Runtime cleanup and native reader releases.
    /// Executed imports additionally require final use, full comparison and
    /// TaskUse completion; unexecuted cancellation grants no import rights.
    pub(super) fn retire_source_port_custody(
        &self,
        py: Python<'_>,
        deliveries: &[Option<Py<PySemanticGradientDelivery>>; 2],
    ) {
        let ports = self
            .source_ports
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .take();
        let witness = self
            .source_witness
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .take();
        let mut leaves = Vec::new();
        for delivery in deliveries.iter().flatten() {
            let delivery = delivery.borrow(py);
            if delivery
                .future_import
                .as_ref()
                .is_some_and(|original| std::ptr::eq(self, &*original.borrow(py)))
            {
                let roster = std::mem::take(
                    &mut *delivery
                        .future_leaves
                        .lock()
                        .unwrap_or_else(std::sync::PoisonError::into_inner),
                );
                leaves.push(roster);
            }
        }
        // All retained owners leave on the creator thread without a Session,
        // group-use, final-use or roster mutex held. Cached results remain.
        drop(leaves);
        drop(ports);
        drop(witness);
    }

    fn require_metadata(&self, py: Python<'_>) -> PyResult<()> {
        let session = self.session.borrow(py);
        session.require_creator()?;
        let stage = *self
            .stage
            .lock()
            .map_err(|_| invalid("prepared import stage is poisoned"))?;
        if !matches!(
            stage,
            ImportPreparationStage::MetadataEntering | ImportPreparationStage::Prepared
        ) {
            return Err(invalid(
                "SOURCE preparation metadata belongs to the original unexecuted import",
            ));
        }
        let owner = session.owner()?;
        self.task_use.borrow(py).require_original_identity(&owner)?;
        owner
            .published_identity(&*self.parent.borrow(py).lease()?)
            .map_err(xlog_err)?;
        Ok(())
    }
}

#[pymethods]
impl PySemanticPreparedTaskImport {
    #[getter]
    fn task_use(&self, py: Python<'_>) -> PyResult<Py<PySemanticTransitionTaskUse>> {
        self.session.borrow(py).require_creator()?;
        Ok(self.task_use.clone_ref(py))
    }

    #[getter]
    fn parent(&self, py: Python<'_>) -> PyResult<Py<PySemanticPublishedParent>> {
        self.session.borrow(py).require_creator()?;
        Ok(self.parent.clone_ref(py))
    }

    /// Acknowledge the original reader-only cancellation performed by native retirement.
    fn finish_cancelled_private_segment(&self, py: Python<'_>) -> PyResult<()> {
        self.require_cancelled_ack(py)
    }

    #[pyo3(signature = (update_step, bank, *, consumer_stream, edit_only=false))]
    fn group_update_vjp(
        &self,
        py: Python<'_>,
        update_step: Py<PySemanticPreparedStep>,
        bank: &Bound<'_, PyAny>,
        consumer_stream: &Bound<'_, PyAny>,
        edit_only: bool,
    ) -> PyResult<Py<PyTuple>> {
        self.require_completed()?;
        let member = self
            .member
            .lock()
            .map_err(|_| invalid("prepared import member is poisoned"))?
            .as_ref()
            .map(|member| member.clone_ref(py))
            .ok_or_else(|| invalid("prepared import lost its genuine completed member"))?;
        let result =
            member
                .borrow(py)
                .group_update_vjp(py, update_step, bank, consumer_stream, edit_only);
        result
    }

    fn source_model_contract_metadata(&self, py: Python<'_>) -> PyResult<Py<PyTuple>> {
        self.require_metadata(py)?;
        let (bytes, layout) = self
            .material
            .source_model_contract_metadata()
            .map_err(xlog_err)?;
        Ok((
            PyBytes::new(py, bytes),
            (
                layout.schema_begin,
                layout.schema_bytes,
                layout.schema_digest_offset,
                layout.generation_offset,
                layout.numerical_digest_offset,
                layout.identity_offset,
            ),
        )
            .into_pyobject(py)?
            .unbind())
    }

    fn source_model_memory_geometry(&self, py: Python<'_>) -> PyResult<Py<PyTuple>> {
        self.require_metadata(py)?;
        let (allocations, storages, views, layouts) = self.material.source_model_memory_geometry();
        let layouts = layouts
            .into_iter()
            .map(|layout| ((layout.role, layout.index), layout))
            .collect::<BTreeMap<_, _>>();
        let storages = PyTuple::new(
            py,
            storages
                .into_iter()
                .map(|row| (row.allocation, row.byte_offset, row.span_bytes)),
        )?;
        let views = views
            .into_iter()
            .map(|view| -> PyResult<_> {
                let layout = layouts.get(&(view.role, view.index)).ok_or_else(|| {
                    invalid("SOURCE model metadata lost its original view layout")
                })?;
                Ok((
                    view.role,
                    view.index,
                    view.storage,
                    view.byte_offset,
                    (
                        layout.element_bytes,
                        layout.scalar_type,
                        layout.rank,
                        layout.logical_axis,
                        PyTuple::new(py, layout.dimensions)?,
                        PyTuple::new(py, layout.strides_bytes)?,
                    ),
                )
                    .into_pyobject(py)?
                    .into_any())
            })
            .collect::<PyResult<Vec<_>>>()?;
        let views = PyTuple::new(py, views)?;
        Ok((PyTuple::new(py, allocations)?, storages, views)
            .into_pyobject(py)?
            .unbind())
    }
}

#[pymethods]
impl PySemanticTransitionController {
    fn execute_task_import(
        &self,
        py: Python<'_>,
        prepared_import: Py<PySemanticPreparedTaskImport>,
    ) -> PyResult<Py<PyAny>> {
        let prepared = prepared_import.borrow(py);
        prepared.require_controller(py, self)?;
        if let Some(result) = prepared
            .completed
            .lock()
            .map_err(|_| invalid("prepared import result is poisoned"))?
            .as_ref()
        {
            return Ok(result.clone_ref(py));
        }
        prepared.require_original(py, self)?;
        let session = self.session.borrow(py);
        if session
            .prepared_task_import
            .lock()
            .map_err(|_| invalid("prepared import custody is poisoned"))?
            .as_ref()
            .is_none_or(|original| original.as_ptr() != prepared_import.as_ptr())
        {
            return Err(invalid(
                "task import execution cannot substitute its retained preparation",
            ));
        }
        let child = session.private_replay_custody()?;
        if let Some(child) = &child {
            child.begin_original_import_execution(py)?;
        }
        {
            let mut stage = prepared
                .stage
                .lock()
                .map_err(|_| invalid("prepared import stage is poisoned"))?;
            if *stage != ImportPreparationStage::Prepared {
                return Err(invalid(
                    "unknown import preparation or execution cannot repeat its original callbacks",
                ));
            }
            *stage = ImportPreparationStage::ExecutionEntering;
        }
        let issued = prepared.task_use.borrow(py);
        let mut import = ColdImportGuard {
            session: &session,
            issuance: issued.issuance.clone(),
            original_decisions: Some(Arc::clone(&prepared.original_decisions)),
            committed: false,
            deferred: true,
            retained_final_use: false,
            retained_preparation: true,
            native_pending: std::cell::Cell::new(false),
        };
        self.check_import_callback(
            py,
            &issued,
            &prepared.parent.borrow(py),
            &import,
            &prepared.expected,
        )?;
        let member = self
            .replay_import(
                py,
                &prepared.task_use,
                prepared.ordinal,
                Arc::clone(&prepared.material),
                Arc::clone(&prepared.original_decisions),
                SemanticTransitionKind::Proposal,
                prepared.restore_invocation.bind(py),
                prepared.pack_policy.bind(py),
                prepared.finish_invocation.bind(py),
                prepared.refresh_snapshot.bind(py),
                &import,
                true,
                Some(&prepared),
            )?
            .ok_or_else(|| invalid("prepared retained import lost its genuine completed member"))?;
        *prepared
            .member
            .lock()
            .map_err(|_| invalid("prepared import member is poisoned"))? =
            Some(member.clone_ref(py));
        if let Some(child) = &child {
            child.retain_member(py, &member)?;
        }
        prepared.complete_future_deliveries(py, &member.borrow(py), &|| {
            self.check_import_callback(
                py,
                &issued,
                &prepared.parent.borrow(py),
                &import,
                &prepared.expected,
            )
        })?;
        let result = (prepared.task_use.clone_ref(py), member)
            .into_pyobject(py)?
            .unbind()
            .into_any();
        *prepared
            .stage
            .lock()
            .map_err(|_| invalid("prepared import stage is poisoned"))? =
            ImportPreparationStage::Complete;
        if let Some(child) = &child {
            child.finish_original_import_execution(py)?;
        }
        *prepared
            .completed
            .lock()
            .map_err(|_| invalid("prepared import result is poisoned"))? =
            Some(result.clone_ref(py));
        import.committed = true;
        Ok(result)
    }
}

#[pymethods]
impl PySemanticTransitionSession {
    #[getter]
    fn prepared_task_import(
        &self,
        py: Python<'_>,
    ) -> PyResult<Option<Py<PySemanticPreparedTaskImport>>> {
        self.require_creator()?;
        Ok(self
            .prepared_task_import
            .lock()
            .map_err(|_| invalid("prepared import custody is poisoned"))?
            .as_ref()
            .map(|owner| owner.clone_ref(py)))
    }
}

#[pymethods]
impl PySemanticPublishedParent {
    fn source_model_contract_metadata(&self, py: Python<'_>) -> PyResult<Py<PyTuple>> {
        let prepared = self.original_prepared_import(py)?;
        let result = prepared.borrow(py).source_model_contract_metadata(py);
        result
    }

    fn source_model_memory_geometry(&self, py: Python<'_>) -> PyResult<Py<PyTuple>> {
        let prepared = self.original_prepared_import(py)?;
        let result = prepared.borrow(py).source_model_memory_geometry(py);
        result
    }
}

impl PySemanticTransitionController {
    pub(super) fn original_model_import_owner(
        &self,
        py: Python<'_>,
        task_use: &PySemanticTransitionTaskUse,
        parent: &PySemanticPublishedParent,
    ) -> PyResult<Option<Py<PySemanticPreparedTaskImport>>> {
        let session = self.session.borrow(py);
        session.require_creator()?;
        let prepared = session
            .prepared_task_import
            .lock()
            .map_err(|_| invalid("prepared import custody is poisoned"))?
            .as_ref()
            .map(|owner| owner.clone_ref(py));
        if let Some(prepared) = &prepared {
            let original = prepared.borrow(py);
            original.require_controller(py, self)?;
            original.require_source_port_owner(py, task_use, parent)?;
        }
        Ok(prepared)
    }
}

#[pymethods]
impl PySemanticPublishedParent {
    pub(super) fn original_prepared_import(
        &self,
        py: Python<'_>,
    ) -> PyResult<Py<PySemanticPreparedTaskImport>> {
        let session = self.session.borrow(py);
        session.require_creator()?;
        let prepared = session
            .prepared_task_import
            .lock()
            .map_err(|_| invalid("prepared import custody is poisoned"))?
            .as_ref()
            .map(|owner| owner.clone_ref(py))
            .ok_or_else(|| invalid("SOURCE metadata requires its original prepared import"))?;
        let original = prepared.borrow(py);
        if !std::ptr::eq(self, &*original.parent.borrow(py))
            || self.task_use.as_ptr() != original.task_use.as_ptr()
        {
            return Err(invalid(
                "SOURCE metadata cannot substitute another import reader",
            ));
        }
        drop(original);
        Ok(prepared)
    }
}
