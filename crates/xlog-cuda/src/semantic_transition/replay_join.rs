//! Original reader completion and publication origin joins for replay import.

use super::*;

/// Retains the original replay lineage until both actual reader rosters join.
pub struct SemanticReplayPublicationJoin {
    issuer: Arc<()>,
    owner: Arc<Mutex<OriginalReplayPublicationJoin>>,
}

struct PublicationJoinParent {
    token: u64,
    identity: SemanticPublishedIdentity,
    reader_pin: Option<Arc<()>>,
}

impl PublicationJoinParent {
    fn matches(&self, lease: &SemanticPublishedLease) -> bool {
        lease.active && self.token == lease.token && self.identity == lease.identity
    }
}

pub(super) struct RetainedMaterialRange {
    pub(super) range: PublicationRange,
    pub(super) capacity: usize,
    pub(super) read: Option<PublicationRead<u8>>,
}

impl RetainedMaterialRange {
    pub(super) fn resolve(
        &mut self,
        session: &mut SemanticTransitionSession,
        poisoned: &mut bool,
    ) -> Result<PublicationMaterialRange, SemanticTransitionError> {
        Ok(PublicationMaterialRange {
            range: self.range,
            capacity: self.capacity,
            bytes: if let Some(read) = &mut self.read {
                session.resolve_publication_read_with_poison(read, poisoned)?
            } else {
                Vec::new()
            },
        })
    }
}

pub(super) struct OriginalReplayPublicationJoin {
    predecessor: PublicationJoinParent,
    successor: PublicationJoinParent,
    material: SemanticReplayMaterial,
    consumers: Option<[OriginalStepConsumers; 2]>,
    cursor: usize,
    completed: bool,
    allocation_reservation: Option<SemanticReplayMaterialReservation>,
}

impl SemanticTransitionSession {
    pub fn begin_replay_publication_join(
        &mut self,
        material: &SemanticReplayMaterial,
        predecessor: &SemanticPublishedLease,
        successor: &SemanticPublishedLease,
    ) -> Result<SemanticReplayPublicationJoin, SemanticTransitionError> {
        self.ensure_quiescent()?;
        self.checked_reader(predecessor)?;
        self.checked_reader(successor)?;
        self.checked_original_consumer_step(predecessor)?;
        self.checked_original_consumer_step(successor)?;
        if predecessor.token == successor.token || self.replay_join.is_some() {
            return Err(publication_input_error(
                "replay origin join requires two original distinct held readers",
            ));
        }
        self.require_replayed_publication_origins(material, predecessor, successor)?;
        let budget = match (
            self.replay_material_budget.as_ref(),
            material.budget.as_ref(),
        ) {
            (Some(original), Some(material)) if original.same_pool(material) => Some(original),
            (None, None) => None,
            _ => {
                return Err(publication_input_error(
                    "replay origin join changes its original capacity pool",
                ))
            }
        };
        let allocation_reservation = budget
            .map(|budget| {
                let mut requested = size_of::<OriginalReplayPublicationJoin>() as u64;
                for lease in [predecessor, successor] {
                    let streams = self.readers[&lease.token]
                        .consumer_streams
                        .len()
                        .checked_add(self.steps[&lease.token].consumer_streams.len())
                        .ok_or(SemanticTransitionError::GenerationExhausted)?;
                    let width = 3 * OwnedPublicationMaterial::tree_node_bound(size_of::<u64>(), 0)?
                        + 2 * size_of::<OriginalConsumerEdge>();
                    requested = requested
                        .checked_add(
                            (streams as u64)
                                .checked_mul(width as u64)
                                .ok_or(SemanticTransitionError::GenerationExhausted)?,
                        )
                        .ok_or(SemanticTransitionError::GenerationExhausted)?;
                }
                budget.reserve_owned(requested, requested)
            })
            .transpose()?;
        let consumers = [
            self.stage_replay_consumers(predecessor)?,
            self.stage_replay_consumers(successor)?,
        ];
        let owner = Arc::new(Mutex::new(OriginalReplayPublicationJoin {
            predecessor: PublicationJoinParent {
                token: predecessor.token,
                identity: predecessor.identity,
                reader_pin: Some(Arc::clone(&self.readers[&predecessor.token].aliases)),
            },
            successor: PublicationJoinParent {
                token: successor.token,
                identity: successor.identity,
                reader_pin: Some(Arc::clone(&self.readers[&successor.token].aliases)),
            },
            material: material.clone(),
            consumers: Some(consumers),
            cursor: 0,
            completed: false,
            allocation_reservation,
        }));
        self.replay_join = Some(Arc::clone(&owner));
        self.replay_join_pending.store(true, Ordering::Release);
        Ok(SemanticReplayPublicationJoin {
            issuer: Arc::clone(&self.publication_issuer),
            owner,
        })
    }

    fn require_replayed_publication_origins(
        &self,
        material: &SemanticReplayMaterial,
        predecessor: &SemanticPublishedLease,
        successor: &SemanticPublishedLease,
    ) -> Result<(), SemanticTransitionError> {
        let expected = material.predecessor_identity();
        if predecessor.header.instance == expected.instance
            || predecessor.header.recovered_instance != expected.instance
            || predecessor.header.publication_word != 0
            || predecessor.identity.logical_digest != expected.logical_digest
            || predecessor.header.original_instance
                != material.predecessor.bank.header.original_instance
            || successor.identity.logical_digest != material.successor_identity().logical_digest
            || successor.header.original_instance
                != material.predecessor.bank.header.original_instance
        {
            return Err(publication_input_error(
                "replay import changed its original restored publication lineage",
            ));
        }
        Ok(())
    }

    fn stage_replay_consumers(
        &self,
        lease: &SemanticPublishedLease,
    ) -> Result<OriginalStepConsumers, SemanticTransitionError> {
        let streams = self.readers[&lease.token]
            .consumer_streams
            .iter()
            .chain(&self.steps[&lease.token].consumer_streams)
            .copied()
            .collect::<BTreeSet<_>>();
        OriginalStepConsumers::new(
            &self.stream,
            Some(lease.identity),
            streams.clone(),
            streams,
            OriginalConsumerPurpose::Observation,
        )
    }

    fn require_replay_join(
        &self,
        capability: &SemanticReplayPublicationJoin,
        original: &OriginalReplayPublicationJoin,
        predecessor: &SemanticPublishedLease,
        successor: &SemanticPublishedLease,
    ) -> Result<(), SemanticTransitionError> {
        self.checked_original_consumer_step(predecessor)?;
        self.checked_original_consumer_step(successor)?;
        if !Arc::ptr_eq(&capability.issuer, &self.publication_issuer)
            || !original.predecessor.matches(predecessor)
            || !original.successor.matches(successor)
            || (!original.completed
                && self
                    .replay_join
                    .as_ref()
                    .is_none_or(|owner| !Arc::ptr_eq(owner, &capability.owner)))
        {
            return Err(SemanticTransitionError::ObservationMismatch);
        }
        self.require_replayed_publication_origins(&original.material, predecessor, successor)?;
        if self.has_unresolved_session_native_effect_except_replay_join()
            || self.has_pending_state_restoration_except(None, None)
            || self.pending
            || self.pending_replay_delivery.is_some()
            || self.pending_training_materialization.is_some()
            || self
                .readers
                .values()
                .any(|reader| reader.retirement_pending || reader.acquisition_pending)
            || self
                .steps
                .values()
                .any(|step| step.consumer_completion.is_some())
        {
            return Err(SemanticTransitionError::Poisoned);
        }
        #[cfg(feature = "semantic-policy")]
        if self.has_pending_actor_refresh_context_except(None)
            || self.has_pending_cold_model_work_completion(None)
        {
            return Err(SemanticTransitionError::Poisoned);
        }
        self.graph
            .ensure_not_poisoned()
            .map_err(SemanticTransitionError::Semantic)
    }

    pub fn resolve_replay_publication_join(
        &mut self,
        capability: &mut SemanticReplayPublicationJoin,
        predecessor: &SemanticPublishedLease,
        successor: &SemanticPublishedLease,
    ) -> Result<(), SemanticTransitionError> {
        let owner = Arc::clone(&capability.owner);
        let mut original = owner
            .lock()
            .map_err(|_| publication_input_error("original replay join owner poisoned"))?;
        self.require_replay_join(capability, &original, predecessor, successor)?;
        if original.completed {
            return Ok(());
        }
        let _ordinary =
            crate::cuda_graph::reserve_uncaptured_stream(&self.stream).map_err(|error| {
                runtime_error("original replay reader join stream admission", error)
            })?;
        while original.cursor < 2 {
            let cursor = original.cursor;
            original
                .consumers
                .as_mut()
                .expect("original replay reader rosters")[cursor]
                .complete(&self.stream, &mut self.stream_waits, true)?;
            original.cursor += 1;
        }
        self.require_replayed_publication_origins(&original.material, predecessor, successor)?;
        original.consumers = None;
        original.predecessor.reader_pin = None;
        original.successor.reader_pin = None;
        original.allocation_reservation = None;
        original.completed = true;
        self.replay_join_pending.store(false, Ordering::Release);
        self.replay_join = None;
        Ok(())
    }

    pub fn replay_publication_join_pending(
        &self,
        capability: &SemanticReplayPublicationJoin,
        predecessor: &SemanticPublishedLease,
        successor: &SemanticPublishedLease,
    ) -> Result<bool, SemanticTransitionError> {
        let original = capability
            .owner
            .lock()
            .map_err(|_| publication_input_error("original replay join owner poisoned"))?;
        self.require_replay_join(capability, &original, predecessor, successor)?;
        Ok(!original.completed
            && original.consumers.as_ref().is_some_and(|consumers| {
                consumers.iter().any(|consumer| {
                    !consumer.completed
                        && consumer.poisoned
                        && (consumer.cursor != 0
                            || consumer.edges.iter().any(|edge| {
                                (edge.record_entered && !edge.recorded && !edge.source_completed)
                                    || (edge.wait_entered && !edge.waited)
                            }))
                })
            }))
    }
}
