use super::*;

#[derive(Clone, PartialEq, Eq)]
pub(super) struct PublicationReleaseTarget {
    token: u64,
    identity: SemanticPublishedIdentity,
    streams: Arc<[u64]>,
}

impl SemanticTransitionSession {
    pub fn join_aborted_publication_release(
        &mut self,
        releases: &mut [(&mut SemanticPublishedLease, &[u64])],
    ) -> Result<(), SemanticTransitionError> {
        if releases.is_empty()
            || releases
                .iter()
                .any(|(lease, _)| !Arc::ptr_eq(&lease.issuer, &self.publication_issuer))
        {
            return Err(publication_input_error(
                "terminal publication release changed its original Session",
            ));
        }
        let targets: Arc<[PublicationReleaseTarget]> = releases
            .iter()
            .map(|(lease, streams)| PublicationReleaseTarget {
                token: lease.token,
                identity: lease.identity,
                streams: streams
                    .iter()
                    .copied()
                    .collect::<BTreeSet<_>>()
                    .into_iter()
                    .collect::<Vec<_>>()
                    .into(),
            })
            .collect::<Vec<_>>()
            .into();
        let tokens = targets
            .iter()
            .map(|target| target.token)
            .collect::<BTreeSet<_>>();
        if tokens.len() != targets.len() {
            return Err(publication_input_error(
                "terminal publication release repeats a reader",
            ));
        }
        let kind = TaskReleaseKind::Publications(targets);
        match &self.task_release {
            TaskReleaseState::Joined(original) if original == &kind => return Ok(()),
            TaskReleaseState::Joining(original) if original == &kind => {}
            TaskReleaseState::Failed(original) if original == &kind => {
                return Err(SemanticTransitionError::Poisoned);
            }
            TaskReleaseState::Idle => {
                #[cfg(feature = "semantic-policy")]
                self.require_closed_evaluations()?;
                if self.publication.is_none()
                    || self.training_canary_source_borrowed()
                    || self.captured.is_some()
                    || self.prepared_segment.is_some()
                    || !self.prepared_resources.is_empty()
                    || self.retired_preparation.is_some()
                    || self.admitted_transition.is_some()
                    || self.readers.keys().any(|token| !tokens.contains(token))
                    || self.steps.keys().any(|token| !tokens.contains(token))
                    || !self.original_session_completion_is_uncontended()
                    || self.has_unresolved_native_effect_except_actor_cancellation_with_effect_exclusion(
                        #[cfg(feature = "semantic-policy")]
                        None,
                        OriginalSessionEffectExclusion {
                            allow_session_poison: true,
                            delivery_acknowledgement: None,
                        },
                    )
                {
                    return Err(publication_input_error(
                        "terminal publication release retains independent consumers or private effects",
                    ));
                }
                for (lease, consumer_streams) in releases.iter() {
                    if self.steps.contains_key(&lease.token) {
                        self.require_step_content_retirement(lease)?;
                        if let Some(original) = &self.steps[&lease.token].consumer_completion {
                            let original = original.lock().map_err(|_| {
                                publication_input_error("original consumer owner poisoned")
                            })?;
                            if original.purpose == OriginalConsumerPurpose::Cancellation
                                || original.requested_streams
                                    != consumer_streams.iter().copied().collect()
                            {
                                return Err(publication_input_error(
                                "terminal publication release changed its original consumer occasion",
                            ));
                            }
                        }
                        if let Some(original) = self
                            .readers
                            .get(&lease.token)
                            .and_then(|reader| reader.retirement_owner.as_ref())
                        {
                            let original = original.lock().map_err(|_| {
                                publication_input_error("original reader retirement owner poisoned")
                            })?;
                            let consumers = original.consumers.lock().map_err(|_| {
                                publication_input_error("original reader consumer owner poisoned")
                            })?;
                            if consumers.requested_streams
                                != consumer_streams.iter().copied().collect()
                            {
                                return Err(publication_input_error(
                                    "terminal publication release changed its closed stream roster",
                                ));
                            }
                        }
                    } else if lease.active || self.readers.contains_key(&lease.token) {
                        return Err(publication_input_error(
                            "terminal publication release lost its original consumer owner",
                        ));
                    }
                }
                self.poisoned = true;
                self.task_release = TaskReleaseState::Joining(kind.clone());
            }
            _ => {
                return Err(publication_input_error(
                    "terminal publication release changed its original operation",
                ));
            }
        }
        self.synchronize_task_release(&kind)?;
        self.require_completed_graph_retirements()?;
        for (lease, consumer_streams) in releases.iter_mut() {
            if let Some(original) = self
                .steps
                .get(&lease.token)
                .and_then(|step| step.consumer_completion.as_ref())
            {
                let purpose = original
                    .lock()
                    .map_err(|_| publication_input_error("original consumer owner poisoned"))?
                    .purpose;
                if purpose == OriginalConsumerPurpose::Observation {
                    self.complete_original_step_consumers(lease.token, consumer_streams, purpose)?;
                    self.steps
                        .get_mut(&lease.token)
                        .expect("completed original consumer step")
                        .consumer_completion = None;
                }
            }
            if self.steps.contains_key(&lease.token) {
                // release() resumes its retained command. A handed-off reader is
                // already inactive and only its remaining content is retired.
                self.release(lease, consumer_streams)?;
            }
        }
        for (lease, _) in releases.iter() {
            self.require_retired_publication(lease)?;
        }
        self.synchronize_task_release(&kind)?;
        self.task_release = TaskReleaseState::Joined(kind);
        Ok(())
    }

    pub(super) fn terminal_publication_retirement(&self, token: u64) -> bool {
        matches!(
            &self.task_release,
            TaskReleaseState::Joining(TaskReleaseKind::Publications(targets))
                if targets.iter().any(|target| target.token == token)
        )
    }

    pub(super) fn retirement_effect_exclusion(
        &self,
        token: u64,
    ) -> OriginalSessionEffectExclusion<'_> {
        OriginalSessionEffectExclusion {
            allow_session_poison: self.terminal_publication_retirement(token),
            delivery_acknowledgement: None,
        }
    }

    pub(super) fn ensure_retirement_quiescent(
        &self,
        token: u64,
    ) -> Result<(), SemanticTransitionError> {
        if self.terminal_publication_retirement(token) {
            return if self.original_completion_may_submit(token, None, None) {
                Ok(())
            } else {
                Err(SemanticTransitionError::Poisoned)
            };
        }
        self.ensure_quiescent_except_cold_report_with_effect_exclusion(
            None,
            self.retirement_effect_exclusion(token),
        )
    }

    pub(super) fn synchronize_task_release(
        &mut self,
        kind: &TaskReleaseKind,
    ) -> Result<(), SemanticTransitionError> {
        let result = self
            .stream
            .context()
            .bind_to_thread()
            .and_then(|_| self.stream.context().synchronize());
        if let Err(error) = result {
            self.poisoned = true;
            if crate::device::known_execution_failure(&error).is_some() {
                self.task_release = TaskReleaseState::Failed(kind.clone());
            }
            return Err(runtime_error("original task terminal join", error));
        }
        Ok(())
    }
}
