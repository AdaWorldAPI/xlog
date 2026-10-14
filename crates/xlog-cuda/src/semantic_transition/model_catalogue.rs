use super::*;

// The first 25 words still identify the context's actual semantic bytes. The
// remaining header binds model rows, owned category ranges and the cold frame.
pub(super) const HEADER_WORDS: usize = 42;
const FRAME_MAGIC: u64 = 0x584c4f474d4f444c;
const ROW_WIDTHS: [usize; 4] = [10, 6, 3, 18];

/// Process-local authority derived only from a completed native cold domain.
/// Fresh restoration reissues it from both real cold recipes, not saved handles.
#[derive(Clone)]
pub struct SemanticFrozenModelContext {
    pub(super) catalogue: Arc<FrozenModelCatalogue>,
    pub(super) member: usize,
}

impl SemanticFrozenModelContext {
    pub fn require_budget(
        &self,
        budget: &SemanticReplayMaterialBudget,
    ) -> Result<(), SemanticTransitionError> {
        if !self.catalogue._reservation.budget.same_pool(budget) {
            return Err(publication_input_error(
                "frozen model context requires its original host capacity pool",
            ));
        }
        Ok(())
    }
}

impl SemanticReplayStateMaterialProof {
    pub fn require_frozen_model_context(
        &self,
        context: Option<&SemanticFrozenModelContext>,
    ) -> Result<(), SemanticTransitionError> {
        let header = &self.owner.codebook_header;
        match (header[40] != 0, context) {
            (false, None) => Ok(()),
            (true, Some(context))
                if words_identity(header, 25) == *context.catalogue.binding.digest.as_bytes() =>
            {
                Ok(())
            }
            _ => Err(publication_input_error(
                "checkpoint model rows require the same freshly frozen original cold domain",
            )),
        }
    }
}

struct CatalogueMember {
    content: SemanticTaskContentIdentity,
    words: Vec<u64>,
    program: Option<Identity256>,
}

/// Both original cold recipes, in their immutable model-row order. It grants
/// no task authority: each projection retains only its own descriptor meanings.
pub(super) struct FrozenModelCatalogue {
    members: [CatalogueMember; 2],
    frame: Vec<u64>,
    binding: SemanticCatalogueBinding,
    statement_bytes: usize,
    _reservation: SemanticReplayMaterialReservation,
}

fn normalized(words: &[u64]) -> impl Iterator<Item = u64> + '_ {
    words.iter().enumerate().map(|(index, &word)| {
        if matches!(index, 14..=16 | 21..=24) {
            0
        } else {
            word
        }
    })
}

impl FrozenModelCatalogue {
    #[cfg(feature = "semantic-policy")]
    pub(super) fn context_for_header(
        self: &Arc<Self>,
        header: &[u64; HEADER_WORDS],
    ) -> Result<SemanticFrozenModelContext, SemanticTransitionError> {
        if header[40] == 0
            || words_identity(header, 25) != *self.binding.digest.as_bytes()
            || header[37] != self.statement_bytes as u64
        {
            return Err(publication_input_error(
                "original source geometry belongs to another frozen model domain",
            ));
        }
        for member in 0..2 {
            let mut matches = header[39] == u64::from(self.members[member].program.is_some());
            for (table, count_word) in [1, 3, 5, 7].into_iter().enumerate() {
                let first = self.members[0].words[count_word];
                let count = first + self.members[1].words[count_word] - 1;
                matches &= header[count_word] == count
                    && header[29 + 2 * table] == if member == 0 { 1 } else { first }
                    && header[30 + 2 * table] == if member == 0 { first } else { count };
            }
            if matches {
                return Ok(SemanticFrozenModelContext {
                    catalogue: Arc::clone(self),
                    member,
                });
            }
        }
        Err(publication_input_error(
            "original source geometry changed its context-owned model row resolution",
        ))
    }

    fn new(
        sessions: [&SemanticTransitionSession; 2],
        statement_bytes: usize,
        reservation: SemanticReplayMaterialReservation,
    ) -> Result<Arc<Self>, SemanticTransitionError> {
        let members = sessions.map(|session| CatalogueMember {
            content: session
                .cold_content
                .as_ref()
                .expect("checked cold content")
                .0,
            words: normalized(&session.codebooks.words).collect(),
            program: session.codebooks.program_binding,
        });
        let mut frame = Vec::with_capacity(
            39 + members
                .iter()
                .map(|member| member.words.len())
                .sum::<usize>(),
        );
        frame.extend([FRAME_MAGIC, 1, statement_bytes as u64]);
        for member in &members {
            for identity in [
                member.content.query,
                member.content.theory_program,
                member.content.result,
            ] {
                frame.extend(identity_words(*identity.as_bytes()));
            }
            frame.push(u64::from(member.program.is_some()));
            frame.extend(identity_words(
                member
                    .program
                    .map_or([0; 32], |identity| *identity.as_bytes()),
            ));
            frame.push(member.words.len() as u64);
            frame.extend_from_slice(&member.words);
        }
        let mut hash = Sha256::new();
        hash.update(b"xlog.semantic.immutable-model-catalogue.v1\0");
        hash.update(CATALOGUE_DIGEST);
        for &word in &frame {
            hash.update(word.to_le_bytes());
        }
        let binding = SemanticCatalogueBinding {
            generation: CATALOGUE_GENERATION,
            digest: Identity256::from_bytes(hash.finalize().into()),
        };
        Ok(Arc::new(Self {
            members,
            frame,
            binding,
            statement_bytes,
            _reservation: reservation,
        }))
    }

    fn project(
        &self,
        original: &ActionCodebooks,
        member: usize,
    ) -> Result<Vec<u64>, SemanticTransitionError> {
        let selected = self
            .members
            .get(member)
            .ok_or(SemanticTransitionError::ObservationMismatch)?;
        if selected.program != original.program_binding
            || !normalized(&original.words).eq(selected.words.iter().copied())
        {
            return Err(publication_input_error(
                "model catalogue requires its original cold context recipe",
            ));
        }
        let mut words = Vec::with_capacity(
            self.frame.len()
                + self
                    .members
                    .iter()
                    .map(|member| member.words.len())
                    .sum::<usize>(),
        );
        words.resize(HEADER_WORDS, 0);
        words[..25].copy_from_slice(&original.words[..25]);
        words[21..25].fill(0);
        words[25..29].copy_from_slice(&identity_words(*self.binding.digest.as_bytes()));
        words[37] = self.statement_bytes as u64;
        words[38] = u64::from(self.members.iter().any(|member| member.program.is_some()));
        words[39] = u64::from(selected.program.is_some());
        let payload_lengths = self
            .members
            .each_ref()
            .map(|member| member.words.len() * 8 - member.words[8] as usize);
        for (table, width) in ROW_WIDTHS.into_iter().enumerate() {
            let count_word = 2 * table + 1;
            let first = self.members[0].words[count_word];
            let count = first
                .checked_add(self.members[1].words[count_word])
                .and_then(|value| value.checked_sub(1))
                .filter(|&value| value <= 65536)
                .ok_or_else(|| {
                    publication_input_error("immutable model field exceeds 65536 categories")
                })?;
            words[2 * table] = words.len() as u64;
            words[count_word] = count;
            words[29 + 2 * table] = if member == 0 { 1 } else { first };
            words[30 + 2 * table] = if member == 0 { first } else { count };
            // One intrinsic NULL, followed by both actual occurrence rosters.
            words.extend(std::iter::repeat_n(0, width));
            for (ordinal, context) in self.members.iter().enumerate() {
                let start = context.words[2 * table] as usize;
                for category in 1..context.words[count_word] as usize {
                    let row = words.len();
                    words.extend_from_slice(
                        &context.words[start + width * category..start + width * (category + 1)],
                    );
                    if table == 2 || table == 1 && words[row + 2] != 0 {
                        words[row] = words[row]
                            .checked_add(if ordinal == 0 {
                                0
                            } else {
                                payload_lengths[0] as u64
                            })
                            .ok_or(SemanticTransitionError::GenerationExhausted)?;
                    }
                }
            }
        }
        words[8] = (words.len() * 8) as u64;
        for context in &self.members {
            words.extend_from_slice(&context.words[context.words[8] as usize / 8..]);
        }
        words[40] = (words.len() * 8) as u64;
        words[41] = (self.frame.len() * 8) as u64;
        words.extend_from_slice(&self.frame);
        words[9] = (words.len() * 8) as u64 - words[8];
        let mut hash = Sha256::new();
        hash.update(b"xlog.semantic.model-context-resolution.v1\0");
        hash.update(original.binding.digest.as_bytes());
        for &word in &words {
            hash.update(word.to_le_bytes());
        }
        words[21..25].copy_from_slice(&identity_words(hash.finalize().into()));
        Ok(words)
    }
}

impl ActionCodebooks {
    pub(super) fn project_model_catalogue(
        &mut self,
        catalogue: Arc<FrozenModelCatalogue>,
        member: usize,
    ) -> Result<(), SemanticTransitionError> {
        let word_bound = catalogue
            .members
            .iter()
            .try_fold(catalogue.frame.len(), |bound, member| {
                bound.checked_add(member.words.len())
            })
            .ok_or(SemanticTransitionError::GenerationExhausted)?;
        let bytes = word_bound
            .checked_mul(3 * size_of::<u64>())
            .and_then(|bytes| {
                bytes.checked_add(2 * COMPONENT_COUNT * size_of::<SemanticComponent>())
            })
            .ok_or(SemanticTransitionError::GenerationExhausted)?;
        let reservation = catalogue
            ._reservation
            .budget
            .reserve_auxiliary(bytes as u64)?;
        let words = catalogue.project(self, member)?;
        let (components, cells) = Self::component_geometry(&words, self.program_binding.is_some())?;
        self.binding.digest = Identity256::from_bytes(words_identity(&words, 21));
        self.words = words;
        self.components = components.into_iter().collect();
        self.input_cells = cells;
        self.model_catalogue = Some((catalogue, member));
        self.model_projection_reservation = Some(Arc::new(reservation));
        Ok(())
    }
}

fn words_identity(words: &[u64], begin: usize) -> [u8; 32] {
    let mut identity = [0; 32];
    for (bytes, word) in identity.chunks_exact_mut(8).zip(&words[begin..begin + 4]) {
        bytes.copy_from_slice(&word.to_le_bytes());
    }
    identity
}

struct ContextBuffers {
    words: Vec<u64>,
    components: Vec<SemanticComponent>,
    cells: usize,
    logits: TrackedCudaSlice<f32>,
    support: TrackedCudaSlice<u8>,
    codebooks: TrackedCudaSlice<u64>,
    device_components: TrackedCudaSlice<SemanticComponent>,
}

impl ContextBuffers {
    fn allocate(
        session: &SemanticTransitionSession,
        catalogue: &FrozenModelCatalogue,
        member: usize,
    ) -> Result<Self, SemanticTransitionError> {
        let words = catalogue.project(&session.codebooks, member)?;
        let (components, cells) = ActionCodebooks::component_geometry(
            &words,
            session.codebooks.program_binding.is_some(),
        )?;
        let components: Vec<_> = components.into_iter().collect();
        let bytes = cells
            .checked_mul(size_of::<f32>() + size_of::<u8>())
            .and_then(|bytes| {
                words
                    .len()
                    .checked_mul(8)
                    .and_then(|words| bytes.checked_add(words))
            })
            .and_then(|bytes| bytes.checked_add(COMPONENT_COUNT * size_of::<SemanticComponent>()))
            .ok_or(SemanticTransitionError::GenerationExhausted)?;
        let mut reservation = session
            .provider
            .memory()
            .reserve_bytes(bytes as u64)
            .map_err(|error| runtime_error("model context allocation", error))?;
        Ok(Self {
            logits: reservation
                .alloc::<f32>(cells)
                .map_err(|error| runtime_error("model context logits", error))?,
            support: reservation
                .alloc::<u8>(cells)
                .map_err(|error| runtime_error("model context support", error))?,
            codebooks: reservation
                .alloc::<u64>(words.len())
                .map_err(|error| runtime_error("model context codebooks", error))?,
            device_components: reservation
                .alloc::<SemanticComponent>(COMPONENT_COUNT)
                .map_err(|error| runtime_error("model context components", error))?,
            words,
            components,
            cells,
        })
    }

    fn install(
        self,
        session: &mut SemanticTransitionSession,
        catalogue: Arc<FrozenModelCatalogue>,
        member: usize,
    ) {
        session.codebooks.binding.digest = Identity256::from_bytes(words_identity(&self.words, 21));
        session.codebooks.words = self.words;
        session.codebooks.components = self.components;
        session.codebooks.input_cells = self.cells;
        session.codebooks.model_catalogue = Some((catalogue, member));
        session.logits = self.logits;
        session.support = self.support;
        session.device_codebooks = self.codebooks;
        session.device_components = self.device_components;
    }
}

impl SemanticTransitionSession {
    pub fn frozen_model_context(
        &self,
    ) -> Result<Option<SemanticFrozenModelContext>, SemanticTransitionError> {
        self.ensure_quiescent()?;
        Ok(self
            .codebooks
            .model_catalogue
            .as_ref()
            .map(|(catalogue, member)| SemanticFrozenModelContext {
                catalogue: Arc::clone(catalogue),
                member: *member,
            }))
    }

    /// Freeze both original row rosters before task import, model allocation,
    /// capture or drawing. Unknown retains one shared original DMA operation.
    pub fn freeze_model_contexts(
        &mut self,
        next: &mut Self,
    ) -> Result<SemanticCatalogueBinding, SemanticTransitionError> {
        if let (Some((left, 0)), Some((right, 1))) = (
            &self.codebooks.model_catalogue,
            &next.codebooks.model_catalogue,
        ) {
            if !Arc::ptr_eq(left, right) {
                return Err(publication_input_error(
                    "model contexts belong to different original catalogues",
                ));
            }
            let binding = left.binding;
            self.resolve_constructor_codebooks(true)?;
            return Ok(binding);
        }
        for context in [&*self, &*next] {
            context.ensure_rebindable()?;
            #[cfg(feature = "semantic-policy")]
            if context.policy.is_some()
                || context.pending_policy.is_some()
                || context.prepared_policy.is_some()
            {
                return Err(publication_input_error(
                    "model context domain cannot change an allocated policy owner",
                ));
            }
            if context.cold_content.is_none()
                || context.cold_statement_records.is_empty()
                || context.task.is_some()
                || context.publication.is_some()
                || context.captured.is_some()
                || context.codebooks.model_catalogue.is_some()
                || !context.model_owners.is_empty()
                || context.text_binding.is_some()
                || context.next_proposal != 0
            {
                return Err(publication_input_error("model context domain must freeze once from both original cold tasks before model use"));
            }
        }
        if !Arc::ptr_eq(self.provider.memory(), next.provider.memory())
            || self.domain.stream_id() != next.domain.stream_id()
        {
            return Err(publication_input_error(
                "model contexts require the original allocator and serial stream",
            ));
        }
        let budget = self
            .replay_material_budget
            .as_ref()
            .ok_or(SemanticTransitionError::NotBound)?;
        if next
            .replay_material_budget
            .as_ref()
            .is_none_or(|other| !budget.same_pool(other))
        {
            return Err(publication_input_error(
                "model contexts require one original host material capacity pool",
            ));
        }
        let word_count = self
            .codebooks
            .words
            .len()
            .checked_add(next.codebooks.words.len())
            .ok_or(SemanticTransitionError::GenerationExhausted)?;
        // Includes both immutable recipes/frame, two projections and overlapping
        // original upload staging. No numerical bank is copied by this operation.
        let bytes = word_count
            .checked_mul(12 * size_of::<u64>())
            .and_then(|bytes| {
                bytes.checked_add(
                    8 * HEADER_WORDS * 8 + 4 * COMPONENT_COUNT * size_of::<SemanticComponent>(),
                )
            })
            .ok_or(SemanticTransitionError::GenerationExhausted)?;
        let reservation = budget.reserve_auxiliary(bytes as u64)?;
        // Reserve the canonical encoder's largest temporary before copying any
        // learned payload. Repeated qualifier occurrences remain real bytes.
        let mut payload_bound = 0usize;
        for context in [&*self, &*next] {
            let admission = context
                .graph
                .admission()
                .ok_or(SemanticTransitionError::NotBound)?;
            for &record in &context.cold_statement_records {
                let statement = admission
                    .records()
                    .records
                    .get(record as usize)
                    .ok_or(SemanticTransitionError::InvalidTaskBank)?;
                let mut bound = 4usize;
                for index in std::iter::once(record).chain(statement.qualifiers.iter().copied()) {
                    let atom = admission
                        .encoded_records
                        .get(index as usize)
                        .ok_or(SemanticTransitionError::InvalidTaskBank)?;
                    bound = bound
                        .checked_add(8)
                        .and_then(|bound| bound.checked_add(atom.bytes.len()))
                        .ok_or(SemanticTransitionError::GenerationExhausted)?;
                }
                payload_bound = payload_bound.max(bound);
            }
        }
        let payload_reservation = budget.reserve_auxiliary(
            payload_bound
                .checked_mul(2)
                .and_then(|bytes| bytes.checked_add(size_of::<Vec<u8>>()))
                .ok_or(SemanticTransitionError::GenerationExhausted)? as u64,
        )?;
        let mut statement_bytes = 0usize;
        for context in [&*self, &*next] {
            let admission = context
                .graph
                .admission()
                .ok_or(SemanticTransitionError::NotBound)?;
            for &record in &context.cold_statement_records {
                let payload = feedback_statement_payload(
                    admission.records(),
                    &admission.encoded_records,
                    record,
                )?;
                statement_bytes = statement_bytes.max(payload.len());
            }
        }
        SemanticFeedbackSchema::from_statement_lengths([statement_bytes])?;
        drop(payload_reservation);
        let catalogue = FrozenModelCatalogue::new([&*self, &*next], statement_bytes, reservation)?;
        let left = ContextBuffers::allocate(self, &catalogue, 0)?;
        let right = ContextBuffers::allocate(next, &catalogue, 1)?;
        let mut writes = Vec::with_capacity(4);
        for buffers in [&left, &right] {
            writes.push(state_restoration::OriginalDeviceWrite::new(
                &self.stream,
                &buffers.words,
                buffers.codebooks.view(),
                true,
            )?);
            writes.push(state_restoration::OriginalDeviceWrite::new(
                &self.stream,
                &buffers.components,
                buffers.device_components.view(),
                true,
            )?);
        }
        let original = Arc::new(Mutex::new(OriginalSessionCodebooks {
            writes,
            cursor: 0,
            poisoned: false,
            complete: false,
            retired: false,
            _model_catalogue: Some(Arc::clone(&catalogue)),
        }));
        let pending = Arc::new(AtomicBool::new(true));
        let binding = catalogue.binding;
        // Both actual allocation owners and shared command custody are installed
        // before the first effect; either Session stays unusable while pending.
        left.install(self, Arc::clone(&catalogue), 0);
        right.install(next, catalogue, 1);
        self.construction_codebooks = Some(Arc::clone(&original));
        next.construction_codebooks = Some(original);
        self.construction_pending = Arc::clone(&pending);
        next.construction_pending = pending;
        self.resolve_constructor_codebooks(true)?;
        Ok(binding)
    }

    /// Whether this exact ordered pair can resolve an entered original upload
    /// with unknown completion. Known refusal cannot create a retry loop merely
    /// because installation retained the original catalogue/allocations.
    pub fn model_contexts_pending(&self, next: &Self) -> bool {
        matches!(
            (&self.codebooks.model_catalogue, &next.codebooks.model_catalogue),
            (Some((left, 0)), Some((right, 1))) if Arc::ptr_eq(left, right)
        ) && Arc::ptr_eq(&self.construction_pending, &next.construction_pending)
            && self.construction_pending.load(Ordering::Acquire)
            && self.construction_codebooks.as_ref().is_some_and(|owner| {
                owner.lock().is_ok_and(|original| {
                    !original.complete
                        && !original.retired
                        && original
                            .writes
                            .get(original.cursor)
                            .is_some_and(|write| write.completion_pending())
                })
            })
    }

    /// Model row identity is distinct from this context's authenticated bytes.
    pub fn model_catalogue_binding(&self) -> SemanticCatalogueBinding {
        self.codebooks
            .model_catalogue
            .as_ref()
            .map_or(self.codebooks.binding, |(catalogue, _)| catalogue.binding)
    }

    pub fn feedback_schema(&self) -> Result<SemanticFeedbackSchema, SemanticTransitionError> {
        let mut schema = SemanticFeedbackSchema::new(self.feedback_statement_bytes()?)?;
        if let Some((catalogue, _)) = &self.codebooks.model_catalogue {
            if schema.statement_bytes > catalogue.statement_bytes {
                return Err(publication_input_error(
                    "task feedback exceeds its original immutable model domain",
                ));
            }
            schema = SemanticFeedbackSchema::from_statement_lengths([catalogue.statement_bytes])?;
        }
        Ok(schema)
    }
}
