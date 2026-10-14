//! Native issuance and single-attempt custody of cold phase records.

use super::*;
use ed25519_dalek::{Signer, SigningKey};
use rand::rngs::OsRng;

const INPUT_DOMAIN: &[u8] = b"xlog.learning-phase.inputs.v2\0";
const RECORD_DOMAIN: &[u8] = b"xlog.learning-phase.record.v1\0";
const RECORD_OVERHEAD: usize = RECORD_DOMAIN.len() + 1 + 32 + 32 + 8 + 32 + 8 + 24 + 8 + 64;
const RECORD_PREFIX: usize = RECORD_OVERHEAD - 64;

#[derive(Clone, Copy, PartialEq, Eq)]
pub(super) enum RecordKind {
    Admission = 0,
    PreparationOutcome = 1,
    Delivery = 2,
    TerminalRefusal = 3,
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub(super) struct RecordLimits {
    pub(super) record_bytes: u64,
    pub(super) total_bytes: u64,
    pub(super) records: u64,
}

impl RecordLimits {
    pub(super) fn read(value: &Bound<'_, PyAny>) -> PyResult<Self> {
        let value = ColdValue::read(value, &mut 1024, 0)?;
        let fields = value.fields(3)?;
        let limits = Self {
            record_bytes: fields[0].unsigned()?,
            total_bytes: fields[1].unsigned()?,
            records: fields[2].unsigned()?,
        };
        if limits.record_bytes == 0
            || limits.record_bytes > limits.total_bytes
            || limits.total_bytes > isize::MAX as u64
            || limits.records == 0
            || limits.records > isize::MAX as u64
        {
            return Err(invalid(
                "phase record limits require explicit positive record/total/count bounds within the platform index range",
            ));
        }
        Ok(limits)
    }

    pub(super) fn encode(self) -> [u8; 24] {
        let mut bytes = [0; 24];
        for (target, value) in bytes.as_chunks_mut::<8>().0.iter_mut().zip([
            self.record_bytes,
            self.total_bytes,
            self.records,
        ]) {
            target.copy_from_slice(&value.to_le_bytes());
        }
        bytes
    }
}

/// A bounded slice of an original signed record or length-delimited payload.
/// Parsing never invokes a Python callback or allocates from a declared length.
pub(super) struct RecordReader<'a>(pub(super) &'a [u8]);

impl<'a> RecordReader<'a> {
    pub(super) fn take(&mut self, count: usize) -> PyResult<&'a [u8]> {
        if count > self.0.len() {
            return Err(invalid("signed phase material is truncated"));
        }
        let (value, remaining) = self.0.split_at(count);
        self.0 = remaining;
        Ok(value)
    }

    pub(super) fn word(&mut self) -> PyResult<u64> {
        Ok(u64::from_le_bytes(self.take(8)?.try_into().unwrap()))
    }

    pub(super) fn field(&mut self) -> PyResult<&'a [u8]> {
        let count = usize::try_from(self.word()?)
            .map_err(|_| invalid("signed phase field length exceeds this host"))?;
        self.take(count)
    }

    pub(super) fn finish(self) -> PyResult<()> {
        if self.0.is_empty() {
            Ok(())
        } else {
            Err(invalid(
                "signed phase material contains an unconsumed suffix",
            ))
        }
    }
}

/// Authenticate the complete original lifecycle before any replacement CUDA
/// owner exists. The issuer and limits come from the independent consumer pin,
/// never from an untrusted record header or a newly generated signing key.
pub(super) struct LifecyclePayloads<'a> {
    pub(super) admission: &'a [u8],
    pub(super) preparation: Option<&'a [u8]>,
    pub(super) completion: &'a [u8],
    pub(super) refused: bool,
}

pub(super) fn lifecycle_payloads<'a>(
    records: &[&'a [u8]],
    issuer: [u8; 32],
    phase_id: [u8; 32],
    limits: RecordLimits,
) -> PyResult<LifecyclePayloads<'a>> {
    if limits.records < 3 || !matches!(records.len(), 2 | 3) {
        return Err(invalid(
            "original phase bounds cannot contain the complete lifecycle",
        ));
    }
    if (records[0].len() as u64)
        .checked_add(limits.record_bytes)
        .and_then(|required| required.checked_add(limits.record_bytes))
        .is_none_or(|required| required > limits.total_bytes)
    {
        return Err(invalid(
            "original phase limits lost their reserved complete outcome custody",
        ));
    }
    let key = ed25519_dalek::VerifyingKey::from_bytes(&issuer)
        .map_err(|_| invalid("independent phase issuer has an invalid encoding"))?;
    let mut previous = [0u8; 32];
    let mut total = 0u64;
    let mut payloads = [&[][..]; 3];
    let mut kinds = [0u8; 3];
    for (ordinal, bytes) in records.iter().copied().enumerate() {
        total = total
            .checked_add(bytes.len() as u64)
            .ok_or_else(|| invalid("original phase record total overflowed"))?;
        if bytes.len() < RECORD_OVERHEAD
            || bytes.len() as u64 > limits.record_bytes
            || total > limits.total_bytes
        {
            return Err(invalid(
                "signed phase chain exceeds its original record bounds",
            ));
        }
        let (message, signature) = bytes.split_at(bytes.len() - 64);
        let signature = ed25519_dalek::Signature::from_slice(signature)
            .map_err(|_| invalid("signed phase record has an invalid signature encoding"))?;
        key.verify_strict(message, &signature)
            .map_err(|_| invalid("signed phase record differs from its independent issuer"))?;
        let mut reader = RecordReader(message);
        if reader.take(RECORD_DOMAIN.len())? != RECORD_DOMAIN {
            return Err(invalid("signed phase record has another original domain"));
        }
        kinds[ordinal] = reader.take(1)?[0];
        if reader.take(32)? != phase_id
            || reader.take(32)? != issuer
            || reader.word()? != ordinal as u64
            || reader.take(32)? != previous
            || reader.word()? != total
            || reader.take(24)? != limits.encode()
        {
            return Err(invalid(
                "signed phase record changed its original lifecycle or limits",
            ));
        }
        payloads[ordinal] = reader.field()?;
        reader.finish()?;
        previous = Sha256::digest(bytes).into();
    }
    let refused = match &kinds[..records.len()] {
        [0, 1, 2] => false,
        [0, 3] | [0, 1, 3] => true,
        _ => {
            return Err(invalid(
                "signed phase chain changed its original lifecycle order",
            ))
        }
    };
    let admission = payloads[0];
    let digest: [u8; 32] = Sha256::digest(admission).into();
    if !admission.starts_with(INPUT_DOMAIN) || digest != phase_id {
        return Err(invalid(
            "phase admission changed its original input identity",
        ));
    }
    let mut inputs = RecordReader(&admission[INPUT_DOMAIN.len()..]);
    for _ in 0..7 {
        inputs.field()?;
    }
    if inputs.field()? != limits.encode() {
        return Err(invalid(
            "phase admission changed its independent original bounds",
        ));
    }
    inputs.finish()?;
    Ok(LifecyclePayloads {
        admission: payloads[0],
        preparation: (records.len() == 3).then_some(payloads[1]),
        completion: payloads[records.len() - 1],
        refused,
    })
}

pub(super) struct RecordAttempt {
    pub(super) kind: RecordKind,
    pub(super) ordinal: u64,
    pub(super) bytes: xlog_cuda::SemanticReplayLoadedMaterial,
    pub(super) digest: [u8; 32],
    pub(super) readback_observed: bool,
}

pub(super) struct PreparedRecordOutput {
    pub(super) output: xlog_cuda::SemanticReplayMaterialOutput,
    header: [u8; RECORD_PREFIX],
    payload_extent: usize,
    signature_written: bool,
    record: Option<xlog_cuda::SemanticReplayLoadedMaterial>,
    _metadata: xlog_cuda::SemanticReplayMaterialReservation,
}

impl PreparedRecordOutput {
    fn finish(&mut self, key: &SigningKey) -> PyResult<xlog_cuda::SemanticReplayLoadedMaterial> {
        if let Some(record) = &self.record {
            return record
                .slice_payload(0..record.as_bytes().len())
                .map_err(xlog_err);
        }
        let body = self.output.seal_frame_body().map_err(xlog_err)?;
        if !self.signature_written {
            let signature = key.sign(body.as_bytes()).to_bytes();
            self.output
                .write_frame(
                    RECORD_PREFIX
                        .checked_add(self.payload_extent)
                        .ok_or_else(|| invalid("signed phase record size overflowed"))?,
                    &signature,
                )
                .map_err(xlog_err)?;
            self.signature_written = true;
        }
        let record = self.output.seal_frame().map_err(xlog_err)?;
        let retained = record
            .slice_payload(0..record.as_bytes().len())
            .map_err(xlog_err)?;
        self.record = Some(record);
        Ok(retained)
    }
}

/// The private key never enters Python or a checkpoint. Each write is retained
/// before calling the original store and may subsequently only be read back.
pub(super) struct PhaseRecords {
    pub(super) phase_id: [u8; 32],
    pub(super) limits: RecordLimits,
    key: SigningKey,
    pub(super) inputs: Arc<[u8]>,
    pub(super) issuer_pinned: bool,
    pub(super) pin_attempted: bool,
    pub(super) preparation_outcome_known: bool,
    pub(super) delivery_known: bool,
    pub(super) refusal_known: bool,
    pub(super) refusal_record: Option<xlog_cuda::SemanticReplayLoadedMaterial>,
    ordinal: u64,
    total_bytes: u64,
    previous: [u8; 32],
    admission: Option<xlog_cuda::SemanticReplayLoadedMaterial>,
    preparation_record: Option<xlog_cuda::SemanticReplayLoadedMaterial>,
    delivery_record: Option<xlog_cuda::SemanticReplayLoadedMaterial>,
    pub(super) attempt: Option<RecordAttempt>,
}

fn append_fields(target: &mut Vec<u8>, fields: &[&[u8]]) -> PyResult<()> {
    for field in fields {
        let length = u64::try_from(field.len())
            .map_err(|_| invalid("phase record field length exceeds its native index"))?;
        target.extend_from_slice(&length.to_le_bytes());
        target.extend_from_slice(field);
    }
    Ok(())
}

pub(super) fn refusal_payload(fields: [&[u8]; 10]) -> PyResult<Arc<[u8]>> {
    let mut payload = b"xlog.learning-phase.refusal.v1\0".to_vec();
    append_fields(&mut payload, &fields)?;
    Ok(payload.into())
}

impl PhaseRecords {
    pub(super) fn new(fields: &[&[u8]], limits: RecordLimits) -> PyResult<Self> {
        let input_bytes = fields
            .iter()
            .try_fold(INPUT_DOMAIN.len() as u64 + 8 + 24, |total, field| {
                total.checked_add(8)?.checked_add(field.len() as u64)
            });
        let admission_bytes = input_bytes
            .and_then(|input_bytes| input_bytes.checked_add(RECORD_OVERHEAD as u64))
            .ok_or_else(|| invalid("phase admission record size overflowed"))?;
        // Reserve both complete outcomes before admission. The caller's original
        // bounds are never enlarged after model work or source retirement.
        if limits.records < 3
            || admission_bytes > limits.record_bytes
            || admission_bytes
                .checked_add(limits.record_bytes)
                .and_then(|required| required.checked_add(limits.record_bytes))
                .is_none_or(|required| required > limits.total_bytes)
        {
            return Err(invalid(
                "phase record bounds cannot retain the full admission, preparation outcome and delivery",
            ));
        }
        let mut inputs = INPUT_DOMAIN.to_vec();
        append_fields(&mut inputs, fields)?;
        append_fields(&mut inputs, &[&limits.encode()])?;
        let phase_id = Sha256::digest(&inputs).into();
        Ok(Self {
            phase_id,
            limits,
            key: SigningKey::generate(&mut OsRng),
            inputs: inputs.into(),
            issuer_pinned: false,
            pin_attempted: false,
            preparation_outcome_known: false,
            delivery_known: false,
            refusal_known: false,
            refusal_record: None,
            ordinal: 0,
            total_bytes: 0,
            previous: [0; 32],
            admission: None,
            preparation_record: None,
            delivery_record: None,
            attempt: None,
        })
    }

    pub(super) fn issuer(&self) -> [u8; 32] {
        self.key.verifying_key().to_bytes()
    }

    pub(super) fn delivered_identity(
        &self,
    ) -> PyResult<([u8; 32], [u8; 32], xlog_cuda::SemanticReplayLoadedMaterial)> {
        if !self.delivery_known
            || self.ordinal != 3
            || self.attempt.is_some()
            || !self.issuer_pinned
        {
            return Err(invalid(
                "checkpoint issuer requires known full signed Delivery readback",
            ));
        }
        let admission = self
            .admission
            .as_ref()
            .ok_or_else(|| invalid("known Delivery lost its original native Admission"))?;
        Ok((
            self.phase_id,
            self.previous,
            admission
                .slice_payload(0..admission.as_bytes().len())
                .map_err(xlog_err)?,
        ))
    }

    pub(super) fn require_preparation_admission(&self) -> PyResult<()> {
        if !self.issuer_pinned
            || self.ordinal != 1
            || self.attempt.is_some()
            || self.admission.is_none()
        {
            return Err(invalid(
                "private phase allocation requires exact durable readback of its original signed admission",
            ));
        }
        Ok(())
    }

    /// Only the exact, signed, durably read-back Admission authorizes private
    /// native writes. Scientific acceptance is not obtained before those writes.
    pub(super) fn confirmed_admission(&self) -> PyResult<xlog_cuda::SemanticReplayLoadedMaterial> {
        self.require_preparation_admission()?;
        let admission = self
            .admission
            .as_ref()
            .ok_or_else(|| invalid("private phase lost its original confirmed native admission"))?;
        admission
            .slice_payload(0..admission.as_bytes().len())
            .map_err(xlog_err)
    }

    /// A confirmed original pin may still precede the first admission write.
    /// An unresolved write is not permission to issue a second one.
    pub(super) fn preparation_admission_needed(&self) -> PyResult<bool> {
        if !self.pin_attempted
            || !self.issuer_pinned
            || self.attempt.is_some()
            || self.preparation_outcome_known
            || self.ordinal > 1
        {
            return Err(invalid(
                "preparation continuation requires the original confirmed issuer and no unresolved record write",
            ));
        }
        Ok(self.ordinal == 0)
    }

    pub(super) fn check_payload_length(&self, payload_length: usize) -> PyResult<u64> {
        let length = (payload_length as u64)
            .checked_add(RECORD_OVERHEAD as u64)
            .ok_or_else(|| invalid("signed phase record size overflowed"))?;
        let total = self
            .total_bytes
            .checked_add(length)
            .ok_or_else(|| invalid("phase record total size overflowed"))?;
        if length > self.limits.record_bytes || total > self.limits.total_bytes {
            return Err(invalid(
                "complete phase record exceeds its original storage bounds",
            ));
        }
        Ok(total)
    }

    fn begin_total(&self, kind: RecordKind, payload_extent: usize) -> PyResult<u64> {
        if !self.issuer_pinned || self.attempt.is_some() {
            return Err(invalid(
                "phase record requires its independently pinned issuer and resolution of the original write",
            ));
        }
        if self.ordinal >= self.limits.records {
            return Err(invalid("original phase record count is exhausted"));
        }
        let ordered = !self.refusal_known
            && match kind {
                RecordKind::Admission => self.ordinal == 0 && self.admission.is_none(),
                RecordKind::PreparationOutcome => self.ordinal == 1 && self.admission.is_some(),
                RecordKind::Delivery => {
                    self.ordinal == 2 && self.preparation_outcome_known && !self.delivery_known
                }
                RecordKind::TerminalRefusal => {
                    self.admission.is_some()
                        && !self.delivery_known
                        && (self.ordinal == 1
                            || self.ordinal == 2 && self.preparation_outcome_known)
                }
            };
        if !ordered {
            return Err(invalid(
                "phase record changed its original signed lifecycle order",
            ));
        }
        self.check_payload_length(payload_extent)
    }

    fn record_header(
        &self,
        kind: RecordKind,
        payload_extent: usize,
        total: u64,
    ) -> PyResult<[u8; RECORD_PREFIX]> {
        let mut bytes = [0; RECORD_PREFIX];
        let kind = [kind as u8];
        let issuer = self.issuer();
        let ordinal = self.ordinal.to_le_bytes();
        let total = total.to_le_bytes();
        let limits = self.limits.encode();
        let length = u64::try_from(payload_extent)
            .map_err(|_| invalid("phase record field length exceeds its native index"))?
            .to_le_bytes();
        let mut cursor = 0usize;
        for field in [
            RECORD_DOMAIN,
            &kind,
            &self.phase_id,
            &issuer,
            &ordinal,
            &self.previous,
            &total,
            &limits,
            &length,
        ] {
            let end = cursor
                .checked_add(field.len())
                .ok_or_else(|| invalid("signed phase record header overflowed"))?;
            bytes
                .get_mut(cursor..end)
                .ok_or_else(|| invalid("signed phase record header differs"))?
                .copy_from_slice(field);
            cursor = end;
        }
        if cursor != RECORD_PREFIX {
            return Err(invalid("signed phase record header differs"));
        }
        Ok(bytes)
    }

    pub(super) fn prepare_output(
        &self,
        kind: RecordKind,
        payload_extent: usize,
        budget: &xlog_cuda::SemanticReplayMaterialBudget,
    ) -> PyResult<PreparedRecordOutput> {
        let total = self.begin_total(kind, payload_extent)?;
        let header = self.record_header(kind, payload_extent, total)?;
        let metadata = budget
            .reserve_auxiliary(std::mem::size_of::<PreparedRecordOutput>() as u64)
            .map_err(xlog_err)?;
        let output = budget
            .prepare_framed_material_output(RECORD_PREFIX, payload_extent, 64)
            .map_err(xlog_err)?;
        output
            .declare_frame_prefix(RECORD_PREFIX)
            .map_err(xlog_err)?;
        output.write_frame(0, &header).map_err(xlog_err)?;
        output
            .declare_spans(payload_extent, &[(0, payload_extent)])
            .map_err(xlog_err)?;
        Ok(PreparedRecordOutput {
            output,
            header,
            payload_extent,
            signature_written: false,
            record: None,
            _metadata: metadata,
        })
    }

    pub(super) fn begin(
        &mut self,
        kind: RecordKind,
        payload: &[u8],
        budget: &xlog_cuda::SemanticReplayMaterialBudget,
    ) -> PyResult<()> {
        let mut original = self.prepare_output(kind, payload.len(), budget)?;
        original.output.write_model(0, payload).map_err(xlog_err)?;
        let payload = original.output.seal_model().map_err(xlog_err)?;
        self.begin_prepared(kind, &mut original, &payload, budget)
    }

    pub(super) fn begin_prepared(
        &mut self,
        kind: RecordKind,
        original: &mut PreparedRecordOutput,
        payload: &xlog_cuda::SemanticReplayLoadedMaterial,
        budget: &xlog_cuda::SemanticReplayMaterialBudget,
    ) -> PyResult<()> {
        let total = self.begin_total(kind, payload.as_bytes().len())?;
        if original.header != self.record_header(kind, payload.as_bytes().len(), total)? {
            return Err(invalid(
                "signed phase output changed its original issuer or lifecycle",
            ));
        }
        original.output.require_budget(budget).map_err(xlog_err)?;
        original
            .output
            .require_model_payload(payload)
            .map_err(xlog_err)?;
        let bytes = original.finish(&self.key)?;
        self.attempt = Some(RecordAttempt {
            kind,
            ordinal: self.ordinal,
            digest: Sha256::digest(bytes.as_bytes()).into(),
            bytes,
            readback_observed: false,
        });
        Ok(())
    }

    pub(super) fn confirm(&mut self, readback: &Bound<'_, PyAny>) -> PyResult<()> {
        let attempt = self
            .attempt
            .as_mut()
            .ok_or_else(|| invalid("phase record readback has no original write attempt"))?;
        require_guarded_material(readback, &attempt.bytes)?;
        let bytes = attempt.bytes.as_bytes();
        let (message, signature) = bytes.split_at(bytes.len() - 64);
        let signature = ed25519_dalek::Signature::from_slice(signature)
            .map_err(|_| invalid("phase record signature has an invalid encoding"))?;
        self.key
            .verifying_key()
            .verify_strict(message, &signature)
            .map_err(|_| invalid("phase record failed original native issuer verification"))?;
        attempt.readback_observed = true;
        Ok(())
    }

    pub(super) fn advance(&mut self) -> PyResult<()> {
        let attempt = self
            .attempt
            .as_ref()
            .ok_or_else(|| invalid("phase record progress has no retained write"))?;
        if !attempt.readback_observed {
            return Err(invalid(
                "phase record progress requires exact durable readback",
            ));
        }
        self.ordinal = self
            .ordinal
            .checked_add(1)
            .ok_or_else(|| invalid("phase record ordinal overflowed"))?;
        self.total_bytes = self
            .total_bytes
            .checked_add(attempt.bytes.as_bytes().len() as u64)
            .ok_or_else(|| invalid("phase record total size overflowed"))?;
        self.previous = attempt.digest;
        let attempt = self
            .attempt
            .take()
            .expect("confirmed original record attempt");
        match attempt.kind {
            RecordKind::Admission => self.admission = Some(attempt.bytes),
            RecordKind::PreparationOutcome => {
                self.preparation_outcome_known = true;
                self.preparation_record = Some(attempt.bytes);
            }
            RecordKind::Delivery => {
                self.delivery_known = true;
                self.delivery_record = Some(attempt.bytes);
            }
            RecordKind::TerminalRefusal => {
                self.refusal_known = true;
                self.refusal_record = Some(attempt.bytes);
            }
        }
        Ok(())
    }

    pub(super) fn completed_lifecycle(
        &self,
    ) -> PyResult<([Option<xlog_cuda::SemanticReplayLoadedMaterial>; 3], usize)> {
        let prepared = self.preparation_record.is_some();
        if !self.issuer_pinned
            || self.attempt.is_some()
            || self.delivery_known == self.refusal_known
            || self.preparation_outcome_known != prepared
            || self.ordinal != if prepared { 3 } else { 2 }
            || self.delivery_known && !prepared
        {
            return Err(invalid(
                "phase closure requires its complete original terminal readback",
            ));
        }
        let admission = self
            .admission
            .as_ref()
            .ok_or_else(|| invalid("phase closure lost its original confirmed admission"))?;
        let admission = admission
            .slice_payload(0..admission.as_bytes().len())
            .map_err(xlog_err)?;
        let completion = if self.refusal_known {
            self.refusal_record.as_ref()
        } else if self.delivery_known {
            self.delivery_record.as_ref()
        } else {
            return Err(invalid(
                "phase closure precedes known terminal record readback",
            ));
        }
        .ok_or_else(|| invalid("phase closure lost its original terminal signed record"))?;
        let completion = completion
            .slice_payload(0..completion.as_bytes().len())
            .map_err(xlog_err)?;
        if let Some(preparation) = &self.preparation_record {
            let preparation = preparation
                .slice_payload(0..preparation.as_bytes().len())
                .map_err(xlog_err)?;
            Ok(([Some(admission), Some(preparation), Some(completion)], 3))
        } else {
            Ok(([Some(admission), Some(completion), None], 2))
        }
    }
}
