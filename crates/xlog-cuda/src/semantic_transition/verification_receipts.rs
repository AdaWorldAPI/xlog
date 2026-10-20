//! Authenticate original external observations before native acquisition.

use super::*;
use serde_json::{Map, Value};

const RECEIPT_DOMAIN: &[u8] = b"dlm-new.tool-verification.receipt.v1\0";
const EFFECT_DOMAIN: &[u8] = b"dlm-new.local-tool.v1\0";
const TOOL_RESULT_DOMAIN: &[u8] = b"dlm-new.tool-result.receipt.v1\0";

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum OriginalObservationOrigin {
    Action(Identity256),
    Intent,
}

pub(super) struct AuthenticatedVerificationReceipt {
    pub identity: Identity256,
    pub outcome: &'static str,
}

pub(super) struct AuthenticatedTaskResults {
    pub observations: Vec<[u64; 18]>,
    pub receipts: Vec<AuthenticatedVerificationReceipt>,
    pub canonical_payload_identity: Identity256,
    pub file_roster_identity: Identity256,
    pub result_capacity_bytes: u64,
    pub disposition: u64,
    pub allocation_owner: Option<Arc<SemanticReplayMaterialReservation>>,
}

fn invalid_receipt() -> SemanticTransitionError {
    publication_input_error("verification material differs from its original authenticated binding")
}

// The issuer's existing canonical JSON law is sorted, ASCII-escaped JSON with
// integer numbers. Preserve that exact law, including non-BMP surrogate pairs;
// serde_json's ordinary UTF-8 serializer is not the same wire representation.
fn json_string(value: &str, output: &mut Vec<u8>) {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    output.push(b'"');
    for character in value.chars() {
        match character {
            '"' => output.extend_from_slice(b"\\\""),
            '\\' => output.extend_from_slice(b"\\\\"),
            '\u{8}' => output.extend_from_slice(b"\\b"),
            '\u{c}' => output.extend_from_slice(b"\\f"),
            '\n' => output.extend_from_slice(b"\\n"),
            '\r' => output.extend_from_slice(b"\\r"),
            '\t' => output.extend_from_slice(b"\\t"),
            character if (' '..='~').contains(&character) => output.push(character as u8),
            character => {
                for unit in character.encode_utf16(&mut [0; 2]) {
                    output.extend_from_slice(b"\\u");
                    for shift in [12, 8, 4, 0] {
                        output.push(HEX[((*unit >> shift) & 15) as usize]);
                    }
                }
            }
        }
    }
    output.push(b'"');
}

fn write_json(value: &Value, output: &mut Vec<u8>) -> Result<(), SemanticTransitionError> {
    match value {
        Value::Null => output.extend_from_slice(b"null"),
        Value::Bool(true) => output.extend_from_slice(b"true"),
        Value::Bool(false) => output.extend_from_slice(b"false"),
        Value::Number(number) if number.is_i64() || number.is_u64() => {
            output.extend_from_slice(number.to_string().as_bytes());
        }
        Value::String(value) => json_string(value, output),
        Value::Array(values) => {
            output.push(b'[');
            for (ordinal, value) in values.iter().enumerate() {
                if ordinal != 0 {
                    output.push(b',');
                }
                write_json(value, output)?;
            }
            output.push(b']');
        }
        Value::Object(values) => {
            output.push(b'{');
            let mut keys: Vec<_> = values.keys().collect();
            keys.sort_unstable();
            for (ordinal, key) in keys.into_iter().enumerate() {
                if ordinal != 0 {
                    output.push(b',');
                }
                json_string(key, output);
                output.push(b':');
                write_json(&values[key], output)?;
            }
            output.push(b'}');
        }
        _ => return Err(invalid_receipt()),
    }
    Ok(())
}

fn json_bytes(value: &Value) -> Result<Vec<u8>, SemanticTransitionError> {
    let mut bytes = Vec::new();
    write_json(value, &mut bytes)?;
    Ok(bytes)
}

fn json_identity(value: &Value) -> Result<Identity256, SemanticTransitionError> {
    Ok(Identity256::from_bytes(
        Sha256::digest(json_bytes(value)?).into(),
    ))
}

fn object<'a>(
    value: &'a Value,
    fields: &[&str],
) -> Result<&'a Map<String, Value>, SemanticTransitionError> {
    let value = value.as_object().ok_or_else(invalid_receipt)?;
    if value.len() != fields.len() || fields.iter().any(|name| !value.contains_key(*name)) {
        return Err(invalid_receipt());
    }
    Ok(value)
}

fn value_clone_allocation_bound(value: &Value) -> Result<u64, SemanticTransitionError> {
    let add = |left: u64, right: u64| {
        left.checked_add(right)
            .ok_or(SemanticTransitionError::GenerationExhausted)
    };
    match value {
        Value::String(value) => {
            u64::try_from(value.len()).map_err(|_| SemanticTransitionError::GenerationExhausted)
        }
        Value::Number(value) if value.is_i64() || value.is_u64() => Ok(20),
        Value::Number(_) => Err(invalid_receipt()),
        Value::Array(values) => {
            let bytes = values
                .len()
                .checked_mul(size_of::<Value>())
                .ok_or(SemanticTransitionError::GenerationExhausted)?;
            values.iter().try_fold(
                u64::try_from(bytes).map_err(|_| SemanticTransitionError::GenerationExhausted)?,
                |sum, value| add(sum, value_clone_allocation_bound(value)?),
            )
        }
        Value::Object(values) => {
            let node = u64::try_from(OwnedPublicationMaterial::tree_node_bound(
                size_of::<String>(),
                size_of::<Value>(),
            )?)
            .map_err(|_| SemanticTransitionError::GenerationExhausted)?;
            values.iter().try_fold(0u64, |sum, (key, value)| {
                add(
                    add(
                        add(sum, node)?,
                        u64::try_from(key.len())
                            .map_err(|_| SemanticTransitionError::GenerationExhausted)?,
                    )?,
                    value_clone_allocation_bound(value)?,
                )
            })
        }
        _ => Ok(0),
    }
}

fn string(value: &Value) -> Result<&str, SemanticTransitionError> {
    value.as_str().ok_or_else(invalid_receipt)
}

fn integer(value: &Value) -> Result<u64, SemanticTransitionError> {
    value.as_u64().ok_or_else(invalid_receipt)
}

fn hex_bytes(value: &Value) -> Result<Vec<u8>, SemanticTransitionError> {
    let text = string(value)?.as_bytes();
    if text.len() % 2 != 0 {
        return Err(invalid_receipt());
    }
    let nibble = |value| match value {
        b'0'..=b'9' => Ok(value - b'0'),
        b'a'..=b'f' => Ok(value - b'a' + 10),
        _ => Err(invalid_receipt()),
    };
    let mut bytes = Vec::with_capacity(text.len() / 2);
    for pair in text.chunks_exact(2) {
        bytes.push(nibble(pair[0])? * 16 + nibble(pair[1])?);
    }
    Ok(bytes)
}

fn identity(value: &Value) -> Result<Identity256, SemanticTransitionError> {
    Ok(Identity256::from_bytes(
        hex_bytes(value)?
            .try_into()
            .map_err(|_| invalid_receipt())?,
    ))
}

fn hex_identity(value: Identity256) -> Value {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut text = String::with_capacity(64);
    for byte in value.as_bytes() {
        text.push(HEX[(byte >> 4) as usize] as char);
        text.push(HEX[(byte & 15) as usize] as char);
    }
    Value::String(text)
}

fn binding_json(binding: &SemanticTaskVerificationBinding) -> Value {
    Value::Array(vec![
        binding.query.into(),
        binding.observation_record.into(),
        hex_identity(binding.obligation_identity),
        hex_identity(binding.requirement_identity),
        hex_identity(binding.verifier_plan_identity),
        hex_identity(binding.acceptance_condition_identity),
        binding.receipt_family.name().into(),
        binding
            .measurement
            .as_ref()
            .map_or(Value::Null, |measurement| {
                Value::Array(vec![
                    hex_identity(measurement.metric_identity),
                    hex_identity(measurement.comparator_identity),
                    hex_identity(measurement.tolerance_identity),
                    hex_identity(measurement.environment_identity),
                ])
            }),
    ])
}

fn roster<'a>(value: &'a Value, count: usize) -> Result<&'a [Value], SemanticTransitionError> {
    let values = value.as_array().ok_or_else(invalid_receipt)?;
    if values.len() != count {
        return Err(invalid_receipt());
    }
    Ok(values)
}

fn verify_receipt_metadata(
    bytes: &[u8],
    receipt_identity: Identity256,
    binding: &SemanticTaskVerificationBinding,
    outcome: &Value,
    budget: Option<&SemanticReplayMaterialBudget>,
) -> Result<(), SemanticTransitionError> {
    let _scratch = budget
        .map(|budget| {
            budget.reserve_auxiliary(
                SemanticReplayMaterialBudget::json_allocation_bound_with_field_copies(bytes, 2)?,
            )
        })
        .transpose()?;
    let value: Value = serde_json::from_slice(bytes).map_err(|_| invalid_receipt())?;
    if json_bytes(&value)? != bytes {
        return Err(invalid_receipt());
    }
    let metadata = object(
        &value,
        &[
            "kind",
            "identity",
            "outcome",
            "refusal_scope",
            "comparator_identity",
            "metric_identity",
            "tolerance_identity",
            "environment_identity",
            "obligation_identity",
            "verifier_plan_identity",
            "acceptance_condition_identity",
        ],
    )?;
    if string(&metadata["kind"])? != binding.receipt_family.name()
        || identity(&metadata["identity"])? != receipt_identity
        || &metadata["outcome"] != outcome
        || !metadata["refusal_scope"].is_null()
        || identity(&metadata["obligation_identity"])? != binding.obligation_identity
        || identity(&metadata["verifier_plan_identity"])? != binding.verifier_plan_identity
        || identity(&metadata["acceptance_condition_identity"])?
            != binding.acceptance_condition_identity
    {
        return Err(invalid_receipt());
    }
    for (name, expected) in [
        (
            "metric_identity",
            binding.measurement.as_ref().map(|law| law.metric_identity),
        ),
        (
            "comparator_identity",
            binding
                .measurement
                .as_ref()
                .map(|law| law.comparator_identity),
        ),
        (
            "tolerance_identity",
            binding
                .measurement
                .as_ref()
                .map(|law| law.tolerance_identity),
        ),
        (
            "environment_identity",
            binding
                .measurement
                .as_ref()
                .map(|law| law.environment_identity),
        ),
    ] {
        if let Some(expected) = expected {
            if identity(&metadata[name])? != expected {
                return Err(invalid_receipt());
            }
        } else if !metadata[name].is_null() {
            return Err(invalid_receipt());
        }
    }
    Ok(())
}

fn require_retired(
    record: &Map<String, Value>,
    budget: Option<&SemanticReplayMaterialBudget>,
) -> Result<i64, SemanticTransitionError> {
    let exit = record["exit_code"].as_i64().ok_or_else(invalid_receipt)?;
    let retirement = object(
        &record["retirement"],
        &[
            "original_launcher_pid",
            "original_namespace_init_pid",
            "launcher_terminal",
            "namespace_init_terminal",
            "original_pidfd_reaped",
            "original_sync_eof",
        ],
    )?;
    if integer(&retirement["original_launcher_pid"])? == 0
        || integer(&retirement["original_namespace_init_pid"])? == 0
        || retirement["original_launcher_pid"] == retirement["original_namespace_init_pid"]
        || retirement["original_pidfd_reaped"] != Value::Bool(true)
        || retirement["original_sync_eof"] != Value::Bool(true)
    {
        return Err(invalid_receipt());
    }
    let terminal = roster(&retirement["launcher_terminal"], 2)?;
    let code = integer(&terminal[0])?;
    let status = terminal[1].as_i64().ok_or_else(invalid_receipt)?;
    if !(code == 1 && status == exit || (code == 2 || code == 3) && status > 0 && -status == exit) {
        return Err(invalid_receipt());
    }
    let init = roster(&retirement["namespace_init_terminal"], 2)?;
    if !(1..=3).contains(&integer(&init[0])?) || integer(&init[1])? > 255 {
        return Err(invalid_receipt());
    }
    let launch_bytes = string(&record["launch_status_hex"])?.len() / 2;
    let scratch_bytes = launch_bytes
        .checked_mul(1 + size_of::<i64>() + size_of::<u64>())
        .and_then(|bytes| bytes.checked_add(4 * (size_of::<i64>() + size_of::<u64>())))
        .ok_or(SemanticTransitionError::GenerationExhausted)?;
    let _scratch = budget
        .map(|budget| {
            budget.reserve_auxiliary(
                u64::try_from(scratch_bytes)
                    .map_err(|_| SemanticTransitionError::GenerationExhausted)?,
            )
        })
        .transpose()?;
    let launch_status = hex_bytes(&record["launch_status_hex"])?;
    let mut exits = Vec::new();
    let mut children = Vec::new();
    for line in launch_status
        .split(|byte| *byte == b'\n')
        .filter(|line| !line.is_empty())
    {
        let _line_scratch = budget
            .map(|budget| budget.reserve_json_parse(line))
            .transpose()?;
        let value: Value = serde_json::from_slice(line).map_err(|_| invalid_receipt())?;
        let value = value.as_object().ok_or_else(invalid_receipt)?;
        if let Some(value) = value.get("exit-code") {
            exits.push(value.as_i64().ok_or_else(invalid_receipt)?);
        }
        if let Some(value) = value.get("child-pid") {
            children.push(integer(value)?);
        }
    }
    if exits != [exit] || children != [integer(&retirement["original_namespace_init_pid"])?] {
        return Err(invalid_receipt());
    }
    Ok(exit)
}

fn verify_outcome(
    record: &Map<String, Value>,
    command: &Map<String, Value>,
    acceptance: &Value,
    binding: &SemanticTaskVerificationBinding,
    budget: Option<&SemanticReplayMaterialBudget>,
) -> Result<(u64, bool, &'static str), SemanticTransitionError> {
    let exit = require_retired(record, budget)?;
    let output_bytes = (string(&record["stdout_hex"])?.len() / 2)
        .checked_add(string(&record["stderr_hex"])?.len() / 2)
        .ok_or(SemanticTransitionError::GenerationExhausted)?;
    let _output_scratch = budget
        .map(|budget| {
            budget.reserve_auxiliary(
                u64::try_from(output_bytes)
                    .map_err(|_| SemanticTransitionError::GenerationExhausted)?,
            )
        })
        .transpose()?;
    let stdout = hex_bytes(&record["stdout_hex"])?;
    let stderr = hex_bytes(&record["stderr_hex"])?;
    let capacity = integer(&command["output_bytes"])?;
    if stdout.len() as u64 > capacity || stderr.len() as u64 > capacity {
        return Err(invalid_receipt());
    }
    let (outcome, truth, rejected) = if let Some(measurement) = &binding.measurement {
        let metric = object(
            &command["measurement"],
            &["metric", "expected", "tolerance"],
        )?;
        let expected = metric["expected"].as_i64().ok_or_else(invalid_receipt)?;
        let tolerance = metric["tolerance"]
            .as_i64()
            .filter(|value| *value >= 0)
            .ok_or_else(invalid_receipt)?;
        let accepted = object(acceptance, &["kind", "metric", "expected", "tolerance"])?;
        if accepted["kind"] != "exit-zero-stdout-signed-i64-with-absolute-tolerance/1"
            || accepted["metric"] != metric["metric"]
            || accepted["expected"] != metric["expected"]
            || accepted["tolerance"] != metric["tolerance"]
        {
            return Err(invalid_receipt());
        }
        let laws = roster(&record["measurement_laws"], 3)?;
        let fixed_strings = [
            "kind",
            "kind",
            "metric",
            "kind",
            "value",
            "absolute-integer-difference-at-most/1",
            "stdout-signed-i64/1",
            "integer-tolerance/1",
        ]
        .iter()
        .try_fold(0u64, |sum, value| {
            sum.checked_add(value.len() as u64)
                .ok_or(SemanticTransitionError::GenerationExhausted)
        })?;
        let node = u64::try_from(OwnedPublicationMaterial::tree_node_bound(
            size_of::<String>(),
            size_of::<Value>(),
        )?)
        .map_err(|_| SemanticTransitionError::GenerationExhausted)?;
        let metric_bytes = value_clone_allocation_bound(&metric["metric"])?;
        let law_bytes = fixed_strings
            .checked_add(20)
            .and_then(|n| node.checked_mul(5).and_then(|nodes| n.checked_add(nodes)))
            .and_then(|n| n.checked_add(metric_bytes))
            .ok_or(SemanticTransitionError::GenerationExhausted)?;
        let _law_scratch = budget
            .map(|budget| budget.reserve_auxiliary(law_bytes))
            .transpose()?;
        if laws[0] != serde_json::json!({"kind": "absolute-integer-difference-at-most/1"})
            || laws[1]
                != serde_json::json!({"kind": "stdout-signed-i64/1", "metric": metric["metric"]})
            || laws[2] != serde_json::json!({"kind": "integer-tolerance/1", "value": tolerance})
            || json_identity(&laws[0])? != measurement.comparator_identity
            || json_identity(&laws[1])? != measurement.metric_identity
            || json_identity(&laws[2])? != measurement.tolerance_identity
            || identity(&record["environment_sha256"])? != measurement.environment_identity
        {
            return Err(invalid_receipt());
        }
        let number = stdout.strip_suffix(b"\n").unwrap_or(&stdout);
        let digits = number.strip_prefix(b"-").unwrap_or(number);
        let canonical_integer = !digits.is_empty()
            && digits.len() <= 19
            && digits.iter().all(u8::is_ascii_digit)
            && (digits.len() == 1 || digits[0] != b'0');
        let actual = canonical_integer
            .then(|| std::str::from_utf8(number).ok()?.parse::<i64>().ok())
            .flatten();
        match actual.filter(|actual| {
            exit == 0 && (i128::from(*actual) - i128::from(expected)).abs() <= i128::from(tolerance)
        }) {
            Some(actual) if actual == expected => ("EXACT", 1, false),
            Some(_) => ("REPRODUCIBLE_NOT_EXACT", 1, false),
            None => ("REJECTED", 2, true),
        }
    } else {
        let accepted = object(acceptance, &["kind", "stdout_sha256"])?;
        if accepted["kind"] != "exit-zero-and-exact-stdout-sha256/1"
            || accepted["stdout_sha256"] != command["success_stdout_sha256"]
            || !command["measurement"].is_null()
            || !record["measurement_laws"].is_null()
        {
            return Err(invalid_receipt());
        }
        if exit == 0
            && Identity256::from_bytes(Sha256::digest(&stdout).into())
                == identity(&accepted["stdout_sha256"])?
        {
            ("PASS", 1, false)
        } else {
            ("FAIL", 2, false)
        }
    };
    if string(&record["outcome"])? != outcome {
        return Err(invalid_receipt());
    }
    Ok((truth, rejected, outcome))
}

pub(super) fn verify_acquired_receipts<'a, I>(
    ground: &SemanticTaskGround,
    materials: I,
    stable_intent: Identity256,
    original_effect: &[u8],
    original_native_payload: &[u8],
    origin: OriginalObservationOrigin,
    key: &[u8; 32],
    budget: Option<&SemanticReplayMaterialBudget>,
) -> Result<AuthenticatedTaskResults, SemanticTransitionError>
where
    I: Clone + ExactSizeIterator<Item = (Identity256, &'a [u8], Option<&'a [u8]>)>,
{
    let SemanticTaskGround::Coding {
        bindings,
        requirements,
    } = ground
    else {
        return Err(invalid_receipt());
    };
    if materials.len() != bindings.len()
        || bindings.is_empty()
        || stable_intent == Identity256::default()
        || original_native_payload.len() < 48
        || original_effect.len() != EFFECT_DOMAIN.len() + 64
        || !original_effect.starts_with(EFFECT_DOMAIN)
        || original_effect[original_effect.len() - 32..] != original_native_payload[..32]
    {
        return Err(invalid_receipt());
    }
    let (origin_kind, origin_identity, support_domain): (u64, _, &[u8]) = match origin {
        OriginalObservationOrigin::Action(identity) if identity != Identity256::default() => {
            (1, identity, b"xlog.authenticated-task-observation.v1\0")
        }
        OriginalObservationOrigin::Intent => (
            2,
            stable_intent,
            b"xlog.authenticated-intent-observation.v1\0",
        ),
        _ => return Err(invalid_receipt()),
    };
    let mut owned_bytes = (3 * size_of::<Value>() + "coding".len()) as u64;
    for row in requirements {
        let nodes = 6usize
            .checked_add(row.required_goal_ordinals.len())
            .and_then(|n| n.checked_add(row.required_binding_ordinals.len()))
            .ok_or(SemanticTransitionError::GenerationExhausted)?;
        owned_bytes = owned_bytes
            .checked_add(
                (nodes as u64)
                    .checked_mul(size_of::<Value>() as u64)
                    .ok_or(SemanticTransitionError::GenerationExhausted)?,
            )
            .ok_or(SemanticTransitionError::GenerationExhausted)?;
    }
    for binding in bindings {
        let binding_bytes =
            (size_of::<Value>() * 8 + 2 * 20 + 4 * 64 + binding.receipt_family.name().len())
                .checked_add(if binding.measurement.is_some() {
                    4 * size_of::<Value>() + 4 * 64
                } else {
                    0
                })
                .ok_or(SemanticTransitionError::GenerationExhausted)?;
        owned_bytes = owned_bytes
            .checked_add(
                u64::try_from(binding_bytes)
                    .map_err(|_| SemanticTransitionError::GenerationExhausted)?
                    .checked_mul(2)
                    .ok_or(SemanticTransitionError::GenerationExhausted)?,
            )
            .and_then(|n| {
                n.checked_add(
                    (size_of::<Value>()
                        + size_of::<[u64; 18]>()
                        + size_of::<AuthenticatedVerificationReceipt>()) as u64,
                )
            })
            .and_then(|n| {
                n.checked_add(
                    OwnedPublicationMaterial::tree_node_bound(size_of::<[u8; 32]>(), 0).ok()?
                        as u64,
                )
            })
            .ok_or(SemanticTransitionError::GenerationExhausted)?;
    }
    let allocation_owner = budget
        .map(|budget| budget.reserve_auxiliary(owned_bytes).map(Arc::new))
        .transpose()?;
    let plan_digest = Identity256::from_bytes(
        original_native_payload[..32]
            .try_into()
            .map_err(|_| invalid_receipt())?,
    );
    let payload_digest = Identity256::from_bytes(Sha256::digest(original_native_payload).into());
    let original_ground = Value::Array(vec![
        "coding".into(),
        Value::Array(bindings.iter().map(binding_json).collect()),
        Value::Array(
            requirements
                .iter()
                .map(|row| {
                    Value::Array(vec![
                        row.query.into(),
                        Value::Array(
                            row.required_goal_ordinals
                                .iter()
                                .copied()
                                .map(Value::from)
                                .collect(),
                        ),
                        Value::Array(
                            row.required_binding_ordinals
                                .iter()
                                .copied()
                                .map(Value::from)
                                .collect(),
                        ),
                    ])
                })
                .collect(),
        ),
    ]);
    let mut observations = Vec::with_capacity(bindings.len());
    let mut receipts = Vec::with_capacity(bindings.len());
    let mut receipt_identities = BTreeSet::new();
    let mut candidate_identity = None;
    let mut result_capacity_bytes = None;
    let mut disposition = 1;
    let total_material_bytes = materials.clone().try_fold(0_u64, |sum, (_, material, _)| {
        sum.checked_add(material.len() as u64)
            .ok_or_else(invalid_receipt)
    })?;
    for (ordinal, (binding, (receipt_identity, material, receipt_metadata))) in
        bindings.iter().zip(materials).enumerate()
    {
        if (matches!(origin, OriginalObservationOrigin::Action(_)) && receipt_metadata.is_none())
            || !receipt_identities.insert(*receipt_identity.as_bytes())
            || Identity256::from_bytes(Sha256::digest(material).into()) != receipt_identity
            || material.len() <= RECEIPT_DOMAIN.len() + 32
            || !material.starts_with(RECEIPT_DOMAIN)
        {
            return Err(invalid_receipt());
        }
        let mac_offset = material.len() - 32;
        if !super::late_replay::receiver_mac_matches(
            &material[..mac_offset],
            &material[mac_offset..],
            key,
        ) {
            return Err(invalid_receipt());
        }
        let raw = &material[RECEIPT_DOMAIN.len()..mac_offset];
        let _receipt_scratch = budget
            .map(|budget| {
                budget.reserve_auxiliary(
                    SemanticReplayMaterialBudget::json_allocation_bound_with_field_copies(raw, 2)?,
                )
            })
            .transpose()?;
        let value: Value = serde_json::from_slice(raw).map_err(|_| invalid_receipt())?;
        if json_bytes(&value)? != raw {
            return Err(invalid_receipt());
        }
        let record = object(
            &value,
            &[
                "stable_identity",
                "plan_sha256",
                "payload_sha256",
                "canonical_payload_sha256",
                "file_identities",
                "plan",
                "binding_ordinal",
                "binding",
                "command",
                "exit_code",
                "retirement",
                "launch_status_hex",
                "stdout_hex",
                "stderr_hex",
                "outcome",
                "measurement_laws",
                "environment_sha256",
            ],
        )?;
        let plan = object(
            &record["plan"],
            &[
                "kind",
                "workspace_root",
                "repository_paths",
                "environment",
                "max_payload_bytes",
                "max_result_bytes",
                "commands",
                "environment_root",
                "environment_file_capacity",
                "environment_byte_capacity",
                "scratch_capacity_bytes",
                "tokenizer",
                "candidate_prefix_begin",
                "candidate_prefix_end",
                "projection",
                "max_total_file_bytes",
                "resources",
                "task_ground",
                "requirements",
                "checkpoint_root",
                "checkpoint_file_capacity_bytes",
                "repository_commit",
                "executables",
                "environment_sha256",
                "qwen_snapshot",
                "request_sha256",
                "verifiers",
                "acceptance_conditions",
            ],
        )?;
        if plan["kind"] != "dlm-new.tool-plan/2"
            || plan["projection"] != "dlm-new.pinned-bytelevel-canonical-files/1"
            || total_material_bytes > integer(&plan["max_result_bytes"])?
        {
            return Err(invalid_receipt());
        }
        let capacity = integer(&plan["max_result_bytes"])?;
        if result_capacity_bytes
            .replace(capacity)
            .is_some_and(|previous| previous != capacity)
        {
            return Err(invalid_receipt());
        }
        let count = bindings.len();
        let commands = roster(plan.get("commands").ok_or_else(invalid_receipt)?, count)?;
        let verifiers = roster(plan.get("verifiers").ok_or_else(invalid_receipt)?, count)?;
        let acceptances = roster(
            plan.get("acceptance_conditions")
                .ok_or_else(invalid_receipt)?,
            count,
        )?;
        let requirements = roster(plan.get("requirements").ok_or_else(invalid_receipt)?, count)?;
        if identity(&record["stable_identity"])? != stable_intent
            || identity(&record["plan_sha256"])? != plan_digest
            || json_identity(&record["plan"])? != plan_digest
            || identity(&record["payload_sha256"])? != payload_digest
            || plan.get("task_ground") != Some(&original_ground)
            || integer(&record["binding_ordinal"])? != ordinal as u64
            || record["binding"] != binding_json(binding)
            || record["command"] != commands[ordinal]
            || json_identity(&verifiers[ordinal])? != binding.verifier_plan_identity
            || json_identity(&acceptances[ordinal])? != binding.acceptance_condition_identity
            || Identity256::from_bytes(
                Sha256::digest(string(&requirements[ordinal])?.as_bytes()).into(),
            ) != binding.requirement_identity
            || plan.get("environment_sha256") != Some(&record["environment_sha256"])
        {
            return Err(invalid_receipt());
        }
        let begin = integer(
            plan.get("candidate_prefix_begin")
                .ok_or_else(invalid_receipt)?,
        )?;
        let end = integer(
            plan.get("candidate_prefix_end")
                .ok_or_else(invalid_receipt)?,
        )?;
        let wire_begin = u64::from_le_bytes(
            original_native_payload[32..40]
                .try_into()
                .map_err(|_| invalid_receipt())?,
        );
        let wire_count = u64::from_le_bytes(
            original_native_payload[40..48]
                .try_into()
                .map_err(|_| invalid_receipt())?,
        );
        if begin != wire_begin
            || end
                .checked_sub(begin)
                .is_none_or(|capacity| wire_count > capacity)
            || wire_count
                .checked_mul(8)
                .and_then(|size| size.checked_add(48))
                != Some(original_native_payload.len() as u64)
        {
            return Err(invalid_receipt());
        }
        let files = record["file_identities"]
            .as_array()
            .ok_or_else(invalid_receipt)?;
        let paths = roster(
            plan.get("repository_paths").ok_or_else(invalid_receipt)?,
            files.len(),
        )?;
        let mut total = 0_u64;
        for (file, path) in files.iter().zip(paths) {
            let file = object(file, &["path", "sha256", "bytes"])?;
            if &file["path"] != path {
                return Err(invalid_receipt());
            }
            identity(&file["sha256"])?;
            total = total
                .checked_add(integer(&file["bytes"])?)
                .ok_or_else(invalid_receipt)?;
        }
        if total
            > integer(
                plan.get("max_total_file_bytes")
                    .ok_or_else(invalid_receipt)?,
            )?
        {
            return Err(invalid_receipt());
        }
        let candidate = (
            identity(&record["canonical_payload_sha256"])?,
            json_identity(&record["file_identities"])?,
        );
        if candidate_identity
            .replace(candidate)
            .is_some_and(|previous| previous != candidate)
        {
            return Err(invalid_receipt());
        }
        let command = object(
            &commands[ordinal],
            &[
                "family",
                "argv",
                "wall_seconds",
                "cpu_seconds_per_process",
                "address_space_bytes_per_process",
                "output_bytes",
                "success_stdout_sha256",
                "measurement",
            ],
        )?;
        let verifier = object(
            &verifiers[ordinal],
            &[
                "kind",
                "command",
                "executable_sha256",
                "environment_sha256",
                "environment",
                "resources",
                "scratch_capacity_bytes",
                "projection",
            ],
        )?;
        let argv = command["argv"]
            .as_array()
            .filter(|argv| !argv.is_empty())
            .ok_or_else(invalid_receipt)?;
        let executable = string(&argv[0])?;
        let executable_bytes = "environment:"
            .len()
            .checked_add(executable.len())
            .and_then(|n| n.checked_mul(2))
            .ok_or(SemanticTransitionError::GenerationExhausted)?;
        let _executable_scratch = budget
            .map(|budget| {
                budget.reserve_auxiliary(
                    u64::try_from(executable_bytes)
                        .map_err(|_| SemanticTransitionError::GenerationExhausted)?,
                )
            })
            .transpose()?;
        let executables = plan
            .get("executables")
            .and_then(Value::as_object)
            .ok_or_else(invalid_receipt)?;
        if string(&command["family"])? != binding.receipt_family.name()
            || verifier["kind"] != "dlm-new.frozen-verifier/1"
            || verifier["command"] != commands[ordinal]
            || executables.get(&format!("environment:{executable}"))
                != Some(&verifier["executable_sha256"])
            || verifier["environment_sha256"] != record["environment_sha256"]
            || [
                "environment",
                "resources",
                "scratch_capacity_bytes",
                "projection",
            ]
            .iter()
            .any(|name| plan.get(*name) != verifier.get(*name))
        {
            return Err(invalid_receipt());
        }
        identity(&verifier["executable_sha256"])?;
        identity(&verifier["environment_sha256"])?;
        let (truth, rejected, outcome) =
            verify_outcome(record, command, &acceptances[ordinal], binding, budget)?;
        if let Some(receipt_metadata) = receipt_metadata {
            verify_receipt_metadata(
                receipt_metadata,
                receipt_identity,
                binding,
                &record["outcome"],
                budget,
            )?;
        }
        if rejected {
            disposition = 2;
        }
        let mut digest = Sha256::new();
        digest.update(support_domain);
        digest.update(origin_identity.as_bytes());
        digest.update(stable_intent.as_bytes());
        digest.update((ordinal as u64).to_le_bytes());
        digest.update(json_bytes(&record["binding"])?);
        digest.update(receipt_identity.as_bytes());
        let support_identity = Identity256::from_bytes(digest.finalize().into());
        let mut words = [0; 18];
        words[0] = origin_kind;
        words[1] = truth;
        for (offset, identity) in [
            (2, receipt_identity),
            (6, origin_identity),
            (10, stable_intent),
            (14, support_identity),
        ] {
            for (word, bytes) in words[offset..offset + 4]
                .iter_mut()
                .zip(identity.as_bytes().chunks_exact(8))
            {
                *word = u64::from_le_bytes(bytes.try_into().map_err(|_| invalid_receipt())?);
            }
        }
        observations.push(words);
        receipts.push(AuthenticatedVerificationReceipt {
            identity: receipt_identity,
            outcome,
        });
    }
    let (canonical_payload_identity, file_roster_identity) =
        candidate_identity.ok_or_else(invalid_receipt)?;
    Ok(AuthenticatedTaskResults {
        observations,
        receipts,
        canonical_payload_identity,
        file_roster_identity,
        result_capacity_bytes: result_capacity_bytes.ok_or_else(invalid_receipt)?,
        disposition,
        allocation_owner,
    })
}

/// Authenticate the receiver's complete original result without an action row.
///
/// The signed wrapper carries the actual verification material bytes. Its items
/// are not reconstructed episode receipts: the same verifier authenticates each
/// original signed record against the frozen native task before the wrapper's
/// redundant metadata is compared with those verified facts.
pub(super) fn verify_completed_tool_result(
    ground: &SemanticTaskGround,
    result: &[u8],
    stable_intent: Identity256,
    recipient: Identity256,
    original_effect: &[u8],
    original_native_payload: &[u8],
    key: &[u8; 32],
    budget: Option<&SemanticReplayMaterialBudget>,
) -> Result<AuthenticatedTaskResults, SemanticTransitionError> {
    let bindings = ground.bindings();
    let header_bytes = TOOL_RESULT_DOMAIN.len() + 4 * 32 + size_of::<u64>();
    if result.len() <= header_bytes + 32
        || !result.starts_with(TOOL_RESULT_DOMAIN)
        || original_effect.len() != EFFECT_DOMAIN.len() + 64
        || !original_effect.starts_with(EFFECT_DOMAIN)
    {
        return Err(invalid_receipt());
    }
    let mac_offset = result.len() - 32;
    let encoded = &result[header_bytes..mac_offset];
    if u64::from_le_bytes(
        result[header_bytes - 8..header_bytes]
            .try_into()
            .map_err(|_| invalid_receipt())?,
    ) != encoded.len() as u64
        || !super::late_replay::receiver_mac_matches(
            &result[..mac_offset],
            &result[mac_offset..],
            key,
        )
    {
        return Err(invalid_receipt());
    }
    for (actual, expected) in result[TOOL_RESULT_DOMAIN.len()..header_bytes - 8]
        .chunks_exact(32)
        .zip([
            stable_intent,
            recipient,
            Identity256::from_bytes(Sha256::digest(original_effect).into()),
            Identity256::from_bytes(Sha256::digest(original_native_payload).into()),
        ])
    {
        if actual != expected.as_bytes() {
            return Err(invalid_receipt());
        }
    }
    let _json_owner = budget
        .map(|budget| {
            budget.reserve_auxiliary(
                SemanticReplayMaterialBudget::json_allocation_bound_with_field_copies(encoded, 2)?,
            )
        })
        .transpose()?;
    let value: Value = serde_json::from_slice(encoded).map_err(|_| invalid_receipt())?;
    if json_bytes(&value)? != encoded {
        return Err(invalid_receipt());
    }
    let wrapper = object(
        &value,
        &[
            "kind",
            "plan_sha256",
            "receipts",
            "file_identities",
            "canonical_payload_sha256",
        ],
    )?;
    if wrapper["kind"] != "dlm-new.tool-result/1"
        || identity(&wrapper["plan_sha256"])?.as_bytes()
            != &original_effect[original_effect.len() - 32..]
    {
        return Err(invalid_receipt());
    }
    let items = roster(&wrapper["receipts"], bindings.len())?;
    let material_bytes = items.iter().try_fold(0u64, |sum, item| {
        let item = object(
            item,
            &[
                "kind",
                "identity",
                "outcome",
                "material_hex",
                "measurement_law_identities",
            ],
        )?;
        let extent = string(&item["material_hex"])?.len();
        if extent == 0 || extent % 2 != 0 {
            return Err(invalid_receipt());
        }
        sum.checked_add((extent / 2) as u64)
            .ok_or(SemanticTransitionError::GenerationExhausted)
    })?;
    let material_bytes = material_bytes
        .checked_add(
            (items.len() as u64)
                .checked_mul(size_of::<(Identity256, Vec<u8>)>() as u64)
                .ok_or(SemanticTransitionError::GenerationExhausted)?,
        )
        .ok_or(SemanticTransitionError::GenerationExhausted)?;
    let _material_owner = budget
        .map(|budget| budget.reserve_auxiliary(material_bytes))
        .transpose()?;
    let mut materials = Vec::with_capacity(items.len());
    for (binding, item) in bindings.iter().zip(items) {
        let item = item.as_object().ok_or_else(invalid_receipt)?;
        if string(&item["kind"])? != binding.receipt_family.name() {
            return Err(invalid_receipt());
        }
        if let Some(measurement) = &binding.measurement {
            for (actual, expected) in roster(&item["measurement_law_identities"], 3)?.iter().zip([
                measurement.comparator_identity,
                measurement.metric_identity,
                measurement.tolerance_identity,
            ]) {
                if identity(actual)? != expected {
                    return Err(invalid_receipt());
                }
            }
        } else if !item["measurement_law_identities"].is_null() {
            return Err(invalid_receipt());
        }
        materials.push((
            identity(&item["identity"])?,
            hex_bytes(&item["material_hex"])?,
        ));
    }
    let authenticated = verify_acquired_receipts(
        ground,
        materials
            .iter()
            .map(|(identity, material)| (*identity, material.as_slice(), None)),
        stable_intent,
        original_effect,
        original_native_payload,
        OriginalObservationOrigin::Intent,
        key,
        budget,
    )?;
    if result.len() as u64 > authenticated.result_capacity_bytes
        || identity(&wrapper["canonical_payload_sha256"])?
            != authenticated.canonical_payload_identity
        || json_identity(&wrapper["file_identities"])? != authenticated.file_roster_identity
    {
        return Err(invalid_receipt());
    }
    for (item, receipt) in items.iter().zip(&authenticated.receipts) {
        let item = item.as_object().ok_or_else(invalid_receipt)?;
        if identity(&item["identity"])? != receipt.identity
            || string(&item["outcome"])? != receipt.outcome
        {
            return Err(invalid_receipt());
        }
    }
    Ok(authenticated)
}
