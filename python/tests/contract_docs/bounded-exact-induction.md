# Bounded Exact Induction

This document describes XLOG's **bounded exact-induction engine** — the non-gradient,
GPU-native path for enumerating all `(left, right)` candidate pairs across four
fixed 2-body Datalog rule templates and returning top-K per template with full
structured metadata.

Use `pyxlog.ilp.induce_exact(..., backend="native")` for the production scorer.
The `backend="python"` reference runs a host-orchestrated
`set_rule_mask`/`evaluate`/`batch_fact_membership_device` loop and requires
`XLOG_ALLOW_PYTHON_ILP_REFERENCE=1` for explicit parity or compatibility work.

- **Crate:** `crates/xlog-induce/`
- **CUDA kernels:** `crates/xlog-cuda/kernels/ilp_exact.cu` (module `xlog_ilp_exact`, entries
  `ilp_exact_score` for `U64` and `ilp_exact_score_u32` for `U32` /
  `Symbol`)
- **Provider launcher:** `crates/xlog-cuda/src/provider/ilp_exact.rs`
- **pyxlog bridge:** `crates/pyxlog/src/ilp_exact.rs` +
  `crates/pyxlog/python/pyxlog/ilp/exact_induce.py`
- **Parity tests:** `python/tests/test_ilp_exact_induce.py`
- **Related:** [Differentiable ILP (dILP)](../../../ROADMAP.md) — the gradient-trained
  counterpart; this engine is exact / bounded and does not use learnable masks.

## Design Goals

1. **No host round trips during the candidate sweep.** Setup host-to-device and device-to-device uploads
   scale with candidate metadata and relation rows; the `(topology, L, R)` sweep itself performs
   no host/device round trips. The production `ilp_exact_score_topk` path
   reduces on device and the tracked device-to-host counter on `CudaKernelProvider`
   ticks **once for a nontrivial scoring call** for the compact selected-row
   export, independent of candidate count. Empty candidates, zero positives or
   zero `k_per_topology` return without that export. This bounded export means
   the complete call is not a ZERO TRANSFER operation.
2. **Per-topology semantic isolation.** Each `(topology, L, R)` triple is
   scored against the topology's rule template in isolation. There is no
   interference across topologies — a direct fix for the Python prototype's
   stale-mask contamination bug (see [Semantics](#semantics) below).
3. **Deterministic output.** Integer counts only; each scoring block owns one
   output slot. The device selector ranks by descending positive coverage,
   ascending negative coverage, then ascending left and right candidate indices.
   The host maps the selected indices to relation identifiers without re-ranking.
4. **Surgical crate boundary.** `xlog-induce` speaks `RelId` +
   `CudaBuffer` handles; name resolution and DLPack unwrapping happen at
   the pyxlog boundary.

## Topologies

The engine scores against four fixed 2-body rule templates. For a head
relation `H(X, Y)` and candidate relations `L` (left body) and `R` (right
body), a query pair `(qx, qy)` is covered by a topology as follows:

| Topology | Rule | Coverage predicate |
|---|---|---|
| `chain`  | `H(X,Y) :- L(X,Z), R(Z,Y)` | ∃ *z* such that `(qx, z) ∈ L ∧ (z, qy) ∈ R` |
| `star`   | `H(X,Y) :- L(X,Y), R(X,Y)` | `(qx, qy) ∈ L ∧ (qx, qy) ∈ R` |
| `fanout` | `H(X,Y) :- L(X,Z), R(X,Y)` | ∃ _ such that `(qx, _) ∈ L ∧ (qx, qy) ∈ R` |
| `fanin`  | `H(X,Y) :- L(X,Y), R(Z,Y)` | `(qx, qy) ∈ L ∧ ∃ z such that (z, qy) ∈ R` |

The kernel implements these predicates directly — it does **not** go
through `evaluate()` / `set_rule_mask`. This is what guarantees per-topology
isolation.

## Architecture

```
Python user
    │
    ▼
pyxlog.ilp.induce_exact(prog, backend="native", ...)
    │   (name resolution, DLPack unwrap, empty-negatives synthesis)
    ▼
CompiledIlpProgram::induce_exact_native  (crates/pyxlog/src/ilp_exact.rs)
    │   (build InduceExactRequest with RelId + &CudaBuffer handles)
    ▼
xlog_induce::induce_exact(provider, &request)  (crates/xlog-induce/src/lib.rs)
    │
    ├──► validate::classify_request   (empty cands / zero positives → empty result)
    │
    ▼
CudaKernelProvider::ilp_exact_score_topk(...)  (crates/xlog-cuda/src/provider/ilp_exact.rs)
    │
    ├── Setup (not counted as device-to-host transfer):
    │     - device-to-device concat candidate arg0/arg1 columns
    │     - host-to-device upload cand_offsets (small prefix-sum array)
    │     - Alloc output pos_covered/neg_covered arrays
    │
    ├── Launch typed kernel  (crates/xlog-cuda/kernels/ilp_exact.cu)
    │     U64    → `ilp_exact_score`
    │     U32    → `ilp_exact_score_u32`
    │     Symbol → `ilp_exact_score_u32` with logical schema preserved
    │     grid  = (C, C, 4)       — one block per (L, R, topology)
    │     block = (256, 1, 1)
    │     Each block writes exactly one slot in pos_covered and neg_covered.
    │
    ├── Device top-K selection over pos_covered / neg_covered
    │
    └── Download (1 tracked device-to-host transfer):
          compact selected top-K rows as u32 field tuples
    │
    ▼
xlog_induce::score_and_reduce  (map selected indices to relation identifiers)
    │
    ▼
ExactInductionResult { candidates: Vec<ScoredCandidate>, total_scored, ... }
    │
    ▼
pyxlog wrapper repackages into Python ExactInductionResult / ScoredCandidate
```

## Launch Geometry

- **Grid dims:** `(C, C, 4)` blocks addressed as `(L, R, topology)`, where
  `C` is the number of candidate relations in the request and topology
  enumeration is `chain=0, star=1, fanout=2, fanin=3`.
- **Block dims:** `(256, 1, 1)` threads. Each block cooperatively scans all
  positive and negative query pairs for its single `(topology, L, R)`
  triple.
- **Output slot:** `slot = topology * (C * C) + L * C + R`. Each block
  writes exactly one slot in each of `pos_covered` and `neg_covered`, so
  there are no cross-block atomics on the scoring path. Shared-memory
  pair-halving reduction is used *within* a block to sum per-thread
  coverage counts.

For example, at C ≈ 5–20 and |P|+|N| ≈ 20–50 the launch is modest
(≤ 1 600 blocks × 256 threads) but ample for occupancy. The inner
per-thread work is O(|L|·|R|) for `chain` and O(|L|+|R|) for the other
three topologies. Runtime depends on the actual row counts and device.

## Data Layout

| Buffer | Type / length | Contents |
|---|---|---|
| `cand_arg0`, `cand_arg1` | `u64` or `u32` × total_rows | Concatenated device-to-device-copied arg0 / arg1 columns of all candidate relations |
| `cand_offsets` | `u32` × (C+1) | Exclusive prefix-sum of candidate row counts; host-to-device uploaded once per call |
| `pos_arg0`, `pos_arg1` | `u64` or `u32` × num_pos | Device-resident positive query pairs (DLPack-imported from the caller's torch tensors) |
| `neg_arg0`, `neg_arg1` | `u64` or `u32` × num_neg | Same for negatives. When the caller passes no negatives, the engine materializes a zero-row pair buffer with the positive pair type so the kernel signature stays uniform. |
| `pos_covered`, `neg_covered` | `u32` × (4·C·C) | Device count arrays; kernel writes each slot exactly once. |
| Selected output | `u32` × (4·K·9) | Device top-K rows, exported once; each row carries topology, left/right indices, coverage, rank, next coverage and tie-class size. |

Column type dispatch is explicit at both native boundaries:

- `xlog_induce::induce_exact` validates that positives, negatives, and every
  candidate buffer are arity-2 pairs with one uniform logical type: `U64`,
  `U32`, or `Symbol`.
- The provider's shared `ilp_exact_score_device` path repeats that validation
  seam before choosing the physical kernel.
- `U64` buffers launch `ilp_exact_score` over `uint64_t` columns.
- `U32` buffers launch `ilp_exact_score_u32` over `uint32_t` columns.
- `Symbol` buffers also launch `ilp_exact_score_u32`, but the `CudaBuffer`
  schema remains `Symbol` and mixed `U32`/`Symbol` requests fail with typed
  diagnostics rather than silently narrowing.

The pyxlog DLPack layer accepts physical 32-bit tensor IDs for `symbol`
schemas only through schema-checked import. This preserves public logical
types while allowing CUDA-resident symbol-id tensors to reach the native scorer
without host widening.

## Semantics

The Python prototype's `backend="python"` loop calls `set_rule_mask(mask_name,
...)` **per inner iteration**, updating only the current topology's mask
while leaving the other three topologies' masks at whatever state they
last held. Because `evaluate()` derives facts from *all* currently-set
masks, stale masks from earlier outer-loop iterations leak derivations
into later topologies' coverage counts.

The native kernel bypasses `evaluate()` and `set_rule_mask` entirely:
it implements the four topology predicates directly on the candidate
row sets. Per-topology isolation is a structural property, not a
runtime hygiene requirement.

The Python prototype gained an opt-in `strict_per_topology: bool = False`
parameter that zeroes the three "other" topology masks before each
topology's inner loop, yielding the same per-topology-isolated scoring
that the kernel produces by construction. **Default is `False`** for
historical reference behavior. The parity test sets
`strict_per_topology=True` explicitly to match the kernel.

## Device-to-Host Transfer Budget

The kernel is designed around the xlog-native device-to-host transfer counter
(`CudaKernelProvider::d2h_transfer_count`, exposed to Python as
`prog.d2h_transfer_count()`).

- **Counted transfers per nontrivial production `induce_exact` call: 1** (one compact
  selected-row export). The transfer count does not grow with candidate pairs;
  the exported allocation contains `4*K*9` `u32` words. The parity test
  `test_induce_exact_native_does_not_scale_device_to_host_with_candidate_pairs`
  allows at most two additional calls when increasing the candidate count. That
  scaling assertion alone does not prove an exact transfer count or zero bytes.
- Setup host-to-device (`cand_offsets`) and device-to-device (candidate column concatenation)
  are not counted as device-to-host transfers.
- Setup reads that would otherwise require a device-to-host transfer (e.g. relation row
  counts) go through the host-side `CudaBuffer::cached_row_count()`
  cache. This is a load-bearing invariant — see the next section.

### `clone_buffer` cached-row-count propagation

Every relation buffer in `CompiledIlpProgram`'s executor store was
deep-copied via `CudaKernelProvider::clone_buffer` on insertion
(`CompiledIlpProgram::put_relation` clones the imported `buffer` into
`live_buffer` before `executor.put_relation`).
`clone_buffer` calls `set_cached_row_count_if_unset(source.cached_row_count())`
on the clone when the source has a populated cache, preserving the
host-visible count across clones. The scorer requires this cached metadata
rather than downloading a row count or treating row capacity as logical extent.

## Validation Contract

Buffer-level invariants enforced by `xlog_induce::induce_exact`:

| Check | Location | Failure mode |
|---|---|---|
| `candidates.is_empty()` | Engine top of `induce_exact` | Returns default `ExactInductionResult` (all zeros) — matches Python reference's `if not body_indices: return …` |
| Buffer arity == 2 | `validate_pair_buffer` | `XlogError::Execution` |
| Column type is exactly one of `U64`, `U32`, `Symbol` | `validate_pair_buffer` | `XlogError::Type` |
| Positive, negative, and candidate pair types match exactly | `require_pair_type` / provider layout check | `XlogError::Type` / `XlogError::Kernel` |
| `cached_row_count()` populated | `cached_rows` helper | `XlogError::Execution` — caller fed a buffer that skipped the DLPack ingest path |
| `positive_count == 0` | `classify_request` (pure) | Returns result with zero candidates but retained neg/candidate counts |

Trivial dead-end classification (`classify_request`) is pure and
unit-tested without CUDA via 5 tests in `xlog-induce/src/validate.rs`.

## Public Types

```rust
// crates/xlog-induce/src/lib.rs
pub struct InduceExactRequest<'a> {
    pub head_rel_idx: RelId,
    pub candidates: &'a [(RelId, &'a CudaBuffer)],
    pub positives: &'a CudaBuffer,
    pub negatives: Option<&'a CudaBuffer>,
    pub config: ExactInductionConfig,
}

pub fn induce_exact(
    provider: &CudaKernelProvider,
    request: &InduceExactRequest<'_>,
) -> Result<ExactInductionResult>;
```

```rust
// crates/xlog-induce/src/types.rs
pub enum Topology { Chain, Star, Fanout, Fanin }

pub struct ExactInductionConfig {
    pub k_per_topology: u32,
    pub deterministic: bool,  // reserved — engine is inherently deterministic
}

pub struct ScoredCandidate {
    pub topology: Topology,
    pub head_rel_idx: RelId,
    pub left_rel_idx: RelId,
    pub right_rel_idx: RelId,
    pub positives_covered: u32,
    pub negatives_covered: u32,
    pub local_rank: u32,
    pub next_positives_covered: u32,
    pub next_negatives_covered: u32,
    pub tie_class_size: u32,
}

pub struct ExactInductionResult {
    pub candidates: Vec<ScoredCandidate>,
    pub total_scored: u32,
    pub candidate_count: u32,
    pub positive_count: u32,
    pub negative_count: u32,
}
```

## Generated Rule Provenance

Generated-rule provenance adds audit records for generated/mined rules in
`crates/xlog-induce/src/provenance.rs`. The scorer still returns
`ExactInductionResult`; provenance is the companion layer for callers that
promote a candidate into a generated rule and need to retain why it was selected.

```rust
pub struct InducedRuleProvenance {
    pub rule_id: String,
    pub rule_source: String,
    pub source_kind: RuleSourceKind,
    pub search_space_size: u64,
    pub predicate_inventory: Vec<String>,
    pub support_rows: Vec<InductionSupportRow>,
    pub rejected_alternatives: Vec<InductionAlternative>,
    pub falsification_count: u64,
    pub generation_trace_hash: String,
}
```

`InductionSupportRow` records retained positive support by relation, row index,
and caller-supplied row hash. `InductionAlternative` records rejected candidate
rules with support and falsification counts. `InductionProvenanceRegistry`
stores the generated-rule records in memory for the caller's audit path.

## Python Surface

```python
from pyxlog.ilp import induce_exact

result = induce_exact(
    prog,                                   # CompiledIlpProgram
    head_relation="p_A",
    candidate_relations=["p_B", "p_C", "p_D"],
    positive_arg0=pos_a0,                   # 1-D device torch tensor
    positive_arg1=pos_a1,
    negative_arg0=neg_a0,                   # optional
    negative_arg1=neg_a1,
    k_per_topology=2,
    deterministic=True,
    backend="native",                       # or "python" (prototype)
    # strict_per_topology=True,             # only meaningful for backend="python"
)
```

Returns an `ExactInductionResult` dataclass with `candidates: list[ScoredCandidate]`.

## Testing

These are manual validation commands, subject to the repository's execution and
resource policy. Their presence here is not a claim that this checkout has been
executed or certified.

- **Host-only unit tests** (`cargo test -p xlog-induce --lib`):
  request classification, topology ordering and reduction reference behavior.
  The pure helpers need no CUDA device.
- **CUDA-gated launcher tests** (`cargo test -p xlog-cuda --lib ilp_exact`):
  `ilp_exact_score_matches_hand_computed_fixture` (hand-derived coverage
  against C=2 candidates), `ilp_exact_score_is_deterministic_across_runs`,
  `ilp_exact_score_handles_empty_negatives`,
  `ilp_exact_score_accepts_u32_pair_buffers`,
  `ilp_exact_score_accepts_symbol_pair_buffers`, and
  `ilp_exact_score_rejects_mixed_pair_types`. Skip with `eprintln!` when no
  CUDA device is present.
- **Python parity test** (`python -m pytest python/tests/test_ilp_exact_induce.py`):
  `test_induce_exact_native_matches_python_reference` and
  the native-vs-Python host-transfer budget parity test.
  Uses `strict_per_topology=True` against the Python reference so both
  backends compute clean per-topology coverage.
- **Typed provider tests**:
  `U32` and `Symbol` fixtures exercise the typed scorer and reject mixed logical
  pair types. The Python parity test separately compares complete native results
  with the strict reference.
  The provider's diagnostic `ilp_exact_score` exports two full count arrays;
  production `ilp_exact_score_topk` instead exports one compact selected-row
  allocation. These are different transfer boundaries.

## Type Dispatch And Packaging Policy

- `U64`: supported through `ilp_exact_score`.
- `U32`: supported through `ilp_exact_score_u32`.
- `Symbol`: supported through the same physical `u32` kernel while retaining
  `Symbol` schema identity and rejecting mixed `U32`/`Symbol` requests.
- Chain topology shared-memory caching: supported through
  `ilp_exact_score_chain_smem` and `ilp_exact_score_chain_smem_u32` when
  `XLOG_ILP_EXACT_CHAIN_SMEM` is enabled and the candidate-row threshold is
  met. Non-chain topologies inside those entry points continue to call the
  baseline matcher.
- PTX policy: `crates/xlog-cuda/kernels/ilp_exact.cu` is checked in; generated
  `ilp_exact.portable.ptx` and architecture-specific `.cubin` files are build
  and packaging artifacts, not source artifacts. `crates/xlog-cuda/build.rs`
  generates portable PTX for every manifest module, including `ilp_exact`;
  `scripts/stage_pyxlog_kernels.sh` stages those artifacts into
  `pyxlog/kernels/`; `scripts/install_pyxlog_for_python.py` rejects a wheel
  that lacks portable PTX. This matches the current ILP-family convention for
  `ilp.cu` and `ilp_credit.cu`.

## Profile-Gated Chain Shared Memory

The profile-gated chain shared-memory scorer adds an A/B-controlled chain scorer that tiles
left-relation rows into dynamic shared memory for the strict chain topology.
It is enabled by default only for candidate relations with at least
`XLOG_ILP_EXACT_CHAIN_SMEM_MIN_ROWS` rows, defaulting to `256`, and can be
disabled with `XLOG_ILP_EXACT_CHAIN_SMEM=0`.

The public scoring and transfer contract is unchanged. The profile runner
records parity and these performance criteria:

- identical coverage signatures for shared-memory and baseline dispatch;
- at least `1.2x` median speedup on the chain-hot fixture;
- at most five percent regression on the small fixture; and
- the same single selected-row export for production top-K induction.

The measurement scripts record these criteria as observations. Source inspection
or historical profiling does not prove a performance result for the current
artifact and GPU.

## See Also

- [CUDA certification](https://xlog.md/architecture/certification) — manual
  CUDA qualification and its evidence boundaries. Generated PTX is a packaging
  artifact; not committing it does not determine certification coverage.
- [Living-world diagnostics](https://xlog.md/guides/diagnostics) — generated-rule provenance
  and diagnostics surfaces for generated rules, source rules, proof traces,
  deltas, temporal metadata, and neural hot loops.
- [Roadmap](../../../ROADMAP.md) — bounded exact induction and related learning work.
