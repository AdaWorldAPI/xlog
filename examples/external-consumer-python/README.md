# External Consumer Example Suite

These examples exercise the public Python APIs for graph queries, relation updates,
neural training and rule induction. Each directory contains a small `.xlog` program and a
self-checking `run.py` that emits one JSON result.

From the repository root, run the full suite on an authorized CUDA execution host:

```bash
python scripts/validate_external_consumer_examples.py --python /usr/local/bin/python \
  --output /tmp/xlog-external-consumer-summary.json
```

Each example must report `status="PASS"`; the validator combines those reports in
the selected JSON file. The examples require CUDA and an installed `pyxlog` wheel
with packaged kernels in the selected interpreter. For a source build, use
`scripts/install_pyxlog_for_python.py --python /usr/local/bin/python` under the
manual LOCAL build policy before transferring the artifacts to the execution host.

| Example | Purpose | Surfaces |
|---|---|---|
| `01_async_streaming_reachability` | Recursive external consumer-style graph query | `LogicProgram`, sessions, `evaluate_async`, `evaluate_stream`, chunked result tensors, memory/progress/graph/host-transfer stats |
| `02_relation_deltas` | Relation maintenance analog | `put_relation`, `insert_relation`, `delete_relation`, `apply_relation_delta`, `delta_stats`, full-replacement equivalence |
| `03_neural_bridge_topk_belnap` | Neural bridge fixture | `.xlog` neural predicate, `register_network(k=..., det=True)`, `forward_backward_tensor`, deterministic top-k, cache stats, Belnap helper |
| `04_native_exact_induction` | external consumer-shaped exact ILP topology selection | learnable `.xlog`, `IlpProgramFactory`, `induce_exact(..., backend="native")`, Python/native parity |
| `05_probabilistic_async_diagnostics` | Probabilistic API diagnostics | probabilistic `.xlog`, `Program.evaluate_async`, exact and MC mode, memory/progress/graph/host-transfer stats |

CUDA tensor checks establish output location. The validator's
`zero_copy_diagnostics` field contains those location checks; it does not measure
allocation sharing or establish ZERO TRANSFER. Review the separate transfer
diagnostics for the operations actually measured.
