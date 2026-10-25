# Runtime Consumer Examples

These ordinary `.xlog` programs exercise queries used by external applications.
The validator runs them through `xlog-cli` and combines their results with
recorded feature measurements and fresh public Python compatibility checks.

The suite covers:

- an external consumer-shaped delta and optimizer fixture;
- two neutral scientific/engineering workloads derived from external-consumer
  flow and signal reasoning, without project-specific terms in the programs;
- a runtime-substrate fixture that documents the exact/index/CSE/adaptive
  primitives required by later epistemic and solver work;
- public compatibility by invoking the existing validators.

The combined validator additionally requires the retained historical feature
measurement bundle under `docs-internal/evidence/`. That local-only archive is
not tracked or present in a clean checkout; without it the combined validator
raises an error and cannot complete. Do not create replacement measurements or
present old files as a fresh runtime qualification. Individual example READMEs
describe their direct CLI commands without this archive dependency.

When that bundle is available, run the validator from the repository root on an
authorized CUDA execution host with an installed `pyxlog` wheel and an existing CLI binary:

```bash
python scripts/validate_runtime_consumers.py --python /usr/local/bin/python \
  --xlog-bin /path/to/xlog --output /tmp/xlog-runtime-consumer-summary.json
```

The summary contains per-example checks, raw command output, transfer diagnostics
and compatibility results. Feature measurements are loaded from
`docs-internal/evidence/*/measurements.json`; each of the six measured features must have
exactly one matching record. Those retained measurements keep their original
source and runtime scope. A passing summary does not refresh historical feature
measurements or prove current full-path ZERO TRANSFER.
