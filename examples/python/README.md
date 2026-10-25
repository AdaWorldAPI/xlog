# Python Examples (`pyxlog`)

These scripts demonstrate the `pyxlog` Python module (built from
`crates/pyxlog`) using **DLPack** for GPU table interchange. One-shot inputs and
the consumer handoff of exported tensor views are zero-copy. Persistent relation
replacements take an owned device-to-device snapshot, and exporting a stored
relation makes a device-to-device retention copy before handing off its buffers.

## Install a wheel

From the repository root, use the helper that stages CUDA kernels, builds for
the selected Python interpreter, and installs the resulting wheel. GPU artifact
builds are manual LOCAL operations under the repository execution policy.

```bash
python scripts/install_pyxlog_for_python.py --python /usr/local/bin/python
```

## Run on an authorized CUDA execution host

```bash
/usr/local/bin/python examples/python/01_dlpack_reachability_torch.py
/usr/local/bin/python examples/python/02_prob_wet_conditioning_torch.py
/usr/local/bin/python examples/python/03_prob_mc_nonmonotone_torch.py
```

These scripts use PyTorch. The module also accepts CUDA-backed DLPack producers
such as cuDF, CuPy, and GPU-backed JAX arrays. Supply one 1-D producer object per relation
column rather than manufacturing raw capsules. Use the same interpreter for
installation and execution. A CUDA output confirms its location; transfer
counters for the measured interval are needed to establish ZERO TRANSFER.
