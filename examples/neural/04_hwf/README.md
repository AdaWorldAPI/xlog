# 04_hwf

CROHME expression-type classification example. The loader rasterizes InkML and
assigns one of four labels: `eq` if LaTeX contains `=`, otherwise `add` for `+`,
`sub` for `-`, and `mul` otherwise. The runner directly supervises `expr_type`
and reports classifier `eval_acc`; it does not recognize or evaluate a complete
formula. Run from this directory with a CUDA build of `pyxlog`.

Expected layout:
- `data/crohme/**/*.inkml`
- `data/crohme/**/*.txt` (same stem as InkML, contains LaTeX)

Run:
- `python train.py --mode ci`
- `python train.py --mode dev`
- `python train.py --mode release`
