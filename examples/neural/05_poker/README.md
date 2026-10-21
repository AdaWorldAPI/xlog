# 05_poker

Real playing-card recognition example.

The runner directly supervises card rank and suit. `eval_joint_proxy` is their
mean classification accuracy, not joint rank-and-suit accuracy or poker-hand
evaluation. Run from this directory with a CUDA build of `pyxlog`.

Expected layout:
- `data/cards/**/*.jpg` (or `.png`)
- Filenames should encode rank+suit, e.g. `AS.jpg`, `10H.png`, `QC.jpg`

Run:
- `python train.py --mode ci`
- `python train.py --mode dev`
- `python train.py --mode release`
