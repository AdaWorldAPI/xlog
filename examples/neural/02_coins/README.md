# 02_coins

Real coin dataset (heads/tails).

The runner trains `coin(image, label)` directly and reports held-out classifier
`test_acc`; it does not train from derived win/loss labels. Run from this directory
with a CUDA build of `pyxlog` and PyTorch installed.

Expected layout:
- `data/coins/train/heads`
- `data/coins/train/tails`
- `data/coins/test/heads`
- `data/coins/test/tails`

Run:
- `python train.py --mode ci`
- `python train.py --mode dev`
- `python train.py --mode release`
