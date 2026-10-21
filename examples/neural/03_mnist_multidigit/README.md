# 03_mnist_multidigit

Real SVHN two-digit recognition example.

Expected layout:
- `data/svhn/train/digitStruct.mat`
- `data/svhn/train/*.png`

The loader filters SVHN to 2-digit samples and trains `digit_left/2` and `digit_right/2`.
Both digits are supervised directly. The reported `eval_joint_proxy` is the
arithmetic mean of their separate accuracies, not the fraction of examples with
both digits correct. Run from this directory with a CUDA build of `pyxlog`.

Run:
- `python train.py --mode ci`
- `python train.py --mode dev`
- `python train.py --mode release`
