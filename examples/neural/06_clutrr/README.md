# 06_clutrr

Four-class relation classification over CLUTRR-style story text. The runner
trains a GRU-backed `rel(story, label)` predicate directly and reports classifier
`eval_acc`; it does not reconstruct family graphs or induce logical rules.
Run from this directory with a CUDA build of `pyxlog`.

Expected layout:
- `data/clutrr/train.jsonl`

Each JSONL row must include:
- `story` (or `text`)
- `target_relation` (or `label`) in `{parent, child, spouse, sibling}`

Run:
- `python train.py --mode ci`
- `python train.py --mode dev`
- `python train.py --mode release`
