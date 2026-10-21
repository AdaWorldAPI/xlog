# Neural-Symbolic Examples

Neural-symbolic training examples demonstrating integration where neural network outputs become probabilistic facts in logic programs.

## Overview

These examples exercise XLOG's neural-symbolic training stack:

- **Neural Predicates**: `nn(network, [inputs], output, [labels]) :: predicate(args).`
- **Network Registration**: PyTorch networks with optimizers and schedulers
- **Tensor Sources**: External data (images, embeddings) indexed by predicates
- **Differentiable Training**: Gradients flow from logic queries through d-DNNF circuits to neural networks
- **Circuit Caching**: Compiled circuits reused across training iterations; the paper's separate MNIST cache ablation measures a 2.74x gain under its stated protocol

## Examples

### 01_minimal — MNIST Addition

Train a CNN to classify MNIST digits using **only addition supervision**.

```
examples/neural/01_minimal/
├── train.py       # Training script
└── data/          # MNIST data (auto-downloaded)
```

**The key insight**: The network never sees digit labels during training. Instead, it learns from queries like "What is the probability that image[i] + image[j] = 7?"

**Run:**
```bash
cd examples/neural/01_minimal
python train.py --engine xlog --epochs 20 --batch-size 64 --train-limit 20000 --eval-limit 10000 --seed 0
```

The `xlog` arm trains through `pyxlog.train_model_tensor` on addition queries.
The `torch` arm is a direct PyTorch addition reference and bypasses the XLOG
compiler, circuit, and gradient bridge. Both use addition supervision and separate
MNIST training and test splits.

**Historical PyTorch-reference benchmark (2026-02-09):**

| Metric | Value |
|--------|-------|
| Command | `python train.py --engine torch --epochs 20 --batch-size 512 --pairs-per-epoch 65536 --train-limit 60000 --eval-limit 10000 --seed 0` |
| Device | `cuda` (`NVIDIA RTX PRO 3000 Blackwell Generation Laptop GPU`) |
| Final train accuracy | `0.9983` |
| Final held-out eval accuracy (`MNIST test`, `n=10000`) | `0.9907` |
| Initial loss | `1.3407` |
| Final loss | `0.0138` |

These numbers belong to the recorded `--engine torch` command, not to the XLOG
arm above. They do not measure XLOG circuit performance or establish a current
runtime result. Training used train-split images and addition supervision;
evaluation used the disjoint MNIST test split.

**Program:**
```prolog
nn(mnist_net, [X], Y, [0,1,2,3,4,5,6,7,8,9]) :: digit(X, Y).
addition(X, Y, Z) :-
    digit(X, LeftDigit),
    digit(Y, RightDigit),
    Z is LeftDigit + RightDigit.
```

**Training:**
```python
# Query: What's P(image[0] + image[1] = 7)?
queries = ["addition(0, 1, 7)", "addition(2, 3, 5)", ...]

# NLL loss trains the network to maximize P(correct_sum)
history = pyxlog.train_model(program, queries, epochs=50, batch_size=32)
```

## Required Example Set

The required neural-symbolic example set is now present in the repository:

| Example | Description |
|---------|-------------|
| **02_coins** | Direct `coin` label supervision; held-out classifier `test_acc` |
| **03_mnist_multidigit** | Direct left/right digit supervision on two-digit SVHN; `eval_joint_proxy` is the mean of the two digit accuracies |
| **04_hwf** | Direct four-way CROHME expression-type classification (`add`, `sub`, `mul`, `eq`); classifier `eval_acc` |
| **05_poker** | Direct rank/suit supervision; `eval_joint_proxy` is the mean of rank and suit accuracy |
| **06_clutrr** | Direct four-class relation supervision over story text; classifier `eval_acc` |

These five entrypoints exercise network registration and direct neural-head
training. Their metrics do not establish compound-query reasoning, complete
formula recognition, poker-hand evaluation, or rule induction from CLUTRR graphs.
The mean-accuracy proxies are not joint exact-match accuracy.

Each `train.py --mode release` run now emits a `FINAL_METRIC` line and enforces
the example-specific minimum accuracy from `examples/neural/<example>/dataset.json`
unless overridden via `--min-accuracy`.

## How Neural-Symbolic Training Works

1. **Define neural predicates** — declare that a network provides probability distributions
2. **Forward pass** — network outputs become weights in annotated disjunctions
3. **Knowledge compilation** — logic program compiled to d-DNNF circuit (cached!)
4. **Weighted model counting** — circuit evaluated for query probability
5. **Backward pass** — gradients flow from NLL loss through circuit to networks
6. **Optimizer step** — PyTorch updates network weights

## Requirements

- Python 3.8+
- PyTorch 2.0+
- torchvision (for MNIST)
- Linux x86_64, NVIDIA GPU, CUDA Toolkit 13.x for the XLOG execution path
- An installed CUDA build of `pyxlog`; follow the repository's [manual build procedure](../../CONTRIBUTING.md)

The repository validator uses `XLOG_NEURAL_FIXTURE_SMOKE=1` to execute every
neural example entrypoint with deterministic in-memory fixtures when optional
external datasets or torchvision are not installed. That fixture mode disables
minimum-accuracy enforcement even when `--min-accuracy` is supplied, and cannot
establish real-dataset acceptance. Keep it unset for benchmark measurements.

## API Reference

```python
import pyxlog

# Compile program with neural predicates
program = pyxlog.Program.compile("""
    nn(net_name, [X], Y, [0,1,...,9]) :: digit(X, Y).
    addition(X, Y, Z) :-
        digit(X, LeftDigit),
        digit(Y, RightDigit),
        Z is LeftDigit + RightDigit.
""")

# Register PyTorch network
program.register_network("net_name", model, optimizer, scheduler=None)

# Add tensor data source
program.add_tensor_source("train", images_tensor)

# Single query training
loss = program.forward_backward("addition(0, 1, 7)")  # host scalar (convenience)

# CUDA tensor loss; this API alone does not certify the whole loop's transfer boundary
loss_t = program.forward_backward_tensor("addition(0, 1, 7)")

# Batch training
stats = program.train_epoch(queries, batch_size=32)
avg_loss = stats.avg_loss

# Full training loop
history = pyxlog.train_model(
    program,
    queries,
    epochs=50,
    batch_size=32,
    log_iter=10
)

# Access training history
print(f"Final loss: {history.epoch_losses[-1]}")
```

## References

- [Whitepaper](../../paper/main.pdf) — architecture, language framing, and neural-symbolic design context
- [Python Bindings](https://xlog.md/reference/python) — `register_network`, `register_embedding`, training-loop APIs, and result surfaces
- [dILP Training](https://xlog.md/neural/rule-learning) — differentiable ILP architecture and trainer contract
