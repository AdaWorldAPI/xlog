# Probabilistic Examples (`xlog-prob`)

These `.xlog` files exercise the probabilistic-programming surface:

- probabilistic facts (`p::atom.`)
- annotated disjunctions (`p1::a1; p2::a2.`)
- evidence (`evidence(atom, true|false).`)
- probabilistic queries (`query(atom).`)
- positive recursion with explicit Monte Carlo selection (`#pragma prob_engine = mc`)

Run the exact examples with the CLI, or select the resident Monte Carlo engine for
the positive recursive example:

```bash
xlog prob examples/prob/01-wet-conditioning.xlog
xlog prob examples/prob/04-recursive-mc.xlog --samples 10000 --seed 42
```

Negation and aggregates are rejected by production resident Monte Carlo. The
explicit `--allow-cpu-oracle` option permits a labeled CPU reference result for
those shapes; it is not evidence of GPU execution. Source pragmas select defaults,
and explicit CLI flags override the matching values.

Python bindings (`crates/pyxlog`) and Rust integrations use the same engine APIs:

- `xlog_prob::exact::ExactDdnnfProgram` (exact)
- `xlog_prob::mc::McProgram` (approximate, with resident fragment checks)
