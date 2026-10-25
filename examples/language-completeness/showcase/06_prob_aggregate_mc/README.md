# Probabilistic Aggregate MC

Configures the source for MC probabilistic inference while keeping the fixture
small enough for explain-time aggregate diagnostics.

Production resident MC rejects aggregate heads. To obtain a sampled CPU reference
result, select that oracle explicitly:

```bash
xlog prob examples/language-completeness/showcase/06_prob_aggregate_mc/program.xlog --allow-cpu-oracle --output json
```

The result is labeled `mc_engine: "cpu-oracle"`. Explain-time aggregate metadata
does not show that this program ran on a GPU. Use the adjacent exact aggregate
example for the supported finite production aggregate path.
