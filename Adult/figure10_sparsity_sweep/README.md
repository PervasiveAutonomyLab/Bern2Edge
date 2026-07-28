# Figure 10 — Adult sparsity sweep

This directory reproduces Figure 10 for the Adult Bernstein network with
architecture `14x32x2` (`h=32`). It sweeps the rule sparsity threshold
`k=1,...,13` at fixed `same_cov_alpha=0.1` and `conflict_alpha=0.2`.

The exact 13 self-contained rule JSONs used for the published curve are under
`artifacts/`. The plotted total accuracy uses the highest-purity matching rule
when multiple rules fire.

## Exact reproduction

From the repository root, after following `INSTALL.md`:

```bash
python Adult/figure10_sparsity_sweep/reproduce_figure10.py
```

The script:

1. loads the fixed Adult test split;
2. evaluates every committed rule artifact with the shared
   `Adult/evaluate_rule_artifacts.py` evaluator;
3. writes `figure10_artifact_metrics.csv`;
4. joins the committed BRAM measurements from `bram_measurements.csv`;
5. verifies all 26 plotted coordinates against `figure10_values.csv`;
6. writes `figure10_sparsity_sweep.pdf` and
   `figure10_sparsity_sweep.png`.

For a seconds-long redraw that does not load the dataset:

```bash
python Adult/figure10_sparsity_sweep/reproduce_figure10.py --plot-only
```

The direct plot values are in `figure10_values.csv`. BRAM values are committed
measurements; the script does not rerun FPGA synthesis.

## Evaluate the artifacts directly

The reusable evaluator can also process the directory independently:

```bash
python Adult/evaluate_rule_artifacts.py \
  Adult/figure10_sparsity_sweep/artifacts \
  --conflict-strategy max_purity \
  --no-stored-validation \
  --output Adult/figure10_sparsity_sweep/standalone_metrics.csv
```

`--no-stored-validation` is needed because each JSON records the companion
coverage-ordered evaluation, while Figure 10 uses maximum-purity conflict
resolution. Coverage, covered accuracy, conflict count, and rule complexity do
not depend on that choice.

## Extraction from scratch

The shared `Adult/run_rule_extraction.py` driver supports `--sparsity-k`,
`--same-cov`, and `--conflict` for fresh extraction runs. Use a separate output
CSV and artifact directory for each `k=1,...,13`. The committed JSONs remain
the authoritative exact-reproduction inputs; fresh extraction may vary if
library versions or the dataset source change.
