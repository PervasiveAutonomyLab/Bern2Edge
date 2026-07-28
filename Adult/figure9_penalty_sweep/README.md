# Figure 9 — Adult penalty sweeps

This directory reproduces the two penalty-sweep panels in Figure 9 from the
committed rule and CART artifacts. The plotted values are means across five
Adult Bernstein-network architectures, not results for `h=32` alone:

- `14x16x2`
- `14x32x2`
- `14x128x2`
- `14x16x8x2`
- `14x32x16x2`

Each penalty ranges from 0.0 to 1.0 in steps of 0.1. Only the configurations
used by the figure are shipped:

- sweep `same_cov_alpha` with `conflict_alpha=0.5`;
- sweep `conflict_alpha` with `same_cov_alpha=0.5`.

Their intersection is stored once, giving
`5 architectures × (11 + 11 - 1) = 105` configurations. Every configuration
has one exact rule JSON and one exact CART `_tree.npz` sidecar under
`artifacts/`.

## Exact reproduction from ready artifacts

From the repository root, after following `INSTALL.md`:

```bash
python Adult/figure9_penalty_sweep/reproduce_figure9.py
```

The script uses the reusable `Adult/evaluate_rule_artifacts.py` evaluator to:

1. load the fixed Adult fold from `Adult/adult_teacher_ordinal.pt`;
2. apply each committed oblique rule to the held-out test set;
3. traverse the committed CART for uncovered samples;
4. recompute rule count, coverage, covered accuracy, total accuracy,
   conflicts, and mean conditions;
5. writes `figure9_metrics_by_arch.csv`;
6. averages the two slices across the five architectures;
7. verifies all 44 plotted coordinates against `figure9_values.csv`;
8. writes both panels as PDF and PNG.

The general evaluator can also be run directly on this artifact directory:

```bash
python Adult/evaluate_rule_artifacts.py \
  Adult/figure9_penalty_sweep/artifacts \
  --output Adult/figure9_penalty_sweep/standalone_metrics.csv
```

The JSON serializer rounds rule weights to six decimals. On a small number of
boundary samples, reapplying those rounded weights changes coverage by 0.01
percentage point. The per-architecture CSV therefore retains both the live
`reevaluated_test_covered_pct` and the generation-time `test_covered_pct`
embedded in each exact JSON. The latter is used for the published mean after
the live value has been checked within 0.02 point. Covered accuracy is
recomputed directly and matches every published coordinate exactly.

For a network-free plot-only check:

```bash
python Adult/figure9_penalty_sweep/reproduce_figure9.py --plot-only
```

This redraws the panels directly from the committed averaged values.

## Extraction from scratch

The ready-artifact evaluator above does not regenerate rules. To repeat rule
extraction from the five shipped student checkpoints, use the shared Adult
driver with the Figure 9 protocol:

```bash
python Adult/run_rule_extraction.py \
  --arch 14x16x2 14x32x2 14x128x2 14x16x8x2 14x32x16x2 \
  --fallback tree --sparsity-k 7 \
  --same-cov 0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 \
  --conflict 0.5 \
  --out-csv Adult/figure9_penalty_sweep/fresh_same_cov.csv \
  --out-json-dir Adult/figure9_penalty_sweep/fresh_same_cov_artifacts

python Adult/run_rule_extraction.py \
  --arch 14x16x2 14x32x2 14x128x2 14x16x8x2 14x32x16x2 \
  --fallback tree --sparsity-k 7 --same-cov 0.5 \
  --conflict 0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 \
  --out-csv Adult/figure9_penalty_sweep/fresh_conflict.csv \
  --out-json-dir Adult/figure9_penalty_sweep/fresh_conflict_artifacts
```

The distinct output paths prevent the shared `(0.5,0.5)` configuration from
being appended twice. Fresh extraction can vary if library versions or the
dataset source change; the committed artifacts are the authoritative
exact-reproduction path.
