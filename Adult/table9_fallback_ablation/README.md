# Table IX — Adult fallback ablation

This directory reproduces Table IX for the Adult `14x32x2` rule model with
`k=7`, `same_cov_alpha=0.5`, and `conflict_alpha=0.1`.

The `artifacts/` directory contains the exact float and int8 rule JSONs and the
ready fallback artifact for each reported variant. The four variants use the
same 50 rules and therefore the same held-out uncovered subset.

## Reproduce the table

From the repository root, after following `INSTALL.md`:

```bash
python Adult/table9_fallback_ablation/reproduce_table9.py
```

The command:

1. loads the fixed Adult held-out test split;
2. applies the committed rule JSONs;
3. evaluates each fallback only where no rule matches;
4. recomputes fallback accuracy and fidelity on that uncovered subset;
5. writes detailed software metrics to `table9_artifact_metrics.csv`;
6. copies full HLS accuracy and fallback-only resources from
   `hardware_results.csv`;
7. writes the direct table data to `table9_values.csv`;
8. renders `table9.tex`.

`accuracy_pct` in the final table is the full rules-plus-fallback HLS accuracy.
`fallback_acc_uncovered_pct` compares fallback predictions with Adult ground
truth only on uncovered held-out test samples.
`fallback_fidelity_uncovered_pct` compares those predictions with the full
Bernstein network on the same samples.

The Small BNN row is the int8 `small_nn` implementation. The full-precision
small-network control is outside the scope of this artifact.

## Files

- `artifacts/lr/`: rule JSONs; fallback parameters are embedded.
- `artifacts/network/`: rule JSONs and the full Bernstein checkpoint.
- `artifacts/small_nn/`: rule JSONs and the `14x4x2` Bernstein checkpoint.
- `artifacts/tree/`: rule JSONs and float/int CART sidecars.
- `hardware_results.csv`: synthesis summary for the four reported variants.
- `table9_artifact_metrics.csv`: live ready-artifact evaluation.
- `table9_values.csv`: direct inputs to the rendered table.
- `table9.tex`: generated LaTeX table.

The synthesis project and per-sample HLS traces are not included; the script
does not rerun synthesis. It treats the committed summary as the authoritative
source for HLS accuracy and resource columns.

## Regenerate the HLS projects

The rule HLS driver generates both the complete classifier and a fallback-only
project for each of LR, full BNN, int8 Small BNN, and CART:

```bash
python Adult/table9_fallback_ablation/generate_and_synthesize_table_ix.py \
  --generate-only
```

Remove `--generate-only` after loading the required Vitis environment to run
fresh synthesis and compare its resource and latency reports with
`hardware_results.csv`:

```bash
source <Vitis>/2024.1/settings64.sh
python Adult/table9_fallback_ablation/generate_and_synthesize_table_ix.py
```
