# Table IV — LUT BNNs versus rule networks

This directory reproduces the Adult software provenance and renders the
LUT-based Bernstein-network versus rule-network hardware comparison on KV260.
It uses the five existing seed-6 Bernstein checkpoints and their matching
`same_cov_alpha=0.5`, `conflict_alpha=0.1` rule JSON/CART artifacts.

## Reproduce

From the repository root, after following `INSTALL.md`:

```bash
python Adult/table4_rule_network_hardware/reproduce_table4.py
```

The command evaluates each `.pth` and ready rule artifact, then writes:

- `table4_artifact_metrics.csv`: live accuracy and direct artifact paths;
- `table4_values.csv`: evaluated metrics joined with post-synthesis results;
- `table4.tex`: generated LaTeX table.

## Inputs

- LUT checkpoints:
  `Adult/rule_checkpoints/kd_fc_<arch>_bern_deg3_alpha0.5_T2_lr0.006_wd0.0001_seed6.pth`
- Rule artifacts:
  `Adult/rule_jsons/<checkpoint-stem>_sca0.5_ca0.1/rules_float.json`
  and the adjacent `fallback_tree_float.npz`
- `hardware_results.csv`: supplied post-synthesis accuracy, latency, DSP, BRAM,
  LUT, and FF values.

The script evaluates software artifacts but does not rerun HLS synthesis.
Consequently, the supplied CSV is authoritative for the displayed
post-synthesis accuracy and hardware columns. The final CSV retains both the
post-synthesis and live evaluated accuracies so any difference remains visible.

## Regenerate the rule-network hardware

Generate the five CART-fallback rule-classifier HLS projects with:

```bash
python Adult/table4_rule_network_hardware/generate_and_synthesize_table_iv.py \
  --generate-only
```

Remove `--generate-only` after loading the required Vitis environment to run
fresh synthesis and compare its latency and resource reports with the committed
Table IV rule rows. The LUT BNN side of Table IV uses the generic FC compiler
documented in `hls/README.md`.
