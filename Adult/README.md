# Adult Census experiments

This directory contains the Adult components used by Tables I, III, VIII, IX,
Figure 9, and part of the fallback analysis in Table IX. Run commands from the
repository root after following `INSTALL.md`.

## Data and provenance

The loader obtains the Adult dataset from OpenML on first use. The committed
`adult_teacher_ordinal.pt` stores the teacher checkpoint and fixed split/
preprocessing metadata. Network and rule checkpoints are research artifacts;
they contain model parameters and preprocessing state, not personal identifiers.
The underlying dataset remains subject to its source terms.

## Quick check: render committed rule results

```bash
python Adult/make_table3.py
python Adult/make_table8.py
```

These commands read `rule_results.csv` and
`penalty_sweep_14x32x2_results.csv`, print Markdown, and write `table3.tex` and
`table8.tex`. They do not train models or extract new rules.

## Evaluate ready rule artifacts

Use `evaluate_rule_artifacts.py` to evaluate existing Adult rule artifacts
without extracting or training anything:

```bash
# One JSON:
python Adult/evaluate_rule_artifacts.py path/to/rules.json

# Every JSON artifact in a directory:
python Adult/evaluate_rule_artifacts.py path/to/artifacts/ --output metrics.csv
```

It recognizes both `<json_stem>_tree.npz` research sidecars and the current
extractor's `fallback_tree_float.npz`, as well as self-contained artifacts.
Each output row reports architecture and penalties, rule count, coverage,
covered accuracy, total accuracy, conflict count, average conditions, and
artifact paths. Use `--conflict-strategy max_purity` when the highest-purity
matching rule should win; see `--help` and the module docstring.

## Regenerate Table III

```bash
python Adult/run_rule_extraction.py
python Adult/make_table3.py
```

The extraction driver loads the shipped Bernstein checkpoints, regenerates
float and quantized rule JSON files, evaluates coverage and accuracy, and writes
`rule_results.csv`. Use `--arch` to select one architecture and `--fallback
tree` for the paper's CART fallback.

## Reproduce Table IV

Table IV compares the five Adult LUT-based Bernstein networks with their
`same_cov_alpha=0.5`, `conflict_alpha=0.1` rule deployments:

```bash
python Adult/table4_rule_network_hardware/reproduce_table4.py
```

The script evaluates the existing `.pth` checkpoints and ready rule/CART
artifacts, joins the supplied post-synthesis accuracy and KV260 hardware
measurements, and generates detailed and direct-value CSVs plus LaTeX. See
`table4_rule_network_hardware/README.md` for the exact path mapping.

## Reproduce Table VII

```bash
python Adult/table7_xc7s15_deployment/reproduce_table7.py
python Adult/table7_xc7s15_deployment/generate_and_synthesize_table_vii.py --generate-only
```

This is the XC7S15 deployment experiment containing six Bernstein widths and
the R50/R29 rule classifiers. See `table7_xc7s15_deployment/README.md`.

## Regenerate Table VIII

```bash
python Adult/run_penalty_sweep.py
python Adult/make_table8.py
```

The driver evaluates the configured `(conflict_alpha, same_cov_alpha)` grid and
writes `penalty_sweep_14x32x2_results.csv` plus rule/fallback sidecars. The
`penalty_sweep_14x32x2_jsons_original/` directory preserves the original paper
outputs; newly generated outputs go to
`penalty_sweep_14x32x2_jsons/`.

## Reproduce Figure 9

Figure 9 reports the two one-dimensional `k=7`, CART-fallback penalty slices
averaged across five architectures. Its exact 105 rule JSON/CART configurations
are isolated under `figure9_penalty_sweep/`:

```bash
# Re-evaluate every committed rule/CART pair, rebuild the averages, verify all
# published coordinates, and redraw both panels:
python Adult/figure9_penalty_sweep/reproduce_figure9.py

# Network-free redraw from the committed averaged CSV:
python Adult/figure9_penalty_sweep/reproduce_figure9.py --plot-only
```

See `figure9_penalty_sweep/README.md` for the artifact inventory, metric
definitions, exact extraction-from-scratch commands, and output files.

## Reproduce Figure 10

Figure 10 sweeps sparsity `k=1,...,13` for the Adult `14x32x2` network at fixed
penalties. Its exact 13 rule artifacts, direct-value CSV, BRAM measurements,
and renderer are isolated under `figure10_sparsity_sweep/`:

```bash
python Adult/figure10_sparsity_sweep/reproduce_figure10.py
python Adult/figure10_sparsity_sweep/reproduce_figure10.py --plot-only
```

The full command re-evaluates all artifacts and verifies every plotted
coordinate. See `figure10_sparsity_sweep/README.md` for the inventory and
metric definitions.

## Reproduce Table IX

Table IX compares four fallback strategies for the same `14x32x2`, `k=7` rule
set. The bundle contains the exact rule/fallback artifacts and the original
synthesis summary:

```bash
python Adult/table9_fallback_ablation/reproduce_table9.py
```

The evaluator recomputes fallback accuracy and fidelity only on uncovered
held-out samples, joins the committed HLS accuracy and fallback-only resource
columns, and writes both CSV and LaTeX outputs. See
`table9_fallback_ablation/README.md` for metric definitions and provenance.

## Outputs

- `kd_compression_results.csv`: Adult student-compression metrics used by the
  Table I aggregation.
- `rule_results.csv`: Table III rule count, complexity, coverage, covered
  accuracy, and total accuracy.
- `table4_rule_network_hardware/`: Table IV checkpoint/rule evaluation,
  supplied HLS measurements, direct-value CSV, and LaTeX table.
- `penalty_sweep_14x32x2_results.csv`: Table VIII joint penalty results.
- `rule_jsons/` and `penalty_sweep_14x32x2_jsons/`: generated float/int8 rules
  and fallback sidecars.
- `figure9_penalty_sweep/`: exact Figure 9 artifacts, averaged plot CSV,
  evaluator, and plots.
- `figure10_sparsity_sweep/`: exact Figure 10 artifacts, direct-value CSV,
  BRAM measurements, evaluator output, and plots.
- `table9_fallback_ablation/`: exact Table IX rule/fallback artifacts, synthesis
  summary, evaluated metrics, direct-value CSV, and LaTeX table.

Regeneration may overwrite derived CSV/JSON/LaTeX outputs. Copy them elsewhere
first if you need to preserve a local run.
