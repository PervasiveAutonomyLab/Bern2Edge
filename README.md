# Bern2Edge

Knowledge-distillation of Bernstein-activation student networks and their
extraction into interpretable, hardware-friendly symbolic rules, on the Adult,
Cover Type, HIGGS, MAGIC, ACS Income, and SST-2 datasets.

Installation, system requirements, and result coverage are documented in
[INSTALL.md](INSTALL.md), [REQUIREMENTS.md](REQUIREMENTS.md), and
[RESULTS.md](RESULTS.md). Artifact evaluators can use the separate
[evaluation guide](ARTIFACT_EVALUATION.md). The paper is included as
[Bern2Edge.pdf](Bern2Edge.pdf).

## Quick start

Run all commands from the repository root.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
python Adult/make_table3.py
python MAGIC/make_table5.py
python Transformer/make_table_xii.py
```

These checks take seconds, need no dataset download, and do not train models.
For a live checkpoint evaluation, run
`python cover_type/reproduce_table_ii.py` (2–5 minutes after the one-time
dataset download). See [RESULTS.md](RESULTS.md) for the command associated with
each paper result.

## Overview

The repository separates reusable code from experiments:

1. **`bern2edge/`** — the importable Python package containing shared models,
   data loaders, training utilities, Bernstein activations, and rule extraction.
2. **Experiment directories** — training drivers, checkpoints, and paper
   results. Table I aggregation is in `table_i_compression/`; student training
   remains in `Adult/`, `cover_type/`, and `higgs_small/`.

### Pipeline overview

A summary of the complete teacher-to-edge workflow.

[![Bern2Edge pipeline](figures/fig1.png)](figures/fig1.pdf)

*Overview of Bern2Edge: A high-accuracy teacher model is distilled into a
compressed BNN student via KD. The resulting representation is synthesized and
deployed via either exact LUT-based realization or symbolic rule extraction.*

## Student BNN compression and synthesis

### Compression and KV260 synthesis (Table I)

The Table I reproducer rebuilds all 18 rows for Adult, Covertype, and
HIGGS-Small. It evaluates the 90 shipped student checkpoints (18 models × five
folds), recomputes accuracy and cross-entropy from the weights, verifies each
accuracy against its checkpoint metadata, and joins the results to the paper's
KV260 synthesis measurements:

```bash
python table_i_compression/reproduce_table_i.py
python table_i_compression/reproduce_table_i.py --device cpu
```

It writes:

- `table_i_compression/table_i_checkpoint_results.csv` — per-fold checkpoint path, live accuracy,
  train CE, test CE, and the stored-vs-live accuracy check;
- `table_i_compression/table_i_results.csv` — the complete 18-row Table I result, including
  five-fold means/standard deviations and Bernstein-minus-ReLU deltas;
- `table_i_compression/table_i_hls_results.csv` — committed HLS latency, DSP, BRAM, and LUT values
  copied from Table I of the paper.

The paper's `CE Loss` convention is the mean training-set hard-label CE, so that
is `ce_loss_mean` in the final table; live held-out CE is retained as
`test_ce_mean` for auditability. The weights are authoritative: small differences
between a regenerated value and the rounded PDF are kept rather than replaced
with paper values. Dataset downloads are cached after the first run.

### Covertype under matched hardware budgets (Table II)

`cover_type/` reproduces the Bernstein-vs-ReLU accuracy comparison under five
matched latency and BRAM budgets. The artifact ships the exact 50 student
checkpoints used by the table: ten model configurations × five folds
(`seed=1000` through `seed=1004`).

```bash
# Re-evaluate all 50 checkpoints and regenerate the accuracy/HLS table:
python cover_type/reproduce_table_ii.py
```

The first run downloads scikit-learn's Covertype dataset. The script reconstructs
the fixed stratified split (`seed=42`) and per-fold preprocessing, evaluates each
checkpoint on its original held-out test fold, verifies all ten five-fold means,
and writes:

- `cover_type/table_ii_checkpoint_results.csv` — every fold accuracy and its
  project-relative `.pth` path;
- `cover_type/table_ii_results.csv` — the five paper rows with full-precision
  accuracies, deltas, HLS latency/BRAM metrics, and checkpoint provenance.

The synthesis export is committed as
`cover_type/covertype_hls_results.csv`. HLS metrics are extracted from this CSV
and matched by complete architecture, activation, and Bernstein degree. A
successful run ends with
`Verified 50 checkpoints and all 10 five-fold means.` See
[cover_type/README.md](cover_type/README.md) for the exact model mapping,
acceptance tolerances, and the documented fourth-row standard-deviation display
difference in the paper.

## Symbolic rule extraction

### Rule-extraction overview

An overview of the activation regimes, rule formation, and resulting rule
regions.

<a href="figures/fig4.pdf">
  <img src="figures/fig4.png" alt="Bernstein activation regimes" width="55%">
</a>

*Bernstein activation curves with analytically derived regime breakpoints. Six
representative neuron activations from a BNN (h = 64) trained on Adult, with
breakpoints from derivative roots and inflection points marked.*

[![Activation-geometry rule formation](figures/fig5.png)](figures/fig5.pdf)

*Activation-geometry-based rule formation. A selected regime in normalized
activation space (t) maps through z-space to an affine constraint in input
space, forming interpretable oblique regions.*

<a href="figures/fig6.pdf">
  <img src="figures/fig6.png" alt="Activation-geometry rule regions" width="55%">
</a>

*Activation-geometry rule regions on the Two Moons dataset. Rules align with
the curved decision boundary via Bernstein-derived breakpoints, enabling compact
partitioning.*

### Adult rule extraction (Table III)

```bash
# Reproduce the rule-extraction table (5 architectures x same-cov penalties):
python Adult/run_rule_extraction.py            # -> Adult/rule_results.csv, Adult/rule_jsons/
python Adult/make_table3.py                    # render the table (Markdown + LaTeX)

# One architecture / a different fallback:
python Adult/run_rule_extraction.py --arch 14x16x2 --fallback tree
```

Each run writes, per config, `rules_float.json` and `rules_int8.json` (weights
quantized to per-vector int8, thresholds to fix<16,8>) plus any fallback
sidecars, and appends a metrics row to `Adult/rule_results.csv`.

Existing Adult rule JSON/CART pairs can be evaluated independently of
extraction:

```bash
python Adult/evaluate_rule_artifacts.py path/to/rules.json
python Adult/evaluate_rule_artifacts.py path/to/artifact_directory --output metrics.csv
```

`rule_results.csv` columns → table: `n_rules` (Rules), `avg_conditions` (ℓ),
`test_covered_pct` (Cov), `test_covered_rule_acc` (Cov.Acc), `test_rule_acc` (Acc_t).

See [Adult/README.md](Adult/README.md) for input provenance, exact outputs,
runtime expectations, and the distinction between rendering committed results
and regenerating rules.

### LUT BNN versus rule-network hardware (Table IV)

The Table IV bundle evaluates the five committed Adult Bernstein checkpoints
and their matching `same_cov_alpha=0.5`, `conflict_alpha=0.1` rule/CART
artifacts, joins the supplied KV260 measurements, and renders the table:

```bash
python Adult/table4_rule_network_hardware/reproduce_table4.py
```

The detailed CSV records the direct `.pth`, rule JSON, and fallback paths for
every row. See
[Adult/table4_rule_network_hardware/README.md](Adult/table4_rule_network_hardware/README.md)
for metric provenance and evaluation scope.

### MAGIC comparison with prior rule extractors (Table V)

```bash
python MAGIC/make_table5.py
```

This renders the committed five-fold results. Full training and extraction
instructions are in [MAGIC/README.md](MAGIC/README.md).

## End-to-end results

### Post-synthesis cross-dataset results (Table VI)

`end_to_end_results/` collects the selected HIGGS-Small, Covertype, and Adult
teacher/student checkpoints plus the Adult `14x16x8x2`,
`same_cov_alpha=0.5` rule result. It evaluates every software artifact, joins
the supplied post-synthesis measurements, calculates resource reductions
relative to each W8A8 teacher, and renders Table VI:

```bash
python end_to_end_results/reproduce_table_vi.py
```

See [end_to_end_results/README.md](end_to_end_results/README.md) for the exact
checkpoint/rule mapping and the distinction between evaluated software
accuracy and supplied post-synthesis accuracy.

## Hyperparameter and fallback ablation

### Joint α_conf × α_sc penalty sweep (Table VIII)

`Adult/run_penalty_sweep.py` sweeps the two greedy-cover penalties on a single
architecture (default `14x32x2`, dense, CART fallback) and writes one row per
`(conflict_alpha, same_cov_alpha)` combo:

```bash
python Adult/run_penalty_sweep.py         # -> Adult/penalty_sweep_14x32x2_results.csv + _jsons/
python Adult/make_table8.py               # render the sweep table (Markdown + LaTeX)
```

The sweep range is set by the clearly-labelled constants at the top of
`run_penalty_sweep.py` (`ARCH`, `CONFLICT_ALPHAS`, `SAME_COV_ALPHAS`), or via flags:

```bash
python Adult/run_penalty_sweep.py --arch 14x32x2 --conflict 0.1 1.0 --same-cov 0.1 0.3 0.5 1.0
```

`Adult/penalty_sweep_14x32x2_jsons_original/` holds the paper's original rule JSONs
for the same combos (copied verbatim, for reference); the tool-regenerated
`rules_float.json`/`rules_int8.json` land in `Adult/penalty_sweep_14x32x2_jsons/`.
Table columns → CSV: Conf=`n_conflicts`, Cov=`test_covered_pct`,
Cov.Acc=`test_covered_rule_acc`, Acc_t=`test_rule_acc`, Rules=`n_rules`, ℓ=`avg_conditions`.

### Penalty-sweep curves (Figure 9)

Figure 9 uses `k=7` rules with a CART fallback and averages the two plotted
penalty slices across five Adult architectures. The artifact bundle contains
only the 105 configurations used by the figure: 11 `alpha_sc` values at
`alpha_conf=0.5`, plus 11 `alpha_conf` values at `alpha_sc=0.5`, with the
intersection deduplicated, for each architecture.

```bash
# Exact path: evaluate the shipped rule JSON/CART pairs, regenerate the
# per-architecture and averaged CSVs, verify all 44 coordinates, and plot:
python Adult/figure9_penalty_sweep/reproduce_figure9.py

# Seconds-long, network-free plot regeneration from the committed averages:
python Adult/figure9_penalty_sweep/reproduce_figure9.py --plot-only
```

The committed direct plot data are in
`Adult/figure9_penalty_sweep/figure9_values.csv`. The figure is averaged across
`14x16x2`, `14x32x2`, `14x128x2`, `14x16x8x2`, and `14x32x16x2`; it is not an
`h=32`-only experiment. See the experiment README for provenance and the
from-scratch extraction commands.

### Sparsity and BRAM trade-off (Figure 10)

Figure 10 sweeps `k=1,...,13` for the Adult `14x32x2` network at
`alpha_sc=0.1` and `alpha_conf=0.2`. The bundle contains the exact 13 rule
artifacts, committed BRAM measurements, direct plot values, and a deterministic
renderer:

```bash
# Re-evaluate all artifacts, verify all 26 coordinates, and redraw the plot:
python Adult/figure10_sparsity_sweep/reproduce_figure10.py

# Seconds-long redraw directly from the committed plot-value CSV:
python Adult/figure10_sparsity_sweep/reproduce_figure10.py --plot-only
```

The direct values are in
`Adult/figure10_sparsity_sweep/figure10_values.csv`. See the experiment README
for the artifact inventory and metric definitions.

### Fallback strategy ablation (Table IX)

Table IX evaluates four fallback variants for one shared `14x32x2`, `k=7` rule
set. The exact rule/fallback artifacts and original synthesis summary are
included:

```bash
python Adult/table9_fallback_ablation/reproduce_table9.py
```

The command evaluates fallback accuracy and fidelity only on uncovered
held-out test samples, joins the committed full HLS accuracy and fallback-only
resource measurements, and writes detailed/direct CSVs plus `table9.tex`.
See `Adult/table9_fallback_ablation/README.md` for the artifact inventory and
metric definitions.

## Rule certification

### Certified robustness on MAGIC (Table X)

```bash
python MAGIC/make_table_x.py
```

This renders the committed Table X metrics. Recomputing the ReLU certificate
column requires the optional `auto_LiRPA` install described in
[INSTALL.md](INSTALL.md).

## Distribution shift

### ACS Income geographic and temporal shifts (Table XI)

`ACS/` trains the full stack in-distribution on ACS Income
**California 2018** and evaluates the networks and extracted rules under
**geographic** shift ({MS, WY, WV}) and **temporal** shift ({2019, 2021, 2022}),
with a **CART** fallback for uncovered inputs. It reuses the shared modules
unchanged via a compatibility shim (`ACS/_compat.py`); ACS data loading
lives in `bern2edge/data.py` (`bern2edge.data.acs_income`).

```bash
# Reproduce paper TABLE XI from the shipped per-seed checkpoints (exact):
python ACS/run_multiseed.py     # -> ACS/results/metrics_multiseed_*.csv
python ACS/make_table_xi.py     # -> ACS/results/table_xi.tex

# Seconds-long, network-free check: render TABLE XI from the committed CSVs:
python ACS/make_table_xi.py
```

`run_multiseed.py` fetches ACS PUMS via folktables on first run (pass `--data-dir`
to reuse a cache). TABLE XI = ReLU Teacher / ReLU / BNN accuracies + the Rules
block (Coverage / Covered acc / Total acc, Total acc = CART fallback), Δ = AVG−ID.
See `ACS/README.md` for details.

## Transformer FFN layers

### TinyBERT4 FFN substitution (Table XII)

`Transformer/` extends Bern2Edge from tabular MLPs to the **FFN sublayers of a
transformer**: all four TinyBERT4 encoder layers get their `312 → 1200 → 312` GeLU
FFN replaced by a narrow `312 → H → 312` FFN (H ∈ {312, 600}) with either a
Bernstein (LUT) activation or a matched-width GeLU control. Training is a
different technique from the tabular experiments — isolation function-matching,
then cold substitution, then KD fine-tuning — so it lives in its own folder with
its own weights and results.

```bash
# Render TABLE XII from the committed CSVs (seconds, no network, no GPU):
python Transformer/make_table_xii.py

# Recompute the accuracy row from the shipped weights, then re-render:
python Transformer/eval_release.py      # --device cpu also works
python Transformer/make_table_xii.py

# Classify one sentence with any variant:
python Transformer/load_and_run.py bern_h312 "this movie was a delight"
```

The **SST-2 accuracy row is recomputed from the weights**. Training from scratch
(`Transformer/run_variant.sh`) needs a GPU and reproduces the numbers
approximately, not exactly — see [Transformer/README.md](Transformer/README.md).

This is the only experiment that needs `transformers` and `datasets`, and the
only one that uses the shared modules' opt-in `BernsteinLayer(init="ramp")` and
`FCModel(act="gelu")` options.

## Layout

```
bern2edge/            shared Python package
table_i_compression/  Table I reproduction
end_to_end_results/   Table VI cross-dataset end-to-end results
Adult/                Adult rule extraction and ablation experiments
cover_type/           Covertype experiment (Table II)
higgs_small/          HIGGS-Small training and checkpoints
MAGIC/                MAGIC experiments (Tables V and X)
ACS/                  ACS Income experiment (Table XI)
Transformer/          TinyBERT4 experiment (Table XII)
figures/              figures used in this README
```

Each experiment directory contains its own README or is documented in the
relevant table section above.
