# Bern2Edge

Knowledge-distillation of Bernstein-activation student networks and their
extraction into interpretable, hardware-friendly symbolic rules, on the Adult,
Cover Type, HIGGS, MAGIC, and ACS Income datasets.

> **Artifact evaluation.** See [INSTALL.md](INSTALL.md) (install + smoke test),
> [REQUIREMENTS.md](REQUIREMENTS.md), [STATUS.md](STATUS.md) (badges), and
> [LICENSE](LICENSE).

The repository has two parts:

1. **KD compression** — train compact Bernstein/ReLU students from a teacher
   (`bern2edge/kdtrain.py`, `bern2edge/models.py`, `bern2edge/data.py`, and each
   dataset's `run_kd_experiments.py`).
2. **Rule extraction** — turn a trained Bernstein student into a small set of
   `band_lo <= w · x < band_hi` rules with a fallback for uncovered inputs, and
   quantize them for hardware (`bern2edge/rule_extraction/`, `Adult/run_rule_extraction.py`).

## Install

```bash
pip install -e .                  # installs the deps and puts bern2edge on the path
```

Or `pip install -r requirements.txt` — the drivers add the repo root to `sys.path`
themselves, so `bern2edge` imports work without installing, as long as you run the
commands below from the repository root. See [INSTALL.md](INSTALL.md).

The Adult loader fetches the dataset from OpenML on first use (needs network
access); the fixed train/dev/test split is read from `Adult/adult_teacher_ordinal.pt`.

## Rule extraction (Adult)

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

### Fixed hyperparameters (paper setting)

These are the defaults in `bern2edge.rule_extraction.ExtractionConfig`:

| Hyperparameter        | Value                         |
|-----------------------|-------------------------------|
| Regime grid           | `N_FIXED_GRID = 5`            |
| Max conditions / rule | `max_depth = 3`              |
| Purity cascade        | `(1.0, 2), (0.95, 3), (0.90, 5)` |
| Conflict penalty      | `conflict_alpha = 0.1`       |
| Same-coverage penalty | `same_cov_alpha ∈ {0.5, 0.1}`|
| Fallback              | CART (`max_depth = 4`)       |
| Sparsification        | dense (`sparsity_k = None`)  |

`rule_results.csv` columns → table: `n_rules` (Rules), `avg_conditions` (ℓ),
`test_covered_pct` (Cov), `test_covered_rule_acc` (Cov.Acc), `test_rule_acc` (Acc_t).

### Penalty sweep (joint α_conf × α_sc, one architecture)

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

## Distribution-shift robustness (ACS Income, TABLE XI)

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

## MAGIC Gamma Telescope (TABLE V rule extraction, TABLE X certified robustness)

`MAGIC/` distils Bernstein students on the MAGIC Gamma Telescope dataset, extracts
`band_lo <= w · x < band_hi` slab rules (**TABLE V**), and certifies local robustness of
the rule system, the Bernstein network, and a matched ReLU network under per-feature
input noise (**TABLE X**). It reuses the shared `bern2edge.data` (MAGIC loader) and
`bern2edge.kdtrain`;
the model core, rule extractor, and certifier are local to the folder.

```bash
# Reproduce both tables from the shipped metrics (seconds, no training, no auto_LiRPA):
python MAGIC/make_table5.py      # TABLE V  (from MAGIC/5_fold_results.csv)
python MAGIC/make_table_x.py     # TABLE X  (from MAGIC/results/table_x.csv)

# Recompute TABLE X live from the shipped weights (needs auto_LiRPA for the ReLU column):
RF=rule_jsons/magic_rules_kd_fc_10x64x32x2_bern_deg3_alpha0.5_T2_lr0.003_wd0.0001_seed42_g5_p85_mc5_d2_adaptive.json
python MAGIC/relu_lirpa_certify.py --arch 10x64x32x2 --rule-file MAGIC/$RF --write-csv MAGIC/results/table_x.csv
```

The **ReLU certified column uses `auto_LiRPA`** (git-install; see [INSTALL.md](INSTALL.md)) —
it is needed **only** to regenerate that column, not to render the shipped tables. Full
train/extract-from-scratch instructions are in [MAGIC/README.md](MAGIC/README.md).

## Neuron / regime visualization

```bash
python -m bern2edge.rule_extraction.visualize_neurons \
    --ckpt Adult/rule_checkpoints/kd_fc_14x64x2_bern_deg3_alpha0.5_T2_lr0.006_wd0.0001_seed6.pth
```

Writes `bern_neuron_plots_<ckpt>/01_neuron_polynomials.pdf`: six shape-diverse
Bernstein neurons with their regime breakpoints (same `N_FIXED_GRID` as the
extractor — the grid constant lives in one place,
`bern2edge/rule_extraction/bern_regimes.py`).

## Layout

```
bern2edge/                # the shared library — everything below is imported as bern2edge.*
  bernstein.py            # Bernstein activation layer
  models.py               # teacher / student networks (FCModel, *TeacherMLP)
  data.py                 # dataset loaders (Adult, Cover Type, HIGGS, MAGIC, ACS)
  kdtrain.py              # knowledge-distillation training loops
  train_utils.py          # plain train / eval helpers
  rule_extraction/        # rule-extraction package (dataset-agnostic)
    bern_regimes.py       # activation geometry + N_FIXED_GRID
    extraction.py         # candidates, greedy cover, fallbacks, metrics, JSON
    quantize.py           # int8 / fix<16,8> rule quantization
    visualize_neurons.py  # neuron/regime figure
Adult/  cover_type/  higgs_small/                # per-dataset drivers, weights, results
Adult/run_rule_extraction.py, Adult/make_table3.py, Adult/rule_checkpoints/
ACS/           # ACS Income distribution-shift experiment (TABLE XI)
  _compat.py              # reuse shim over the shared modules
  train_teacher.py, run_multiseed.py, make_table_xi.py
  results/                # shipped models.pt + combo cache + CSVs (results/table_xi.tex)
MAGIC/                    # MAGIC Gamma Telescope (TABLE V rule extraction, TABLE X certification)
  run_kd_experiments.py, extract_rules.py, relu_lirpa_certify.py
  bern_net.py, rule_extraction_magic.py, rule_certify.py   # local model core / extractor / certifier
  make_table5.py, make_table_x.py                          # render TABLE V / TABLE X
  student_model_weights/, rule_jsons/, 5_fold_results.csv, results/table_x.csv   # shipped artifacts
pyproject.toml  requirements.txt                 # dependencies / optional `pip install -e .`
CITATION.cff  INSTALL.md  REQUIREMENTS.md  STATUS.md  LICENSE   # artifact-evaluation docs
```
