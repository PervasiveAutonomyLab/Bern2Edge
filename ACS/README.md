# Distribution-shift robustness on ACS Income (paper TABLE XI)

Does the geometry-derived symbolic rule set keep its **coverage** and **purity**
off-distribution? This experiment trains the full Bern2Edge stack in-distribution
on **ACS Income, California 2018**, then evaluates the networks and the extracted
rules under **geographic** shift ({MS, WY, WV}, same year) and **temporal** shift
(California {2019, 2021, 2022}). Uncovered inputs are handled by a **CART**
fallback (depth 4).

It reuses Bern2Edge's shared modules unchanged — `models.AdultTeacherMLP` /
`bern2edge.models.FCModel`, `bern2edge.bernstein.BernsteinLayer`,
`bern2edge.kdtrain.kd_train_models`, and the `bern2edge.rule_extraction` package —
through a thin compatibility shim ([_compat.py](_compat.py)); ACS data loading lives
in the repo's [bern2edge/data.py](../bern2edge/data.py) (`bern2edge.data.acs_income`).
No Bern2Edge shared module is modified.

## Reproduce TABLE XI

```bash
# From the repo root, with the environment installed (see ../INSTALL.md).
python ACS/run_multiseed.py       # -> results/metrics_multiseed_*.csv
python ACS/make_table_xi.py       # -> results/table_xi.tex + RESULTS_table_xi.md
```

`run_multiseed.py` loads the committed per-seed checkpoints
(`results/_multiseed_cache/seed*/models.pt` = teacher + Bernstein/ReLU students)
and the per-combo cache (`combo_sc0.1_cf0.2.json`), recomputes the network
baselines, re-extracts the rules from the Bernstein student, applies the
deterministic CART fallback, and writes `metrics_multiseed_raw.csv`. Because the
stochastic pieces (networks) are frozen and the fallback is deterministic, this
reproduces TABLE XI **exactly**. `make_table_xi.py` then renders it.

The first run fetches ACS PUMS via folktables (needs network once); pass
`--data-dir /path/to/existing/acs/cache` to reuse a cache. To retrain everything
from scratch, delete `results/_multiseed_cache/` first — the network rows then
land within the reported seed std (the reused Bernstein init/warmup differ from
the original by design; the shared modules are not modified), while the rules'
CART numbers stay deterministic given the student.

### How TABLE XI is generated

```
models.pt (.pth)  ──►  Bernstein student weights  ──►  network-purity rules
   │                                                          │
   ├─► teacher / student accuracies (ReLU Teacher, ReLU, BNN) │
   │                                                          ▼
   │                                          CART fallback on the uncovered residual
   ▼                                                          │
metrics_multiseed_raw.csv  ◄──────────────────────────────────┘
   │
   └─►  make_table_xi.py  ──►  table_xi.tex  (TABLE XI)
```

Rules are re-extracted from the student's first-layer geometry each run
(deterministic), so nothing is read from stale rule dumps.

## What TABLE XI is

TABLE XI selects, from the per-condition results:

| TABLE XI element | Source |
|---|---|
| **ReLU Teacher / ReLU / BNN** (Acc %) | ReLU FC teacher / same-size ReLU student / Bernstein student |
| **Rules** (Coverage %, Covered acc %, Total acc %) | α_sc = 0.1, α_conf = 0.2; **Total acc = CART fallback** |
| **Geographic** columns | MS, WY, WV; **GEO-AVG** = mean over the three per seed |
| **Temporal** columns | 2019, 2021, 2022; **TEMP-AVG** = mean over the three (2020 PUMS not released — COVID-19) |
| **Δ** | AVG − ID (pp) |
| ± | sample std (ddof=1) over 5 seeds [42, 1, 2, 3, 4] |

> **Fidelity note.** Every mean and per-condition value matches the published PDF
> exactly. `make_table_xi.py` computes GEO/TEMP-AVG, their std, and Δ
> self-consistently over the {MS, WY, WV} / {2019, 2021, 2022} conditions. Two
> purely cosmetic differences vs the typeset PDF remain: (i) the PDF's GEO-AVG
> *std* cell (e.g. 0.64 for the teacher) was carried over from an earlier
> aggregation that also averaged the SD/PR states, whereas this experiment reports
> the subset std (0.57); (ii) two Δ cells differ by 0.1 pp from last-digit
> rounding. The underlying per-seed accuracies are identical.

## Layout

```
_compat.py            reuse shim over Bern2Edge (make_loader, MLP alias, ad/M facades, kd adapter)
train_teacher.py      ReLU FC teacher on CA-2018 (fit_relu_mlp reused by run_multiseed)
run_multiseed.py      TABLE XI driver: 5 seeds -> teacher/students/rules/CART -> metrics
make_table_xi.py      render TABLE XI ({MS,WY,WV}+{2019,21,22}, CART, Δ) -> results/table_xi.tex
results/              CSVs, RESULTS_multiseed.md, table_xi.tex, and
                      _multiseed_cache/seed*/ (models.pt + combo_sc0.1_cf0.2.json)
data/                 folktables ACS PUMS cache (git-ignored; fetched at runtime)
```

## Configuration (fixed hyperparameters)

| Component | Values |
|---|---|
| Teacher MLP | `[512,256,128]`, dropout 0.1, AdamW lr 1e-3, wd 1e-4, batch 256, early-stop patience 16, acc-gate 0.80 |
| KD students | arch `[10,32,2]`, bern degree 3, KD α 0.5, T 2, lr 3e-3, wd 1e-4, 100 epochs |
| Rule extraction | grid 5, max_depth 3, purity cascade `[(1.0,2),(0.95,3),(0.90,5)]`, dense; α_sc 0.1, α_conf 0.2 |
| Fallback | CART `DecisionTreeClassifier(max_depth=4, random_state=seed)` on the uncovered residual |
| Seeds | [42, 1, 2, 3, 4] |

Features (10): AGEP, COW, SCHL, MAR, OCCP, POBP, RELP, WKHP, SEX, RAC1P —
categoricals kept as raw integer codes, all columns standardized (scaler fit on
CA-2018 train). `RELSHIPP→RELP` semantic recode aligns the relationship feature
across the 2019+ temporal splits.
