# ACS Income — Distribution-Shift Robustness (multi-seed, CART fallback)

Mean ± std (sample std) over **5 seeds** ([42, 1, 2, 3, 4]). Percentages. GEO-AVG = mean over the geo-shift states (MS, WY, WV); TEMP-AVG = mean over the temporal-shift years (2019, 2021, 2022); both computed per seed then aggregated. Rule systems use the **CART** fallback (depth 4) for uncovered inputs.

Preprocessing: raw-code categoricals + standardize-all; `RELSHIPP→RELP` semantic recode on the temporal splits (the 2020 ACS 1-Year PUMS was not released — COVID-19). Rule extraction: grid=5, depth=3, p90 cascade, dense.

## Network baselines (total accuracy %)

| System | ID: CA-2018 (trained on) | Geo: MS-2018 | Geo: WY-2018 | Geo: WV-2018 | Temporal: CA-2019 | Temporal: CA-2021 | Temporal: CA-2022 | GEO-AVG | TEMP-AVG |
|---|---|---|---|---|---|---|---|---|---|
| ReLU FC teacher | 81.28±0.09 | 73.73±0.66 | 74.61±0.57 | 74.61±0.69 | 80.74±0.09 | 79.82±0.13 | 79.44±0.11 | 74.32±0.56 | 80.00±0.09 |
| Bernstein student | 80.89±0.29 | 73.79±0.67 | 74.77±0.61 | 74.79±0.80 | 80.49±0.32 | 79.67±0.38 | 79.26±0.37 | 74.45±0.67 | 79.81±0.36 |
| ReLU student (same size) | 80.55±0.15 | 73.42±0.75 | 74.45±0.81 | 74.43±0.73 | 80.03±0.15 | 79.23±0.19 | 78.84±0.25 | 74.10±0.74 | 79.36±0.19 |

## Rules (α_sc=0.1, α_conf=0.2, CART fb)  (rules 80±12, train_cov 90.2±1.0%)

| Metric | ID: CA-2018 (trained on) | Geo: MS-2018 | Geo: WY-2018 | Geo: WV-2018 | Temporal: CA-2019 | Temporal: CA-2021 | Temporal: CA-2022 | GEO-AVG | TEMP-AVG |
|---|---|---|---|---|---|---|---|---|---|
| Coverage % | 90.09±0.86 | 86.10±0.96 | 85.07±1.68 | 86.29±1.51 | 90.01±0.89 | 90.35±0.77 | 90.35±0.71 | 85.82±1.36 | 90.24±0.78 |
| Covered acc % | 80.29±0.20 | 77.31±1.34 | 74.82±0.92 | 77.09±0.96 | 79.38±0.31 | 78.62±0.43 | 78.00±0.53 | 76.41±0.94 | 78.67±0.42 |
| Total acc % | 78.58±0.12 | 73.41±1.08 | 72.44±1.41 | 73.61±1.55 | 77.78±0.15 | 77.24±0.32 | 76.70±0.41 | 73.15±1.32 | 77.24±0.29 |

## Reproduce

Run from the repository root:

```bash
python ACS/run_multiseed.py --seeds 42 1 2 3 4 --combos 0.1,0.2
python ACS/make_table_xi.py     # render TABLE XI
```
