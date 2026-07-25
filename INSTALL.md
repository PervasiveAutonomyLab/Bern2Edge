# Install

## 1. Environment

```bash
# Python 3.10 recommended. A virtualenv/conda env is advised.
python -m venv .venv && source .venv/bin/activate     # or: conda create -n bern2edge python=3.10
pip install -e .
```

`pip install -e .` installs the dependencies **and** puts the shared `bern2edge`
library on your import path. If you prefer not to install anything, plain
`pip install -r requirements.txt` also works: every driver adds the repository root
to `sys.path` itself, so `bern2edge` resolves as long as you run the commands below
**from the repository root**.

A CPU-only PyTorch is sufficient for all results. See [REQUIREMENTS.md](REQUIREMENTS.md)
for versions and hardware.

### Optional: `auto_LiRPA` (only for the MAGIC ReLU certified column, TABLE X)

Needed only to **regenerate** the ReLU column of MAGIC TABLE X from scratch
(`MAGIC/relu_lirpa_certify.py`). Rendering the shipped MAGIC tables does not need it.
Install from GitHub — the PyPI release pins `torch<1.13` and would downgrade your torch:

```bash
pip install --no-deps git+https://github.com/Verified-Intelligence/auto_LiRPA.git
pip install graphviz
```

## 2. Smoke test (no network, no training — ~seconds)

Render the paper's **TABLE XI** from the committed multi-seed results:

```bash
python ACS/make_table_xi.py
```

Expected: a Markdown table printed to stdout (and written to
`ACS/results/table_xi.tex` + `RESULTS_table_xi.md`) whose first rows read

```
| ReLU Teacher | Acc. (%)         | 81.28±0.09 | 74.32±0.57 | -7.0 | 73.73 | 74.61 | 74.61 | 80.00±0.09 | -1.3 | ...
| BNN          | Acc. (%)         | 80.89±0.29 | 74.45±0.67 | -6.4 | 73.79 | 74.77 | 74.79 | 79.81±0.36 | -1.1 | ...
| Rules        | Total acc. (%)   | 78.58±0.13 | 73.15±1.32 | -5.4 | 73.41 | 72.44 | 73.61 | 77.24±0.29 | -1.3 | ...
```

If that renders, the install is good. (These match the published TABLE XI; see
`ACS/README.md` for the note on the two cosmetic std/Δ differences.)

## 3. Functional test (reproduces TABLE XI from shipped checkpoints)

```bash
# Fetches ACS PUMS via folktables on first run (needs network once).
python ACS/run_multiseed.py
python ACS/make_table_xi.py
```

This rewrites `ACS/results/metrics_multiseed_*.csv` +
`RESULTS_multiseed.md` from the committed per-seed checkpoints and renders TABLE XI.
It is byte-identical to the committed CSVs. To skip the download, pass
`--data-dir /path/to/existing/acs/cache`.

## 4. MAGIC tables (no network, no training, no auto_LiRPA — ~seconds)

```bash
python MAGIC/make_table5.py      # TABLE V  (rule extraction)
python MAGIC/make_table_x.py     # TABLE X  (certified robustness)
```

Both read shipped metrics CSVs and print the published numbers. To recompute TABLE X
live from the shipped weights (needs `auto_LiRPA`, see step 1), and for the
train/extract-from-scratch pipeline, see [MAGIC/README.md](MAGIC/README.md).

## 5. Other experiments (Adult / Cover Type / HIGGS)

See the main [README.md](README.md) — e.g. the Adult rule table:

```bash
python Adult/run_rule_extraction.py && python Adult/make_table3.py
```
The Adult loader fetches from OpenML on first use.
