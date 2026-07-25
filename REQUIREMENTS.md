# Requirements

## Software

- **Python** 3.10 (3.9+ works; the ACS experiment was validated on 3.9).
- **Python packages** (`requirements.txt`; versions are lower bounds known to work):

  | Package | Version | Used by |
  |---|---|---|
  | numpy | ≥ 1.24 | all |
  | torch | ≥ 2.0 | all (training, Bernstein layers) |
  | scikit-learn | ≥ 1.2 | data splits, StandardScaler, CART fallback |
  | pandas | ≥ 1.5 | KD result tables |
  | scipy | ≥ 1.10 | dependency of the above |
  | matplotlib | ≥ 3.6 | neuron/regime figures |
  | openml | ≥ 0.13 | Adult dataset fetch (Adult experiments only) |
  | folktables | ≥ 0.0.12 | ACS Income fetch (`ACS/` only) |
  | auto_LiRPA | GitHub `main` | MAGIC ReLU certified column only (TABLE X, regenerated); git-install, see INSTALL.md |

  No GPU-specific build is required; a CPU-only PyTorch wheel is sufficient.
  `auto_LiRPA` is **not** in `requirements.txt` (its PyPI release would downgrade torch);
  it is optional and only needed to recompute MAGIC TABLE X's ReLU column from scratch.

- **OS**: Linux or macOS. Validated on Linux (x86-64, kernel 5.14).

## Hardware

- **CPU is sufficient** for every result in this artifact. A CUDA GPU is
  optional and only speeds up from-scratch (re)training.
- **Memory**: 8 GB RAM is comfortable.
- **Disk**:
  - Repo + shipped checkpoints/results: well under 100 MB.
  - The ACS experiment downloads a folktables PUMS cache to
    `ACS/data/` on first run — **~523 MB** (git-ignored). Reproducing
    the shipped TABLE XI from checkpoints needs this data only to recompute the
    network baselines; point `--data-dir` at an existing cache to skip the
    download.

## Network

- First run of the Adult experiments fetches from OpenML; first run of the ACS
  experiment fetches ACS PUMS via folktables. Both cache locally and need
  network access only once. TABLE XI's renderer (`make_table_xi.py`) needs no
  network — it runs on the committed CSVs.

## Runtime (rough, CPU)

| Task | Time |
|---|---|
| `make_table_xi.py` (render TABLE XI from committed CSVs) | seconds |
| `run_multiseed.py` from shipped checkpoints (exact TABLE XI) | a few minutes (+ one-time data download) |
| `run_multiseed.py` full retrain (5 seeds) | ~1–2 hours on CPU (faster on GPU) |
| MAGIC `make_table5.py` / `make_table_x.py` (render from shipped CSVs) | seconds |
| MAGIC `relu_lirpa_certify.py` (recompute TABLE X from shipped weights) | ~1–2 minutes (needs auto_LiRPA) |
| MAGIC `extract_rules.py` (one student) | ~1–2 minutes (first run builds the candidate cache) |
| MAGIC `run_kd_experiments.py` full retrain (3 configs × 5 seeds) | ~30–60 minutes on CPU |

MAGIC uses no network (its `magic04.data` is shipped) and no GPU.
