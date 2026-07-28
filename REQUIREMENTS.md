# Requirements

## Software

- Linux or macOS; validated on Linux x86-64.
- Python 3.10 recommended; Python 3.9 or newer is supported.
- Python dependencies listed in `requirements.txt` and `pyproject.toml`.
- Dependency versions in `requirements.txt` are tested lower bounds, not a
  byte-for-byte environment lock. Record the resolved environment with
  `python -m pip freeze > environment.txt` when evaluating or archiving a run.
- Optional `auto_LiRPA` for recomputing the ReLU certification column in
  MAGIC Table X. See `INSTALL.md`.

A CPU-only PyTorch installation is sufficient to render tables and recompute
results from the shipped checkpoints. From-scratch transformer training requires
a CUDA GPU.

## Hardware

- 8 GB system RAM is sufficient for the CPU workflows.
- A CUDA GPU is optional except for `Transformer/run_variant.sh`.
- Transformer training uses about 11 GB of GPU memory; it was validated on an
  NVIDIA A30 with 24 GB.
No VM or container image is currently supplied. The documented Python
environment uses commodity software; the optional `auto_LiRPA` installation is
the only non-standard dependency and is needed only for a live recomputation of
one Table X column.

## Storage

Allow approximately:

- 350 MB for the repository and shipped checkpoints;
- 523 MB for the ACS PUMS download;
- 75 MB for the cached Covertype dataset;
- 100 MB for cached SST-2 and TinyBERT files.

Adult and HIGGS-Small downloads also use the OpenML cache.

## Network access

The following experiments download data on first use and then use a local cache:

| Experiment | Download source |
|---|---|
| Adult and HIGGS-Small | OpenML |
| Covertype | scikit-learn |
| ACS Income | US Census data through `folktables` |
| Transformer | SST-2 and TinyBERT from Hugging Face |

MAGIC does not download experiment data because
`MAGIC/magic04.data` is included in the repository. Its renderers, training,
rule extraction, and certification can therefore run without dataset network
access after the Python environment is installed. Installing dependencies,
including optional `auto_LiRPA`, may still require internet access.

All table renderers operate on committed result files and require no network.

## Approximate runtimes

| Task | Runtime |
|---|---|
| Table renderers | Seconds |
| Table I checkpoint evaluation | A few minutes |
| Table II checkpoint evaluation | 2–5 minutes |
| ACS Table XI checkpoint evaluation | A few minutes |
| MAGIC Table X live certification | 1–2 minutes |
| Transformer Table XII evaluation | About 1 minute on GPU; a few minutes on CPU |
| Transformer smoke training | About 7 minutes on a GPU |
| Transformer full training, one variant | About 3 hours on an A30 |
