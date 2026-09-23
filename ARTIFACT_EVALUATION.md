# Artifact evaluation guide — v1.0.0 archive

> **Historical evaluation document.** This file is retained to document the
> review workflow used for the archived Bern2Edge v1.0.0 CODES 2026 artifact.
> Regular users should start with `README.md`, `INSTALL.md`, and `RESULTS.md`.

The immutable v1.0.0 release is available from:

- GitHub: https://github.com/PervasiveAutonomyLab/Bern2Edge/releases/tag/v1.0.0
- Zenodo: https://doi.org/10.5281/zenodo.21726441

The current `main` branch may contain changes made after the archived release.
Use the tagged release or Zenodo archive when reproducing the original artifact
evaluation environment.

Run the commands below from the repository root. Installation details are in
`INSTALL.md`; hardware, software, storage, and network constraints are in
`REQUIREMENTS.md`.

## 1. Record the revision and environment

```bash
git rev-parse HEAD
git status --short
python --version
python -m pip freeze > environment.txt
```

The v1.0.0 release bundle was prepared with the paper, source code, model
checkpoints, rule artifacts, reproduction scripts, HLS source-generation
workflows, and artifact-evaluation documentation.

## 2. Install

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

## 3. Smoke test: seconds, no dataset download

```bash
python Adult/make_table3.py
python Adult/make_table8.py
python MAGIC/make_table5.py
python MAGIC/make_table_x.py
python ACS/make_table_xi.py
python Transformer/make_table_xii.py
```

Pass condition: every command exits with status 0 and prints its named table.
These commands render committed result files; they do not retrain models.

## 4. Live checkpoint test: 2–5 minutes

```bash
python cover_type/reproduce_table_ii.py
```

The first run downloads Covertype. The run passes when it ends with:

```text
Verified 50 checkpoints and all 10 five-fold means.
Wrote cover_type/table_ii_checkpoint_results.csv
Wrote cover_type/table_ii_results.csv
```

This recomputes software accuracy from all 50 Table II checkpoints.

## 5. Select deeper checks

Use `RESULTS.md` to choose a paper result and its command.

Long or optional workflows include:

| Workflow | Requirement | Typical runtime |
|---|---|---:|
| Table I checkpoint evaluation | CPU; Adult/HIGGS downloads | minutes |
| Table I HLS source generation | CPU; no Vitis required | seconds |
| Table I fresh csim/csynth | Vitis HLS 2024.1; KV260 target | tool-dependent |
| Table II fresh csim/csynth | Vitis HLS 2024.1; KV260 target | tool-dependent |
| Table IV rule HLS source generation | CPU; no Vitis required | seconds |
| Table IV fresh rule csynth | Vitis HLS 2024.1; KV260 target | tool-dependent |
| Table IX full/fallback HLS source generation | CPU; no Vitis required | seconds |
| Table IX fresh csynth | Vitis HLS 2024.1; KV260 target | tool-dependent |
| Table XII Transformer HLS source generation | CPU; no Vitis required | seconds |
| Table XII fresh csynth | Vitis HLS 2024.1; KV260 target | tool-dependent |
| Table VII XC7S15 source generation | CPU; no Vitis required | seconds |
| Table VII fresh csynth/post-route | Vitis HLS and Vivado 2024.1; XC7S15 | tool-dependent |
| Table X live certification | `auto_LiRPA` | 1–2 minutes |
| Table XI live evaluation | ACS download (~523 MB) | minutes |
| Table XII live evaluation | SST-2/TinyBERT download | minutes |
| Transformer training | CUDA GPU (~11 GB) | ~3 hours per variant |

## 6. Original artifact-evaluation context

The CODES 2026 artifact call defined three independent candidate badges:
**Available**, **Reviewed**, and **Reproducible**. The original v1.0.0
documentation was structured to provide evidence for those evaluation criteria.

This retained guide documents that process; it does not assert which badges
were ultimately awarded.

Official CODES 2026 artifact information:
https://esweek.org/call-for-artifacts-codessisss/
