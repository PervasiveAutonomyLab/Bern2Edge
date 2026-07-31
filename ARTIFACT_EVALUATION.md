# Artifact evaluation guide

This is the shortest review path for the Bern2Edge artifact. Run commands from
the repository root. Installation details are in `INSTALL.md`; hardware,
software, storage, and network constraints are in `REQUIREMENTS.md`.

## 1. Confirm the submitted revision

Record:

```bash
git rev-parse HEAD
git status --short
python --version
python -m pip freeze > environment.txt
```

The submitted archive must contain `Bern2Edge.pdf`, `LICENSE`, `README.md`,
`INSTALL.md`, `REQUIREMENTS.md`, `STATUS.md`, and `CITATION.cff`. Compare the
archive DOI and revision with the values in `STATUS.md` and `CITATION.cff`.

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

This recomputes software accuracy from 50 checkpoints.

## 5. Select deeper checks

Use `RESULTS.md` to choose a paper result and its command. The labels mean:

- **Included:** the stated software result can be recomputed or regenerated.
- **Partial:** only the stated portion is recomputed.
- **Render:** formatting committed values, not reproducing an experiment.

Long or optional workflows:

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

## 6. Scope and pass criteria

The artifact supports independent regeneration of the results marked Included
or Partial in `RESULTS.md`.

## Badge evidence

- **Available:** version-specific DOI, open-source license, complete archived
  revision, documentation, paper, code, data/checkpoints.
- **Reviewed:** clean installation, successful smoke test, and successful
  selected live runs with documented outputs.
- **Reproducible:** an independent evaluator regenerates the computational
  results that are explicitly supported in `RESULTS.md`.

Official requirements:
<https://esweek.org/call-for-artifacts-codessisss/>.
