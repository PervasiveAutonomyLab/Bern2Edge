# Artifact status — badges claimed

This artifact is submitted for the CODES+ISSS / ESWEEK artifact-evaluation badges
**Available**, **Reviewed (Functional)**, and **Reproducible**.

## Available

- The complete source, documentation, shipped model checkpoints, and result CSVs
  are archived in a public repository with a persistent DOI (see the artifact
  submission for the archival link).
- An open-source license is included ([LICENSE](LICENSE), MIT), and citation
  metadata in [CITATION.cff](CITATION.cff) (add the Zenodo DOI after archiving).
- Datasets are obtained programmatically from their public sources (OpenML for
  Adult; the US Census ACS PUMS via `folktables` for ACS Income), so the artifact
  is self-contained given network access on first run. The MAGIC Gamma Telescope
  data (`MAGIC/magic04.data`) is shipped in the repository, so MAGIC needs no
  network.

## Reviewed (Functional)

- [INSTALL.md](INSTALL.md) gives a one-command install and a seconds-long,
  network-free **smoke test** (`python ACS/make_table_xi.py`) that
  renders the paper's TABLE XI from the committed results.
- Every table has a documented, runnable command that produces the stated output
  (see [README.md](README.md) and `ACS/README.md`). Producer scripts
  write CSV/JSON; renderer scripts (`make_table*.py`, `make_table_xi.py`) emit the
  Markdown/LaTeX tables. All metrics are recomputed from saved weights, never read
  from logs.
- [REQUIREMENTS.md](REQUIREMENTS.md) pins the software/hardware environment.

## Reproducible

- **TABLE XI (`ACS/`) — exact.** The shipped per-seed checkpoints
  (`results/_multiseed_cache/seed*/models.pt`) plus the deterministic CART
  fallback let an independent party regenerate TABLE XI bit-for-bit:
  `run_multiseed.py` → `make_table_xi.py`. The regenerated
  `metrics_multiseed_raw.csv` is byte-identical to the committed copy. (The
  renderer computes GEO/TEMP-AVG, std, and Δ self-consistently over the
  {MS, WY, WV}/{2019, 2021, 2022} conditions; all means and per-condition values
  match the published PDF — see `ACS/README.md` for two documented
  cosmetic std/Δ differences.)
- **Full retrain.** Deleting the checkpoint cache retrains everything from
  scratch. The rules' CART numbers are deterministic given the student; the
  network accuracy rows land within the reported seed std (not bit-identical,
  because the reused Bernstein init/warmup differ from the original by design — the
  shared modules are not modified). Documented in `ACS/README.md`.
- **MAGIC TABLE V & TABLE X — exact.** `MAGIC/make_table5.py` and
  `MAGIC/make_table_x.py` render both tables from the shipped metrics
  (`MAGIC/5_fold_results.csv`, `MAGIC/results/table_x.csv`) — the published numbers,
  no network or training. TABLE X is also recomputed live from the shipped student
  weights + headline rule JSON by `MAGIC/relu_lirpa_certify.py` (it reproduces
  `results/table_x.csv` and asserts certificate soundness + monotonicity).
  Training/extraction from scratch reproduces TABLE V within seed std (a from-scratch
  student has different learned neuron shapes, hence a different rule count).
- **Other tables** (Adult rule extraction / penalty sweep, KD compression) are
  reproduced by their per-table commands in [README.md](README.md).

## Reuse / provenance note

The ACS experiment reuses the repository's shared modules unchanged
(`bern2edge.models`, `bern2edge.bernstein`, `bern2edge.kdtrain`,
`bern2edge.rule_extraction`) via a thin compatibility shim `ACS/_compat.py`; ACS data
loading lives in `bern2edge/data.py` (`bern2edge.data.acs_income`). No shared module
was modified to add this experiment, so the previously published Adult / Cover Type /
HIGGS results are unaffected.

The MAGIC experiment reuses the shared `bern2edge.data` (its MAGIC loader) and
`bern2edge.kdtrain` for training; its model core, rule extractor, and certifier are kept local to
`MAGIC/` (the rule-extraction method differs from `bern2edge/rule_extraction/`, and the
Bernstein interval-bound certification lives only in `MAGIC/bern_net.py`). MAGIC's
`magic04.data` is shipped, so it needs no network. `auto_LiRPA` (git-install) is
required only to recompute TABLE X's ReLU column from scratch.
