# Transformer FFN substitution — TinyBERT4 / SST-2 (TABLE XII)

Extends Bern2Edge from tabular MLPs to the **FFN sublayers of a transformer**.
All four TinyBERT4 encoder layers have their `312 → 1200 → 312` GeLU FFN replaced
by a narrow `312 → H → 312` FFN (H ∈ {312, 600}) with either a **Bernstein**
(LUT) activation or a plain **GeLU** control. Everything else — embeddings,
attention, pooler, classifier — is stock TinyBERT4.

**TABLE XII, SST-2 accuracy row** (recomputed here from the shipped weights):

|                 | TinyBERT4 | h=600 GeLU | h=600 Bern | h=312 GeLU | h=312 Bern |
| --------------- | --------- | ---------- | ---------- | ---------- | ---------- |
| Acc. (%)        | 90.37     | 90.02      | **90.48**  | 89.11      | 90.02      |
| FFN params (×4) | 3,001,248 | 1,501,248  | 1,539,648  | 781,248    | 801,216    |

At matched width Bernstein beats GeLU (+0.91 pp at h=312, +0.46 pp at h=600);
Bern h=312 ties GeLU h=600 at ~half the FFN parameters; Bern h=600 exceeds the
teacher at −48.7 % FFN parameters.

> **Scope.** This folder covers **training and evaluation**. The latency and FPGA
> resource rows of TABLE XII come from Vitis HLS synthesis on a KV260 and are
> **transcribed from the paper** into `results/table_xii_hls.csv` — they are not
> recomputed here. The renderer labels them as such.

## Reproduce

Three tiers, cheapest first. Run from the repository root.

**1. Render the table** (seconds, no network, no GPU) — the smoke test:

```bash
python Transformer/make_table_xii.py
```

Reads the two committed CSVs and prints TABLE XII (also writes
`results/RESULTS_table_xii.md` and `results/table_xii.tex`).

**2. Recompute the accuracy row from the weights** — the reproducibility claim:

```bash
python Transformer/eval_release.py       # GPU if present; --device cpu also works
python Transformer/make_table_xii.py
```

Loads the teacher and all four releases, evaluates them on the SST-2 dev split,
and checks each against the accuracy recorded when the release was built. Any
deviation above `--tol` (default 1e-4) fails the run. This regenerates
`results/table_xii_acc.csv` **byte-identically**. Needs network once, to fetch
SST-2 (via `datasets`) and the stock BERT parts (`huawei-noah/TinyBERT_General_4L_312D`).

Single-sentence check:

```bash
python Transformer/load_and_run.py bern_h312 "this movie was a delight"
```

**3. Train from scratch** (GPU, hours) — see below. This reproduces the results
**approximately, not exactly**; the shipped weights are the authoritative source
for the published numbers.

## Training: three stages

The transformer experiment uses a different technique from the tabular ones —
the student FFN is first fit to the teacher's FFN _in isolation_, then swapped
in, then fine-tuned end to end:

| Stage | Script                      | What it does                                                                               | Loss    |
| ----- | --------------------------- | ------------------------------------------------------------------------------------------ | ------- |
| 1     | `stage1_general_match.py`   | fit a `312→H→312` FFN to one encoder layer's FFN of the **general** pretrained TinyBERT4   | MSE     |
| 2     | `stage2_finetuned_match.py` | warm-start from Stage 1, refit against the **SST-2-fine-tuned** teacher's FFN distribution | MSE     |
| 3     | `stage3_substitute.py`      | cold-swap all four fitted FFNs into the teacher, then KD fine-tune                         | KD + CE |

Bernstein specifics: degree 15 (16 control points per channel); coefficients start
from a near-identity **ramp** (`bern2edge.bernstein.BernsteinLayer(init="ramp")`)
rather than the random `xavier` the tabular experiments use, because the layer is
fitting an existing function. Per-channel input bounds are calibrated from data
and recalibrated periodically, then frozen. Stages 1–2 use separate optimizer
groups (linear `lr=1e-3 wd=0.01`, coefficients `lr=3e-3 wd=0.0`).

Run one variant end to end:

```bash
# full recipe (~3 h on one A30 for h=312)
ACT=bern HIDDEN=312 LAYERS="0 1 2 3" bash Transformer/run_variant.sh

# pipeline smoke: layer 0 only, few epochs (~7 min) -- proves it runs
ACT=bern HIDDEN=312 LAYERS="0" MODE=smoke bash Transformer/run_variant.sh
```

`ACT` ∈ {`bern`, `gelu`}, `HIDDEN` ∈ {312, 600}. Output goes to
`scratch/<tag>/`, never overwriting `models/`. Stages are skipped when their
checkpoint exists, so an interrupted run resumes. Convert a finished Stage-3
checkpoint into a release with `build_release.py` (it verifies the accuracy
before saving).

The teacher itself can be retrained from stock TinyBERT4:

```bash
python Transformer/finetune_teacher.py --task sst2
```

**Why from-scratch does not reproduce the digits.** Stage 1 starts from a random
perturbation of the ramp, Stage 2's bound recalibration depends on batch order,
and Stage 3's KD is a short fine-tune from a cold swap. The accuracies land near
the published ones, not on them. Use the shipped weights for exact numbers.

## Layout

```
Transformer/
  SPEC.md                    the network + both FFN kernels, in full (read this
                             for the Bernstein LUT math)

  shared_utils.py            SST-2/GLUE loaders, FFN capture, bound calibration,
                             FFN monkey-patch, evaluate_classifier
  bernbert.py                BernBertForSequenceClassification -- the clean module
                             with the FFN math written out and dead GeLU weights
                             removed (what the releases load into)

  finetune_teacher.py        stock TinyBERT4 -> SST-2 GeLU teacher
  stage1_general_match.py    Stage 1
  stage2_finetuned_match.py  Stage 2
  stage3_substitute.py       Stage 3
  run_variant.sh             drives all three stages for one (ACT, HIDDEN)
  build_release.py           Stage-3 checkpoint -> verified release .pt + .meta.json

  eval_release.py            recompute the accuracy row -> results/table_xii_acc.csv
  make_table_xii.py          render TABLE XII (Markdown + LaTeX)
  load_and_run.py            classify one sentence with any variant

  models/
    release_{bern,gelu}_h{312,600}.pt   the four TABLE XII variants
    release_*.meta.json                 act / hidden / degree / verified accuracy
    teacher_gelu_9037.pt                the GeLU TinyBERT4 teacher (90.37 %)
    stage1/, stage2/                    per-layer warm-starts for all four variants
  results/
    table_xii_acc.csv        accuracy row -- RECOMPUTED by eval_release.py
    table_xii_hls.csv        latency/DSP/BRAM/FF/LUT -- TRANSCRIBED from the paper
    table_xii.tex, RESULTS_table_xii.md
```

## Reuse of the shared library

Unlike `MAGIC/`, this folder keeps **no local model core**: the FFN student is a
plain `bern2edge.models.FCModel` over `bern2edge.bernstein.BernsteinLayer`. Two
opt-in, default-preserving options were added to those shared modules for it:

- `BernsteinLayer(..., init="ramp")` — the near-identity initialization described
  above. The default stays `"xavier"`, so every tabular experiment is unaffected.
- `FCModel(..., act="gelu")` — the matched-width GeLU control. Previously any
  non-`"bern"` activation fell through to ReLU.

Both were verified not to change existing behavior: `MAGIC/results/table_x.csv`
and `ACS/results/metrics_multiseed_raw.csv` both regenerate byte-identically, and
default-constructed layers produce bit-identical coefficients.

## Note on the paper text

Section IV-H says "10 epochs of KD fine-tuning". The shipped weights were trained
with **8** Stage-3 epochs and best-validation checkpoint selection (the h=312
Bernstein variant peaks at epoch 5 of 8). `run_variant.sh` uses 8 to match the
artifact.
