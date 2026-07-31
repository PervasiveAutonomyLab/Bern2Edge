# Covertype TABLE II artifact

This directory reproduces TABLE II, the Covertype accuracy comparison under
matched latency and BRAM budgets. It contains the ten reported student models,
five cross-validation checkpoints per model, and the HLS measurements used in
the table.

## Quick reproduction

From the repository root, after installing the project dependencies:

```bash
python cover_type/reproduce_table_ii.py
```

The script downloads scikit-learn's Covertype dataset on the first run, rebuilds
the fixed five folds (`seed=42`), evaluates all 50 `.pth` checkpoints, and:

1. verifies every recomputed fold accuracy against the accuracy embedded in its
   checkpoint with a `0.005` percentage-point cross-environment tolerance;
2. computes the sample mean and sample standard deviation over the five folds
   and verifies every mean against its full-precision published reference with
   a `1e-8` percentage-point tolerance (to accommodate decimal serialization
   of the source CSV);
3. matches each accuracy group to its actual HLS model by architecture,
   activation, and Bernstein degree;
4. writes `table_ii_checkpoint_results.csv` and `table_ii_results.csv`.

Expected final message:

```text
Verified 50 checkpoints and all 10 five-fold means.
Wrote cover_type/table_ii_checkpoint_results.csv
Wrote cover_type/table_ii_results.csv
```

## Files

- `student_model_weights/`: all 50 Table II checkpoints (10 models × 5 folds).
- `covertype_hls_results.csv`: raw HLS synthesis results supplied for the paper.
- `table_ii_checkpoint_results.csv`: one row per checkpoint, including its
  project-relative path, stored accuracy, recomputed accuracy, and difference.
- `table_ii_results.csv`: merged Table II accuracy/HLS results, including the
  five checkpoint paths used for every reported mean.
- `reproduce_table_ii.py`: executable end-to-end checkpoint evaluator and join.
- `covertype_teacher_weights.pth` and `run_kd_experiments.py`: teacher and
  from-scratch student-training driver.

## Published display and exact values

The merged CSV retains full-precision five-fold statistics. Rounded to two
decimal places, the accuracy means are:

| Latency | BRAM | Bernstein | ReLU | ΔAcc |
|---:|---:|---:|---:|---:|
| ≤200 | ≤30 | 76.50 | 74.93 | +1.57 |
| ≤400 | ≤55 | 82.98 | 81.87 | +1.11 |
| ≤1000 | ≤120 | 91.09 | 88.97 | +2.13 |
| ≤2500 | ≤140 | 95.05 | 93.53 | +1.52 |
| ≤7500 | ≤180 | 96.39 | 95.62 | +0.77 |

For the fourth Bernstein row, the exact five-checkpoint sample standard
deviation is `0.056756...`, which rounds to `0.06`. The paper image displays
`0.05`; the artifact reports the value recomputed from all five shipped
checkpoints without altering it.

## Fresh HLS reproduction

Generate the ten Bernstein and ReLU HLS source projects used by Table II:

```bash
python cover_type/reproduce_table_ii_hardware.py --generate-only
```

The driver uses fold 0 (`seed=1000`) as the canonical hardware checkpoint for
each model. After loading Vitis, omit `--generate-only` to run csim/csynth,
parse the reports, and compare fresh latency, BRAM, DSP, FF, and LUT values
with `covertype_hls_results.csv`. The Vitis version and setup command are
currently TODO. The generic `.pth` compiler is documented in
[`hls/README.md`](../hls/README.md).

## Artifact-evaluation scope

This directory provides the executable code, model weights, raw measurements,
derived measurements, explicit model-to-row provenance, and expected output
needed to review and independently reproduce TABLE II. Repository-level
installation, requirements, license, status, citation, paper, and the
all-results coverage matrix are at the project root.

PyTorch evaluation can move a very small number of samples across a decision
boundary on a different software/hardware stack. The per-checkpoint CSV exposes
this rather than hiding it; the largest accepted difference is bounded to
`0.005` percentage points. The acceptance criterion for the reported result is
the full-precision five-fold mean.
