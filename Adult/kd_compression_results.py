"""
Adult (ordinal) — Bernstein-vs-ReLU summary table, computed from model weights.

Loads every published student checkpoint in ``student_model_weights/``, re-runs
each model on its cross-validation fold, and aggregates the metrics that make up
the Bernstein-vs-ReLU comparison table:

  * Acc (%)   : test accuracy, mean ± std over the 5 folds.
  * Train CE  : cross-entropy on the fold's training set (matches the paper
                table's "CE Loss" column).
  * Test CE   : cross-entropy on the held-out test set (reported alongside).

All numbers are recomputed from the weights — nothing is read from raw training
logs. Writes ``kd_compression_results.csv`` and prints a formatted table.

Checkpoint -> fold mapping: each file stores ``seed`` = 6 + fold_id, and the
folds come from ``adult_ordinal_dataloaders_kfold`` (the same split protocol used
at training time, anchored on the teacher checkpoint).

Run:
    python kd_compression_results.py
"""

import glob
import os
import sys

import pandas as pd
import torch
import torch.nn as nn

# Shared library modules live one directory up (Bern2Edge/).
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from bern2edge.models import FCModel                              # noqa: E402
from bern2edge.data import adult_ordinal_dataloaders_kfold        # noqa: E402
from bern2edge.kdtrain import eval_one_epoch                       # noqa: E402

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

WEIGHTS_DIR  = "student_model_weights"
TEACHER_CKPT = "adult_teacher_ordinal.pt"
OUT_CSV      = "kd_compression_results.csv"
DATA_SEED    = 6      # must match the seed used at training time


def arch_label(arch):
    """Render a width list as the paper's set notation, e.g. {14, 16, 2}."""
    return "{" + ", ".join(str(w) for w in arch) + "}"


def evaluate_checkpoints(folds):
    """Load every checkpoint, recompute metrics on its fold, return a tidy DataFrame."""
    criterion = nn.CrossEntropyLoss()
    rows = []

    for path in sorted(glob.glob(os.path.join(WEIGHTS_DIR, "*.pth"))):
        ckpt = torch.load(path, map_location=device, weights_only=False)
        arch, act, degree = ckpt["arch"], ckpt["activation"], ckpt["degree"]
        fold_id = ckpt["seed"] - DATA_SEED
        train_loader, _, test_loader = folds[fold_id]

        # Rebuild the exact student and load its weights (Bernstein input_bounds
        # buffers are part of the state_dict, so calibration is restored too).
        model = FCModel(layer_sizes=arch, degree=degree, act=act, last_bern=False).to(device)
        model.load_state_dict(ckpt["state_dict"])

        train_ce, _        = eval_one_epoch(model, train_loader, criterion, device)
        test_ce, test_acc  = eval_one_epoch(model, test_loader, criterion, device)
        test_acc *= 100.0

        # Sanity check: recomputed accuracy should match the stored value.
        if abs(test_acc - ckpt["test_acc"]) > 0.05:
            print(f"  WARNING {os.path.basename(path)}: recomputed acc {test_acc:.2f}% "
                  f"!= stored {ckpt['test_acc']:.2f}%")

        rows.append({
            "arch":       arch_label(arch),
            "arch_size":  sum(arch),          # for ordering rows by model size
            "activation": act,
            "degree":     degree,
            "seed":       ckpt["seed"],
            "test_acc":   test_acc,
            "train_ce":   train_ce,
            "test_ce":    test_ce,
        })

    return pd.DataFrame(rows)


def summarize(per_model):
    """Aggregate per-(arch, activation) over folds and attach Bern-vs-ReLU deltas."""
    summary = (
        per_model
        .groupby(["arch", "arch_size", "activation"], as_index=False)
        .agg(
            degree=("degree", "first"),
            acc_mean=("test_acc", "mean"),
            acc_std=("test_acc", "std"),
            ce_train_mean=("train_ce", "mean"),
            ce_train_std=("train_ce", "std"),
            ce_test_mean=("test_ce", "mean"),
            ce_test_std=("test_ce", "std"),
            n_folds=("test_acc", "count"),
        )
    )

    # Per-architecture Bern - ReLU deltas.
    summary["delta_acc"] = float("nan")
    summary["delta_ce_train"] = float("nan")
    summary["delta_ce_test"] = float("nan")
    for arch, grp in summary.groupby("arch"):
        if {"bern", "relu"} <= set(grp["activation"]):
            b = grp[grp["activation"] == "bern"].iloc[0]
            r = grp[grp["activation"] == "relu"].iloc[0]
            mask = summary["arch"] == arch
            summary.loc[mask, "delta_acc"] = b["acc_mean"] - r["acc_mean"]
            summary.loc[mask, "delta_ce_train"] = b["ce_train_mean"] - r["ce_train_mean"]
            summary.loc[mask, "delta_ce_test"] = b["ce_test_mean"] - r["ce_test_mean"]

    # Order: smallest model first, ReLU above Bern within each architecture.
    act_order = {"relu": 0, "bern": 1}
    summary = (
        summary
        .assign(_act=summary["activation"].map(act_order))
        .sort_values(["arch_size", "_act"])
        .drop(columns="_act")
        .reset_index(drop=True)
    )
    return summary


def print_table(summary):
    """Print a console table mirroring the paper's Adult block."""
    print("\n" + "=" * 78)
    print(f"{'Arch.':<18}{'Act.':<7}{'Acc. (%)':<16}{'Train CE':<11}"
          f"{'Test CE':<10}{'ΔAcc / ΔCE':<14}")
    print("-" * 78)
    for arch, grp in summary.groupby("arch", sort=False):
        for _, row in grp.iterrows():
            acc = f"{row.acc_mean:.2f} ± {row.acc_std:.2f}"
            # ΔAcc / ΔCE(train) is shown once per architecture (on the Bern row).
            delta = (f"{row.delta_acc:+.2f} / {row.delta_ce_train:+.3f}"
                     if row.activation == "bern" else "")
            print(f"{arch:<18}{row.activation:<7}{acc:<16}"
                  f"{row.ce_train_mean:<11.3f}{row.ce_test_mean:<10.3f}{delta:<14}")
        print("-" * 78)


def main():
    folds, _, _ = adult_ordinal_dataloaders_kfold(
        teacher_ckpt=TEACHER_CKPT, batch_size=256, eval_bs=8192, k=5, seed=DATA_SEED
    )
    per_model = evaluate_checkpoints(folds)
    if per_model.empty:
        print(f"No checkpoints found in '{WEIGHTS_DIR}/'.")
        return

    summary = summarize(per_model)
    summary.drop(columns="arch_size").to_csv(OUT_CSV, index=False)
    print(f"\nWrote {OUT_CSV}")
    print_table(summary)


if __name__ == "__main__":
    main()
