"""Reproduce Bern2Edge paper Table I from the shipped model weights.

Accuracy and cross-entropy are evaluated live for all 90 student checkpoints.
FPGA fields are joined from ``table_i_hls_results.csv``; that file is a direct
transcription of the paper because this artifact does not rerun Vitis HLS.

Run from any directory:

    python table_i_compression/reproduce_table_i.py

Outputs:
    table_i_checkpoint_results.csv  one row per checkpoint/fold
    table_i_results.csv             the 18 rows of paper Table I
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(REPO))

from bern2edge.data import (  # noqa: E402
    adult_ordinal_dataloaders_kfold,
    covertype_dataloaders_kfold,
    higgs_small_dataloaders_kfold,
)
from bern2edge.kdtrain import eval_one_epoch  # noqa: E402
from bern2edge.models import FCModel  # noqa: E402

HLS_CSV = HERE / "table_i_hls_results.csv"
CHECKPOINT_CSV = HERE / "table_i_checkpoint_results.csv"
TABLE_CSV = HERE / "table_i_results.csv"
N_FOLDS = 5
ACCURACY_TOLERANCE_PP = 0.05

DATASETS = {
    "HIGGS-Small": {
        "directory": REPO / "higgs_small",
        "architectures": {(28, 16, 2), (28, 16, 8, 2), (28, 128, 2)},
        "seed_base": 42,
    },
    "Covertype": {
        "directory": REPO / "cover_type",
        "architectures": {
            (54, 64, 32, 7),
            (54, 128, 64, 7),
            (54, 256, 128, 7),
        },
        "seed_base": 1000,
    },
    "Adult": {
        "directory": REPO / "Adult",
        "architectures": {(14, 16, 2), (14, 32, 16, 2), (14, 128, 2)},
        "seed_base": 6,
    },
}


def build_loaders(dataset: str):
    if dataset == "HIGGS-Small":
        folds, _, _ = higgs_small_dataloaders_kfold(
            n_splits=N_FOLDS, batch_size=512, seed=42
        )
    elif dataset == "Covertype":
        folds, _, _ = covertype_dataloaders_kfold(
            batch_size=1024, k=N_FOLDS, seed=42, test_size=0.15
        )
    else:
        folds, _, _ = adult_ordinal_dataloaders_kfold(
            teacher_ckpt=str(REPO / "Adult" / "adult_teacher_ordinal.pt"),
            batch_size=256,
            eval_bs=8192,
            k=N_FOLDS,
            seed=6,
        )
    return folds


def load_checkpoint(path: Path, device: torch.device):
    # Adult's teacher contains sklearn objects, and using the same explicit
    # setting everywhere keeps compatibility across PyTorch versions.
    return torch.load(path, map_location=device, weights_only=False)


def evaluate_dataset(dataset: str, device: torch.device):
    config = DATASETS[dataset]
    folds = build_loaders(dataset)
    paths = sorted((config["directory"] / "student_model_weights").glob("*.pth"))
    criterion = nn.CrossEntropyLoss()
    rows = []

    for path in paths:
        checkpoint = load_checkpoint(path, device)
        architecture = tuple(checkpoint["arch"])
        if architecture not in config["architectures"]:
            continue
        fold_id = int(checkpoint["seed"]) - config["seed_base"]
        if not 0 <= fold_id < N_FOLDS:
            raise RuntimeError(f"{path.name}: invalid fold seed {checkpoint['seed']}")

        model = FCModel(
            layer_sizes=checkpoint["arch"],
            degree=checkpoint["degree"],
            act=checkpoint["activation"],
            last_bern=False,
        ).to(device)
        model.load_state_dict(checkpoint["state_dict"])
        train_ce, _ = eval_one_epoch(model, folds[fold_id][0], criterion, device)
        test_ce, test_accuracy = eval_one_epoch(
            model, folds[fold_id][2], criterion, device
        )
        test_accuracy *= 100.0
        stored_accuracy = float(checkpoint["test_acc"])
        difference = abs(test_accuracy - stored_accuracy)
        if difference > ACCURACY_TOLERANCE_PP:
            raise RuntimeError(
                f"{path.name}: recomputed accuracy {test_accuracy:.6f}% does "
                f"not match checkpoint metadata {stored_accuracy:.6f}% "
                f"(difference {difference:.6f} pp)"
            )

        rows.append(
            {
                "dataset": dataset,
                "architecture": "x".join(map(str, architecture)),
                "activation": checkpoint["activation"],
                "degree": checkpoint["degree"],
                "fold_id": fold_id,
                "seed": checkpoint["seed"],
                "checkpoint_path": path.relative_to(REPO).as_posix(),
                "recomputed_accuracy_percent": test_accuracy,
                "recomputed_train_ce": train_ce,
                "recomputed_test_ce": test_ce,
                "stored_accuracy_percent": stored_accuracy,
                "absolute_accuracy_difference_pp": difference,
            }
        )

    expected = len(config["architectures"]) * 2 * N_FOLDS
    if len(rows) != expected:
        raise RuntimeError(
            f"{dataset}: found {len(rows)} Table I checkpoints; expected {expected}"
        )
    return rows


def aggregate(checkpoints: pd.DataFrame):
    summary = (
        checkpoints.groupby(
            ["dataset", "architecture", "activation"], as_index=False, sort=False
        )
        .agg(
            degree=("degree", "first"),
            accuracy_mean_percent=("recomputed_accuracy_percent", "mean"),
            accuracy_std_percent=("recomputed_accuracy_percent", "std"),
            ce_loss_mean=("recomputed_train_ce", "mean"),
            ce_loss_std=("recomputed_train_ce", "std"),
            test_ce_mean=("recomputed_test_ce", "mean"),
            test_ce_std=("recomputed_test_ce", "std"),
            n_folds=("fold_id", "count"),
        )
    )

    summary["delta_accuracy_pp"] = math.nan
    summary["delta_ce_loss"] = math.nan
    for (_, architecture), group in summary.groupby(["dataset", "architecture"]):
        bern = group[group.activation == "bern"]
        relu = group[group.activation == "relu"]
        if len(bern) != 1 or len(relu) != 1:
            raise RuntimeError(f"Missing ReLU/Bern pair for {architecture}")
        mask = (summary.dataset == group.iloc[0].dataset) & (
            summary.architecture == architecture
        )
        summary.loc[mask, "delta_accuracy_pp"] = (
            bern.iloc[0].accuracy_mean_percent - relu.iloc[0].accuracy_mean_percent
        )
        summary.loc[mask, "delta_ce_loss"] = (
            bern.iloc[0].ce_loss_mean - relu.iloc[0].ce_loss_mean
        )

    hls = pd.read_csv(HLS_CSV)
    table = summary.merge(
        hls,
        on=["dataset", "architecture", "activation"],
        how="left",
        validate="one_to_one",
    )
    if table[["latency_cycles", "dsp", "bram_18k", "lut"]].isna().any().any():
        raise RuntimeError("One or more Table I rows have no matching HLS entry")

    dataset_order = {"HIGGS-Small": 0, "Covertype": 1, "Adult": 2}
    activation_order = {"relu": 0, "bern": 1}
    architecture_order = {
        architecture: index
        for index, architecture in enumerate(hls["architecture"].drop_duplicates())
    }
    return (
        table.assign(
            _dataset=table.dataset.map(dataset_order),
            _architecture=table.architecture.map(architecture_order),
            _activation=table.activation.map(activation_order),
        )
        .sort_values(["_dataset", "_architecture", "_activation"])
        .drop(columns=["_dataset", "_architecture", "_activation"])
        .reset_index(drop=True)
    )


def print_table(table: pd.DataFrame):
    display = table.copy()
    display["accuracy_percent"] = display.apply(
        lambda row: (
            f"{row.accuracy_mean_percent:.2f} ± "
            f"{row.accuracy_std_percent:.2f}"
        ),
        axis=1,
    )
    display["delta_accuracy_ce"] = display.apply(
        lambda row: (
            f"{row.delta_accuracy_pp:+.2f} / {row.delta_ce_loss:+.3f}"
            if row.activation == "bern"
            else ""
        ),
        axis=1,
    )
    columns = [
        "dataset",
        "architecture",
        "activation",
        "accuracy_percent",
        "ce_loss_mean",
        "delta_accuracy_ce",
        "latency_cycles",
        "dsp",
        "bram_18k",
        "lut",
    ]
    print(display[columns].to_string(index=False, float_format=lambda x: f"{x:.3f}"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="Evaluation device (default: CUDA when available).",
    )
    args = parser.parse_args()
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    rows = []
    for dataset in DATASETS:
        print(f"Evaluating {dataset} on {device} ...", flush=True)
        rows.extend(evaluate_dataset(dataset, device))
    checkpoints = pd.DataFrame(rows)
    table = aggregate(checkpoints)
    checkpoints.to_csv(CHECKPOINT_CSV, index=False)
    table.to_csv(TABLE_CSV, index=False)
    print(f"\nWrote {CHECKPOINT_CSV.relative_to(HERE)}")
    print(f"Wrote {TABLE_CSV.relative_to(HERE)}\n")
    print_table(table)


if __name__ == "__main__":
    main()
