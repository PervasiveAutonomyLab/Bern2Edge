"""Reproduce Covertype TABLE II from the shipped five-fold checkpoints.

The script evaluates every checkpoint on its original fold, verifies the
recomputed accuracy against the value stored in the checkpoint, and joins the
five-fold accuracy summaries to the committed HLS synthesis measurements.

Run from any directory:

    python cover_type/reproduce_table_ii.py

Outputs:
    cover_type/table_ii_checkpoint_results.csv
    cover_type/table_ii_results.csv
"""

from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(REPO))

from bern2edge.data import covertype_dataloaders_kfold  # noqa: E402
from bern2edge.kdtrain import eval_one_epoch  # noqa: E402
from bern2edge.models import FCModel  # noqa: E402


WEIGHTS_DIR = HERE / "student_model_weights"
HLS_CSV = HERE / "covertype_hls_results.csv"
CHECKPOINT_CSV = HERE / "table_ii_checkpoint_results.csv"
TABLE_CSV = HERE / "table_ii_results.csv"
DATA_SEED = 42
N_FOLDS = 5
CHECKPOINT_ACCURACY_TOLERANCE_PP = 0.005
MEAN_ACCURACY_TOLERANCE_PP = 0.005

# These are the ten checkpoint groups used by TABLE II. Hyperparameters are
# included to make the architecture-to-model mapping explicit and auditable.
CONFIGS = [
    {
        "budget_latency_cycles": 200,
        "budget_bram_18k": 30,
        "hidden_layers": [8],
        "bern": {"degree": 3, "alpha": 0.85, "temperature": 2.0},
        "relu": {"degree": None, "alpha": 0.5, "temperature": 4.0},
    },
    {
        "budget_latency_cycles": 400,
        "budget_bram_18k": 55,
        "hidden_layers": [32],
        "bern": {"degree": 3, "alpha": 0.5, "temperature": 2.0},
        "relu": {"degree": None, "alpha": 0.0, "temperature": 1.0},
    },
    {
        "budget_latency_cycles": 1000,
        "budget_bram_18k": 120,
        "hidden_layers": [64, 32],
        "bern": {"degree": 3, "alpha": 0.5, "temperature": 2.0},
        "relu": {"degree": None, "alpha": 0.5, "temperature": 2.0},
    },
    {
        "budget_latency_cycles": 2500,
        "budget_bram_18k": 140,
        "hidden_layers": [128, 64],
        "bern": {"degree": 5, "alpha": 0.0, "temperature": 1.0},
        "relu": {"degree": None, "alpha": 0.5, "temperature": 2.0},
    },
    {
        "budget_latency_cycles": 7500,
        "budget_bram_18k": 180,
        "hidden_layers": [256, 128],
        "bern": {"degree": 5, "alpha": 0.5, "temperature": 2.0},
        "relu": {"degree": None, "alpha": 0.5, "temperature": 2.0},
    },
]

# Exact five-fold means obtained by evaluating the shipped checkpoints with the
# published split. The first four are also present in
# results_kd_ct_summary.csv; the remaining six are obtained from
# final_results_kd_with_folds.csv and the shipped checkpoints. Values printed in
# the paper are these means rounded to two decimals.
EXPECTED_MEANS = {
    (200, "bern"): 76.495777492,
    (200, "relu"): 74.92817147000001,
    (400, "bern"): 82.97939232799999,
    (400, "relu"): 81.866853314,
    (1000, "bern"): 91.09188544152744,
    (1000, "relu"): 88.96594455663669,
    (2500, "bern"): 95.05438773636864,
    (2500, "relu"): 93.53130163392693,
    (7500, "bern"): 96.39136221773454,
    (7500, "relu"): 95.61800073434918,
}


def checkpoint_name(hidden_layers, activation, params, seed):
    arch = "x".join(map(str, [54, *hidden_layers, 7]))
    degree = params["degree"] if params["degree"] is not None else "NA"
    alpha = f"{params['alpha']:g}"
    temperature = f"{params['temperature']:g}"
    return (
        f"kd_fc_{arch}_{activation}_deg{degree}_alpha{alpha}_T{temperature}"
        f"_lr0.006_wd0.0001_seed{seed}.pth"
    )


def hls_model_name(hidden_layers, activation, degree):
    arch = "x".join(map(str, [54, *hidden_layers, 7]))
    return f"bern_d{degree}_{arch}" if activation == "bern" else f"relu_{arch}"


def load_hls():
    frame = pd.read_csv(HLS_CSV, skipinitialspace=True)
    frame.columns = frame.columns.str.strip()
    for column in frame.select_dtypes(include="object"):
        frame[column] = frame[column].str.strip()
    if frame["model"].duplicated().any():
        raise RuntimeError("HLS CSV contains duplicate model names")
    return frame.set_index("model")


def validate_metadata(checkpoint, config, activation, params, seed, path):
    expected_arch = [54, *config["hidden_layers"], 7]
    expected = {
        "arch": expected_arch,
        "activation": activation,
        "degree": params["degree"],
        "alpha": params["alpha"],
        "T": params["temperature"],
        "seed": seed,
    }
    for key, value in expected.items():
        actual = checkpoint.get(key)
        equal = actual == value
        if isinstance(value, float):
            equal = math.isclose(float(actual), value, rel_tol=0.0, abs_tol=1e-12)
        if not equal:
            raise RuntimeError(
                f"{path.name}: metadata {key}={actual!r}, expected {value!r}"
            )


def evaluate_all():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    folds, _, _ = covertype_dataloaders_kfold(
        batch_size=1024, k=N_FOLDS, seed=DATA_SEED, test_size=0.15
    )
    criterion = nn.CrossEntropyLoss()
    rows = []

    for config in CONFIGS:
        for activation in ("bern", "relu"):
            params = config[activation]
            for fold_id in range(N_FOLDS):
                seed = 1000 + fold_id
                path = WEIGHTS_DIR / checkpoint_name(
                    config["hidden_layers"], activation, params, seed
                )
                if not path.is_file():
                    raise FileNotFoundError(f"Missing TABLE II checkpoint: {path}")

                checkpoint = torch.load(path, map_location=device)
                validate_metadata(
                    checkpoint, config, activation, params, seed, path
                )
                model = FCModel(
                    layer_sizes=checkpoint["arch"],
                    degree=checkpoint["degree"],
                    act=checkpoint["activation"],
                    last_bern=False,
                ).to(device)
                model.load_state_dict(checkpoint["state_dict"])
                _, recomputed = eval_one_epoch(
                    model, folds[fold_id][2], criterion, device
                )
                recomputed *= 100.0
                stored = float(checkpoint["test_acc"])
                difference = abs(recomputed - stored)
                if difference > CHECKPOINT_ACCURACY_TOLERANCE_PP:
                    raise RuntimeError(
                        f"{path.name}: recomputed accuracy {recomputed:.12f} "
                        f"does not match stored accuracy {stored:.12f}; "
                        f"difference={difference:.3g} pp"
                    )

                rows.append(
                    {
                        "budget_latency_cycles": config["budget_latency_cycles"],
                        "budget_bram_18k": config["budget_bram_18k"],
                        "activation": activation,
                        "hidden_layers": json.dumps(config["hidden_layers"]),
                        "degree": params["degree"],
                        "alpha": params["alpha"],
                        "temperature": params["temperature"],
                        "fold_id": fold_id,
                        "seed": seed,
                        "checkpoint_path": path.relative_to(REPO).as_posix(),
                        "stored_accuracy_percent": stored,
                        "recomputed_accuracy_percent": recomputed,
                        "absolute_difference_pp": difference,
                    }
                )
    return pd.DataFrame(rows)


def build_table(checkpoints, hls):
    rows = []
    for config in CONFIGS:
        row = {
            "budget_latency_cycles": config["budget_latency_cycles"],
            "budget_bram_18k": config["budget_bram_18k"],
        }
        for activation in ("bern", "relu"):
            params = config[activation]
            subset = checkpoints[
                (checkpoints["budget_latency_cycles"]
                 == config["budget_latency_cycles"])
                & (checkpoints["activation"] == activation)
            ].sort_values("fold_id")
            if len(subset) != N_FOLDS:
                raise RuntimeError(
                    f"Expected {N_FOLDS} {activation} folds for "
                    f"latency budget {config['budget_latency_cycles']}"
                )

            prefix = "bernstein" if activation == "bern" else "relu"
            model_name = hls_model_name(
                config["hidden_layers"], activation, params["degree"]
            )
            if model_name not in hls.index:
                raise RuntimeError(f"Missing HLS row: {model_name}")
            synth = hls.loc[model_name]
            accuracies = subset["recomputed_accuracy_percent"]
            paths = subset["checkpoint_path"].tolist()
            accuracy_mean = accuracies.mean()
            expected_mean = EXPECTED_MEANS[
                (config["budget_latency_cycles"], activation)
            ]
            mean_difference = abs(accuracy_mean - expected_mean)
            if mean_difference > MEAN_ACCURACY_TOLERANCE_PP:
                raise RuntimeError(
                    f"{model_name}: five-fold mean {accuracy_mean:.12f} "
                    f"does not match expected mean {expected_mean:.12f}; "
                    f"difference={mean_difference:.3g} pp"
                )

            row.update(
                {
                    f"{prefix}_model": model_name,
                    f"{prefix}_hidden_layers": json.dumps(
                        config["hidden_layers"]
                    ),
                    f"{prefix}_degree": params["degree"],
                    f"{prefix}_alpha": params["alpha"],
                    f"{prefix}_temperature": params["temperature"],
                    f"{prefix}_accuracy_mean_percent": accuracy_mean,
                    f"{prefix}_accuracy_std_percent": accuracies.std(ddof=1),
                    f"{prefix}_expected_accuracy_mean_percent": expected_mean,
                    f"{prefix}_mean_absolute_difference_pp": mean_difference,
                    f"{prefix}_n_folds": len(accuracies),
                    f"{prefix}_latency_cycles": int(synth["latency_cycles"]),
                    f"{prefix}_bram_18k": int(synth["BRAM_18K"]),
                    f"{prefix}_checkpoint_paths": json.dumps(paths),
                }
            )
        row["delta_accuracy_pp"] = (
            row["bernstein_accuracy_mean_percent"]
            - row["relu_accuracy_mean_percent"]
        )
        rows.append(row)
    return pd.DataFrame(rows)


def main():
    os.chdir(HERE)
    checkpoints = evaluate_all()
    table = build_table(checkpoints, load_hls())
    checkpoints.to_csv(CHECKPOINT_CSV, index=False, float_format="%.12g")
    table.to_csv(TABLE_CSV, index=False, float_format="%.12g")

    display = table[
        [
            "budget_latency_cycles",
            "budget_bram_18k",
            "bernstein_accuracy_mean_percent",
            "bernstein_accuracy_std_percent",
            "bernstein_latency_cycles",
            "bernstein_bram_18k",
            "relu_accuracy_mean_percent",
            "relu_accuracy_std_percent",
            "relu_latency_cycles",
            "relu_bram_18k",
            "delta_accuracy_pp",
        ]
    ]
    print(display.to_string(index=False, float_format=lambda value: f"{value:.6f}"))
    print(
        f"\nVerified {len(checkpoints)} checkpoints and all 10 "
        "five-fold means."
    )
    print(f"Wrote {CHECKPOINT_CSV.relative_to(REPO)}")
    print(f"Wrote {TABLE_CSV.relative_to(REPO)}")


if __name__ == "__main__":
    main()
