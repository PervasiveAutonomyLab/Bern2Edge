"""Reproduce Adult Table IV from ready network and rule artifacts.

The script evaluates five Bernstein checkpoints and their matching
same_cov_alpha=0.5, conflict_alpha=0.1 rule/CART artifacts. It then joins the
supplied post-synthesis measurements in hardware_results.csv and renders the
table. HLS synthesis itself is not rerun.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn

HERE = Path(__file__).resolve().parent
ADULT = HERE.parent
REPO = ADULT.parent
sys.path.insert(0, str(REPO))

from Adult.evaluate_rule_artifacts import (  # noqa: E402
    evaluate_rule_artifact,
    load_adult_test_split,
)
from bern2edge.data import adult_ordinal_dataloaders_kfold  # noqa: E402
from bern2edge.kdtrain import eval_one_epoch  # noqa: E402
from bern2edge.models import FCModel  # noqa: E402

ARCHITECTURES = (
    "14x16x2",
    "14x32x2",
    "14x128x2",
    "14x16x8x2",
    "14x32x16x2",
)
HARDWARE_CSV = HERE / "hardware_results.csv"
METRICS_CSV = HERE / "table4_artifact_metrics.csv"
VALUES_CSV = HERE / "table4_values.csv"
TEX_PATH = HERE / "table4.tex"


def paths_for(architecture):
    stem = (
        f"kd_fc_{architecture}_bern_deg3_alpha0.5_T2_"
        "lr0.006_wd0.0001_seed6"
    )
    checkpoint = ADULT / "rule_checkpoints" / f"{stem}.pth"
    run_dir = ADULT / "rule_jsons" / f"{stem}_sca0.5_ca0.1"
    return checkpoint, run_dir / "rules_float.json"


def evaluate_lut(checkpoint_path, test_loader, device):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = FCModel(
        checkpoint["arch"],
        checkpoint["degree"],
        act=checkpoint["activation"],
        last_bern=checkpoint.get("last_bern", False),
    ).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    _, measured = eval_one_epoch(
        model, test_loader, nn.CrossEntropyLoss(), device
    )
    measured *= 100.0
    stored = float(checkpoint["test_acc"])
    if abs(measured - stored) > 0.05:
        raise RuntimeError(
            f"{checkpoint_path.name}: measured={measured}, stored={stored}"
        )
    return stored, measured


def evaluate_all(device):
    folds, _, _ = adult_ordinal_dataloaders_kfold(
        str(ADULT / "adult_teacher_ordinal.pt"),
        batch_size=256,
        eval_bs=8192,
        k=5,
        seed=6,
    )
    test_loader = folds[0][2]
    X_test, y_test = load_adult_test_split()
    rows = []

    for architecture in ARCHITECTURES:
        checkpoint_path, rule_path = paths_for(architecture)
        if not checkpoint_path.is_file() or not rule_path.is_file():
            raise FileNotFoundError(f"Missing artifacts for {architecture}")

        stored, measured = evaluate_lut(checkpoint_path, test_loader, device)
        rows.append(
            {
                "architecture": architecture,
                "method": "LUT",
                "artifact_type": "student_checkpoint",
                "artifact_path": checkpoint_path.relative_to(REPO).as_posix(),
                "fallback_artifact": "",
                "stored_accuracy_pct": stored,
                "evaluated_accuracy_pct": measured,
            }
        )

        metrics = evaluate_rule_artifact(
            rule_path, X_test, y_test, conflict_strategy="coverage"
        )
        rows.append(
            {
                "architecture": architecture,
                "method": "Rules",
                "artifact_type": "rule_json_cart",
                "artifact_path": rule_path.relative_to(REPO).as_posix(),
                "fallback_artifact": metrics["fallback_artifact"],
                "stored_accuracy_pct": float(metrics["test_rule_acc"]),
                "evaluated_accuracy_pct": float(
                    metrics["reevaluated_test_rule_acc"]
                ),
            }
        )
    return pd.DataFrame(rows)


def render_tex(table):
    lines = [
        r"\begin{tabular}{llrrrrrr}",
        r"\toprule",
        r"Arch & Method & Acc. (\%) & Latency & DSP & BRAM & LUT & FF \\",
        r"\midrule",
    ]
    for architecture, group in table.groupby("architecture", sort=False):
        for index, row in enumerate(group.itertuples(index=False)):
            arch = architecture if index == 0 else ""
            lines.append(
                f"{arch} & {row.method} & {row.accuracy_pct:.2f} & "
                f"{int(row.latency_cycles):,} & {int(row.dsp):,} & "
                f"{int(row.bram):,} & {int(row.lut):,} & {int(row.ff):,} \\\\"
            )
        lines.append(r"\midrule")
    lines[-1] = r"\bottomrule"
    lines.append(r"\end{tabular}")
    TEX_PATH.write_text("\n".join(lines) + "\n")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    metrics = evaluate_all(device)
    metrics.to_csv(METRICS_CSV, index=False)

    hardware = pd.read_csv(HARDWARE_CSV)
    table = hardware.merge(
        metrics,
        on=["architecture", "method"],
        how="left",
        validate="one_to_one",
    )
    table["accuracy_source"] = "supplied post-synthesis result"
    table["published_minus_evaluated_pp"] = (
        table["accuracy_pct"] - table["evaluated_accuracy_pct"]
    )
    table.to_csv(VALUES_CSV, index=False)
    render_tex(table)

    print(table[[
        "architecture", "method", "accuracy_pct", "evaluated_accuracy_pct",
        "latency_cycles", "dsp", "bram", "lut", "ff",
    ]].to_string(index=False))
    print(f"Wrote {METRICS_CSV.relative_to(REPO)}")
    print(f"Wrote {VALUES_CSV.relative_to(REPO)}")
    print(f"Wrote {TEX_PATH.relative_to(REPO)}")


if __name__ == "__main__":
    main()
