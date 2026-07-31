"""Evaluate the canonical artifacts and render paper Table VII."""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn

HERE = Path(__file__).resolve().parent
ADULT = HERE.parent
REPO = ADULT.parent
sys.path.insert(0, str(REPO))

from Adult.evaluate_rule_artifacts import evaluate_rule_artifact, load_adult_test_split
from bern2edge.data import adult_ordinal_dataloaders_kfold
from bern2edge.kdtrain import eval_one_epoch
from bern2edge.models import FCModel

CHECKPOINTS = {
    4: "kd_fc_14x4x2_bern_deg3_alpha0.5_T2_lr0.006_wd0.0001_seed9.pth",
    8: "kd_fc_14x8x2_bern_deg3_alpha0.85_T2_lr0.006_wd0.0001_seed7.pth",
    16: "kd_fc_14x16x2_bern_deg3_alpha0.85_T2_lr0.006_wd0.0001_seed8.pth",
    32: "kd_fc_14x32x2_bern_deg3_alpha0.5_T2_lr0.006_wd0.0001_seed7.pth",
    64: "kd_fc_14x64x2_bern_deg3_alpha0_T1_lr0.006_wd0.0001_seed9.pth",
    128: "kd_fc_14x128x2_bern_deg3_alpha0_T1_lr0.006_wd0.0001_seed7.pth",
}
R50 = ADULT / "table9_fallback_ablation/artifacts/tree/rules_float.json"
R29 = ADULT / ("rule_jsons/kd_fc_14x16x8x2_bern_deg3_alpha0.5_T2_"
               "lr0.006_wd0.0001_seed6_sca0.5_ca0.1/rules_float.json")


def evaluate():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    folds, _, _ = adult_ordinal_dataloaders_kfold(
        str(ADULT / "adult_teacher_ordinal.pt"), batch_size=256,
        eval_bs=8192, k=5, seed=6)
    rows = []
    for width, name in CHECKPOINTS.items():
        path = ADULT / "student_model_weights" / name
        ckpt = torch.load(path, map_location=device, weights_only=False)
        model = FCModel(ckpt["arch"], ckpt["degree"],
                        act=ckpt["activation"], last_bern=False).to(device)
        model.load_state_dict(ckpt["state_dict"])
        fold = int(ckpt["seed"]) - 6
        _, acc = eval_one_epoch(model, folds[fold][2], nn.CrossEntropyLoss(), device)
        rows.append({"model": "Bern", "config": f"h={width}",
                     "artifact_path": path.relative_to(REPO).as_posix(),
                     "evaluated_accuracy_fp32_pct": 100.0 * acc})
    X, y = load_adult_test_split()
    for config, path in (("R50", R50), ("R29", R29)):
        result = evaluate_rule_artifact(path, X, y, conflict_strategy="coverage")
        rows.append({"model": "Rules", "config": config,
                     "artifact_path": path.relative_to(REPO).as_posix(),
                     "evaluated_accuracy_fp32_pct": result["test_rule_acc"]})
    return pd.DataFrame(rows)


def main():
    metrics = evaluate()
    metrics.to_csv(HERE / "table7_artifact_metrics.csv", index=False)
    hardware = pd.read_csv(HERE / "hardware_results.csv")
    table = hardware.merge(metrics, on=["model", "config"], validate="one_to_one")
    table.to_csv(HERE / "table7_values.csv", index=False)
    lines = [r"\begin{tabular}{llrrrrrrrr}", r"\toprule",
             r"Model & Config. & LUT & FF & DSP & BRAM18K & Lat. & Pwr. & Acc. & Acc$_{fp}$ \\",
             r"\midrule"]
    for row in table.itertuples(index=False):
        fp = "--" if pd.isna(row.accuracy_fp32_pct) else f"{row.accuracy_fp32_pct:.2f}"
        lines.append(f"{row.model} & {row.config} & {row.lut:,} & {row.ff:,} & "
                     f"{row.dsp} & {row.bram18k} & {row.latency_cycles} & "
                     f"{row.power_mw} & {row.accuracy_pct:.2f} & {fp} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", ""]
    (HERE / "table7.tex").write_text("\n".join(lines))
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()
