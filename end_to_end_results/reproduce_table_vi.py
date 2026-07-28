"""Reproduce Table VI software accuracies and join the supplied HLS results.

The network rows are evaluated from the committed teacher/student checkpoints.
The Adult Rules row is evaluated from its ready JSON/CART artifact. Hardware
columns are copied from ``hardware_results.csv``; synthesis is not rerun.

Run from the repository root:
    python end_to_end_results/reproduce_table_vi.py
"""

from __future__ import annotations

import csv
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
    covertype_dataloaders,
    higgs_small_dataloaders_kfold,
)
from bern2edge.kdtrain import eval_one_epoch  # noqa: E402
from bern2edge.models import AdultTeacherMLP, FCModel, TeacherMLP  # noqa: E402
from Adult.evaluate_rule_artifacts import (  # noqa: E402
    evaluate_rule_artifact,
    load_adult_test_split,
)

HARDWARE_CSV = HERE / "hardware_results.csv"
METRICS_CSV = HERE / "table_vi_artifact_metrics.csv"
VALUES_CSV = HERE / "table_vi_values.csv"
TEX_PATH = HERE / "table_vi.tex"

RULE_JSON = (
    REPO
    / "Adult/rule_jsons"
    / "kd_fc_14x16x8x2_bern_deg3_alpha0.5_T2_lr0.006_wd0.0001_seed6_sca0.5_ca0.1"
    / "rules_float.json"
)

STUDENT_GROUPS = {
    "HIGGS-Small": {
        "architecture": "28x16x8x2",
        "directory": REPO / "higgs_small/student_model_weights",
        "pattern": "kd_fc_28x16x8x2_bern_deg3_alpha0.5_T2_lr0.006_wd0_seed*.pth",
        "seed_base": 42,
        "loader": lambda: higgs_small_dataloaders_kfold(
            n_splits=5, batch_size=512, seed=42
        )[0],
    },
    "Covertype": {
        "architecture": "54x256x128x7",
        "directory": REPO / "cover_type/student_model_weights",
        "pattern": "kd_fc_54x256x128x7_bern_deg5_alpha0.5_T2_lr0.006_wd0.0001_seed*.pth",
        "seed_base": 1000,
        "loader": None,
    },
    "Adult": {
        "architecture": "14x16x2",
        "directory": REPO / "Adult/student_model_weights",
        "pattern": "kd_fc_14x16x2_bern_deg3_alpha0.85_T2_lr0.006_wd0.0001_seed*.pth",
        "seed_base": 6,
        "loader": lambda: adult_ordinal_dataloaders_kfold(
            str(REPO / "Adult/adult_teacher_ordinal.pt"),
            batch_size=256,
            eval_bs=8192,
            k=5,
            seed=6,
        )[0],
    },
}


def accuracy(model, loader, device):
    _, value = eval_one_epoch(model, loader, nn.CrossEntropyLoss(), device)
    return 100.0 * value


def evaluate_students(device):
    rows = []
    cover_folds = None
    for dataset, cfg in STUDENT_GROUPS.items():
        if dataset == "Covertype":
            from bern2edge.data import covertype_dataloaders_kfold

            cover_folds = covertype_dataloaders_kfold(
                batch_size=1024, k=5, seed=42, test_size=0.15
            )[0]
            folds = cover_folds
        else:
            folds = cfg["loader"]()
        paths = sorted(cfg["directory"].glob(cfg["pattern"]))
        if len(paths) != 5:
            raise RuntimeError(f"{dataset}: expected five checkpoints, found {len(paths)}")
        for path in paths:
            ckpt = torch.load(path, map_location=device, weights_only=False)
            fold = int(ckpt["seed"]) - cfg["seed_base"]
            model = FCModel(
                layer_sizes=ckpt["arch"],
                degree=ckpt["degree"],
                act=ckpt["activation"],
                last_bern=False,
            ).to(device)
            model.load_state_dict(ckpt["state_dict"])
            measured = accuracy(model, folds[fold][2], device)
            stored = float(ckpt["test_acc"])
            if abs(measured - stored) > 0.05:
                raise RuntimeError(
                    f"{path.name}: measured {measured:.6f}, stored {stored:.6f}"
                )
            rows.append(
                {
                    "dataset": dataset,
                    "method": "Bern2Edge (LUT)",
                    "architecture": cfg["architecture"],
                    "artifact_type": "student_checkpoint",
                    "artifact_path": path.relative_to(REPO).as_posix(),
                    "fold": fold,
                    "stored_accuracy_pct": stored,
                    "evaluated_accuracy_pct": measured,
                }
            )
    return rows


def evaluate_teachers(device):
    rows = []

    # HIGGS teacher uses the fixed held-out test set shared by all five folds.
    higgs_folds, d_in, n_classes = higgs_small_dataloaders_kfold(
        n_splits=5, batch_size=512, seed=42
    )
    path = REPO / "higgs_small/higgs_small_teacher.pth"
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model = TeacherMLP(d_in, [231, 121], 0.0, n_classes).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    rows.append(("HIGGS-Small", "28x231x121x2", path,
                 accuracy(model, higgs_folds[0][2], device)))

    # Covertype teacher uses its original fixed train/test preprocessing.
    _, _, test_loader, d_in, n_classes = covertype_dataloaders(
        batch_size=1024, test_size=0.15, seed=42
    )
    path = REPO / "cover_type/covertype_teacher_weights.pth"
    model = TeacherMLP(d_in, [1024, 1024, 512], 0.1, n_classes).to(device)
    model.load_state_dict(torch.load(path, map_location=device, weights_only=False))
    rows.append(("Covertype", "54x1024x1024x512x7", path,
                 accuracy(model, test_loader, device)))

    # Adult teacher checkpoint preserves its exact held-out split/preprocessor.
    adult_folds, _, n_classes = adult_ordinal_dataloaders_kfold(
        str(REPO / "Adult/adult_teacher_ordinal.pt"),
        batch_size=256,
        eval_bs=8192,
        k=5,
        seed=6,
    )
    path = REPO / "Adult/adult_teacher_ordinal.pt"
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model = AdultTeacherMLP(
        ckpt["input_dim"], ckpt["d_layers"], ckpt["dropout"], n_classes
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    rows.append(("Adult", "14x42x503x503x503x111x2", path,
                 accuracy(model, adult_folds[0][2], device)))

    return [
        {
            "dataset": dataset,
            "method": "Teacher (W8A8)",
            "architecture": architecture,
            "artifact_type": "teacher_checkpoint",
            "artifact_path": path.relative_to(REPO).as_posix(),
            "fold": "",
            "stored_accuracy_pct": "",
            "evaluated_accuracy_pct": measured,
        }
        for dataset, architecture, path, measured in rows
    ]


def evaluate_adult_rules():
    X_test, y_test = load_adult_test_split()
    result = evaluate_rule_artifact(
        RULE_JSON,
        X_test,
        y_test,
        conflict_strategy="coverage",
    )
    measured = float(result["test_rule_acc"])
    return {
        "dataset": "Adult",
        "method": "Bern2Edge (Rules)",
        "architecture": "14x16x8x2",
        "artifact_type": "rule_json_cart",
        "artifact_path": RULE_JSON.relative_to(REPO).as_posix(),
        "fold": 0,
        "stored_accuracy_pct": measured,
        "evaluated_accuracy_pct": measured,
    }


def render_tex(table):
    lines = [
        r"\begin{tabular}{lllrrrrr}",
        r"\toprule",
        r"Dataset & Method & Architecture & Acc. (\%) & Latency & DSPs & BRAMs & URAMs \\",
        r"\midrule",
    ]
    for dataset, group in table.groupby("dataset", sort=False):
        for index, row in enumerate(group.itertuples(index=False)):
            name = dataset if index == 0 else ""
            uram = "--" if pd.isna(row.uram) else str(int(row.uram))
            acc = f"{row.accuracy_pct:.2f}" if row.method.endswith("(Rules)") else f"{row.accuracy_pct:.1f}"
            def resource(value, reduction):
                if pd.isna(reduction):
                    return f"{int(value):,}"
                arrow = r"\downarrow" if reduction >= 0 else r"\uparrow"
                return f"{int(value):,} ({arrow} {abs(reduction):.1f}\\%)"
            lines.append(
                f"{name} & {row.method} & {row.architecture} & {acc} & "
                f"{resource(row.latency_cycles, row.latency_reduction_pct)} & "
                f"{resource(row.dsp, row.dsp_reduction_pct)} & "
                f"{resource(row.bram, row.bram_reduction_pct)} & {uram} \\\\"
            )
        lines.append(r"\midrule")
    lines[-1] = r"\bottomrule"
    lines.append(r"\end{tabular}")
    TEX_PATH.write_text("\n".join(lines) + "\n")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows = evaluate_students(device)
    rows.extend(evaluate_teachers(device))
    rows.append(evaluate_adult_rules())
    metrics = pd.DataFrame(rows)
    metrics.to_csv(METRICS_CSV, index=False)

    summaries = (
        metrics.groupby(
            ["dataset", "method", "architecture"], as_index=False, sort=False
        )
        .agg(
            evaluated_accuracy_pct=("evaluated_accuracy_pct", "mean"),
            n_artifacts=("artifact_path", "count"),
            artifact_paths=("artifact_path", lambda values: ";".join(values)),
        )
    )
    hardware = pd.read_csv(HARDWARE_CSV)
    table = hardware.merge(
        summaries,
        on=["dataset", "method", "architecture"],
        validate="one_to_one",
    )
    teacher = table.method.eq("Teacher (W8A8)")
    rules = table.method.eq("Bern2Edge (Rules)")
    table["accuracy_source"] = "evaluated checkpoint mean"
    table.loc[teacher, "accuracy_source"] = "evaluated teacher checkpoint"
    table.loc[rules, "accuracy_source"] = "evaluated ready rule/CART artifact"
    table["published_minus_evaluated_pp"] = (
        table["accuracy_pct"] - table["evaluated_accuracy_pct"]
    )
    for column in ("latency_cycles", "dsp", "bram", "uram"):
        output = column.replace("_cycles", "") + "_reduction_pct"
        table[output] = float("nan")
        for dataset, group in table.groupby("dataset"):
            baseline = group.loc[group.method.eq("Teacher (W8A8)"), column].iloc[0]
            mask = table.dataset.eq(dataset) & ~table.method.eq("Teacher (W8A8)")
            table.loc[mask, output] = 100.0 * (
                baseline - table.loc[mask, column]
            ) / baseline
    table.to_csv(VALUES_CSV, index=False)
    render_tex(table)

    print(table[[
        "dataset", "method", "architecture", "accuracy_pct", "evaluated_accuracy_pct",
        "latency_cycles", "dsp", "bram", "uram"
    ]].to_string(index=False))
    print(f"Wrote {METRICS_CSV.relative_to(REPO)}")
    print(f"Wrote {VALUES_CSV.relative_to(REPO)}")
    print(f"Wrote {TEX_PATH.relative_to(REPO)}")


if __name__ == "__main__":
    main()
