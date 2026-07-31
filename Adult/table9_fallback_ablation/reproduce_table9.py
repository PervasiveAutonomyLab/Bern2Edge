#!/usr/bin/env python3
"""Evaluate the four ready fallbacks and render Adult Table IX.

Run from the repository root:

    python Adult/table9_fallback_ablation/reproduce_table9.py

Full accuracy and hardware columns are copied from the committed synthesis
summary. Fallback accuracy and fidelity are recomputed only on held-out test
samples not covered by any rule.
"""

import csv
import sys
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO))

from Adult.evaluate_rule_artifacts import (  # noqa: E402
    _network_predict,
    evaluate_rule_artifact,
    load_adult_test_split,
)


ARTIFACTS = HERE / "artifacts"
HARDWARE_CSV = HERE / "hardware_results.csv"
METRICS_CSV = HERE / "table9_artifact_metrics.csv"
TABLE_CSV = HERE / "table9_values.csv"
TABLE_TEX = HERE / "table9.tex"
VARIANTS = (
    ("lr", "Linear Regression", "lr"),
    ("network", "Full BNN", "network"),
    ("small_nn", "Small BNN (int8)", "small_nn"),
    ("tree", "CART", "tree"),
)
METRIC_FIELDS = (
    "fallback",
    "n_rules",
    "test_covered_pct",
    "test_uncovered_pct",
    "rules_plus_fallback_acc",
    "rules_plus_fallback_fidelity",
    "fallback_acc_uncovered",
    "fallback_fidelity_uncovered",
    "rule_json",
    "fallback_artifact",
)
TABLE_FIELDS = (
    "fallback",
    "accuracy_pct",
    "fallback_acc_uncovered_pct",
    "fallback_fidelity_uncovered_pct",
    "lut",
    "ff",
    "dsp",
    "latency",
)


def read_hardware():
    with HARDWARE_CSV.open(newline="") as stream:
        return {row["variant"]: row for row in csv.DictReader(stream)}


def one_decimal(value):
    """Paper-style decimal rounding (half up, not Python's ties-to-even)."""
    return str(Decimal(str(value)).quantize(Decimal("0.1"), rounding=ROUND_HALF_UP))


def evaluate():
    X_test, y_test = load_adult_test_split()
    reference = ARTIFACTS / "network" / "fallback_network.pth"
    y_model = _network_predict(reference, X_test)
    hardware = read_hardware()
    metrics_rows = []
    table_rows = []

    for directory, label, synthesis_variant in VARIANTS:
        path = ARTIFACTS / directory / "rules_float.json"
        result = evaluate_rule_artifact(
            path,
            X_test,
            y_test,
            y_model=y_model,
        )
        metrics_rows.append({
            "fallback": label,
            "n_rules": result["n_rules"],
            "test_covered_pct": result["test_covered_pct"],
            "test_uncovered_pct": result["test_uncovered_pct"],
            "rules_plus_fallback_acc": result["test_rule_acc"],
            "rules_plus_fallback_fidelity": result["test_fidelity"],
            "fallback_acc_uncovered": result["fallback_acc_uncovered"],
            "fallback_fidelity_uncovered": result["fallback_fidelity_uncovered"],
            "rule_json": result["rule_json"],
            "fallback_artifact": result["fallback_artifact"],
        })
        hw = hardware[synthesis_variant]
        table_rows.append({
            "fallback": label,
            "accuracy_pct": f"{float(hw['HLS_acc_%']):.2f}",
            "fallback_acc_uncovered_pct": (
                one_decimal(result["fallback_acc_uncovered"])
            ),
            "fallback_fidelity_uncovered_pct": (
                one_decimal(result["fallback_fidelity_uncovered"])
            ),
            "lut": hw["fb_LUT"],
            "ff": hw["fb_FF"],
            "dsp": hw["fb_DSP"],
            "latency": hw["fb_lat"],
        })

    write_csv(METRICS_CSV, METRIC_FIELDS, metrics_rows)
    write_csv(TABLE_CSV, TABLE_FIELDS, table_rows)
    return table_rows


def write_csv(path, fields, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def render_latex(rows):
    best_acc = max(float(row["accuracy_pct"]) for row in rows)
    best_fid = max(float(row["fallback_fidelity_uncovered_pct"]) for row in rows)
    minima = {
        key: min(int(row[key]) for row in rows)
        for key in ("lut", "ff", "dsp", "latency")
    }

    def bold_if(value, condition):
        return rf"\textbf{{{value}}}" if condition else str(value)

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Fallback strategy ablation on Adult "
        r"($h=32$, $\alpha_{\mathrm{sc}}=0.5$, "
        r"$\alpha_{\mathrm{conf}}=0.1$). Resource columns report the fallback "
        r"component only. $\mathrm{Acc}_{\mathrm{unc}}$ and "
        r"$\mathrm{Fid}_{\mathrm{unc}}$ are evaluated on uncovered held-out "
        r"test samples.}",
        r"\label{tab:fallback-ablation}",
        r"\begin{tabular}{lrrrrrrr}",
        r"\toprule",
        r"Fallback & Acc. & $\mathrm{Acc}_{\mathrm{unc}}$ & "
        r"$\mathrm{Fid}_{\mathrm{unc}}$ & LUT & FF & DSP & Lat. \\",
        r"\midrule",
    ]
    for row in rows:
        values = [
            row["fallback"],
            bold_if(row["accuracy_pct"], float(row["accuracy_pct"]) == best_acc),
            row["fallback_acc_uncovered_pct"],
            bold_if(
                row["fallback_fidelity_uncovered_pct"],
                float(row["fallback_fidelity_uncovered_pct"]) == best_fid,
            ),
            bold_if(row["lut"], int(row["lut"]) == minima["lut"]),
            bold_if(row["ff"], int(row["ff"]) == minima["ff"]),
            bold_if(row["dsp"], int(row["dsp"]) == minima["dsp"]),
            bold_if(row["latency"], int(row["latency"]) == minima["latency"]),
        ]
        lines.append(" & ".join(values) + r" \\")
    lines.extend((r"\bottomrule", r"\end{tabular}", r"\end{table}", ""))
    TABLE_TEX.write_text("\n".join(lines))


def main():
    rows = evaluate()
    render_latex(rows)
    print(f"Evaluated four fallback artifacts -> {METRICS_CSV.relative_to(REPO)}")
    print(f"Wrote table CSV and LaTeX under {HERE.relative_to(REPO)}/")
    for row in rows:
        print(
            f"{row['fallback']}: HLS acc={row['accuracy_pct']}%, "
            f"unc acc={row['fallback_acc_uncovered_pct']}%, "
            f"unc fid={row['fallback_fidelity_uncovered_pct']}%"
        )


if __name__ == "__main__":
    main()
