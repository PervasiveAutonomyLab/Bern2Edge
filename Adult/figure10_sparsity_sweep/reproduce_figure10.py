#!/usr/bin/env python3
"""Re-evaluate the shipped k-sweep artifacts and redraw Figure 10.

Run from the repository root:

    python Adult/figure10_sparsity_sweep/reproduce_figure10.py

This writes the complete evaluation CSV, verifies all 26 plotted coordinates,
and produces PDF and PNG plots. Pass ``--plot-only`` to redraw directly from
the committed plot-value CSV.
"""

import argparse
import csv
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO))

from Adult.evaluate_rule_artifacts import (  # noqa: E402
    evaluate_rule_artifact,
    load_adult_test_split,
)


ARTIFACTS = HERE / "artifacts"
BRAM_CSV = HERE / "bram_measurements.csv"
VALUES_CSV = HERE / "figure10_values.csv"
METRICS_CSV = HERE / "figure10_artifact_metrics.csv"
NAME_RE = re.compile(r"_k(?P<k>[1-9]|1[0-3])\.json$")
METRIC_FIELDS = (
    "sparsity_k",
    "arch",
    "same_cov_alpha",
    "conflict_alpha",
    "n_rules",
    "test_covered_pct",
    "test_covered_rule_acc",
    "test_rule_acc",
    "n_conflicts",
    "avg_conditions",
    "rule_json",
)


def artifact_records():
    records = []
    for path in ARTIFACTS.glob("*.json"):
        match = NAME_RE.search(path.name)
        if not match:
            raise RuntimeError(f"unexpected artifact filename: {path.name}")
        records.append((int(match.group("k")), path))
    records.sort()
    if [k for k, _ in records] != list(range(1, 14)):
        raise RuntimeError("expected exactly one artifact for each k=1,...,13")
    return records


def read_csv(path):
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def evaluate_all():
    X_test, y_test = load_adult_test_split()
    rows = []
    for k, path in artifact_records():
        result = evaluate_rule_artifact(
            path,
            X_test,
            y_test,
            conflict_strategy="max_purity",
            validate_stored=False,
        )
        rows.append({
            "sparsity_k": k,
            **{field: result[field] for field in METRIC_FIELDS
               if field != "sparsity_k"},
        })
        print(f"Evaluated k={k}: total accuracy={result['test_rule_acc']:.2f}%")

    with METRICS_CSV.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=METRIC_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    return rows


def build_values(metrics):
    bram = {int(row["sparsity_k"]): int(row["bram"]) for row in read_csv(BRAM_CSV)}
    if sorted(bram) != list(range(1, 14)):
        raise RuntimeError("BRAM CSV must contain exactly k=1,...,13")
    return [{
        "sparsity_k": str(row["sparsity_k"]),
        "bram": str(bram[row["sparsity_k"]]),
        "total_accuracy_pct": f"{float(row['test_rule_acc']):.2f}",
    } for row in metrics]


def verify_values(actual):
    expected = read_csv(VALUES_CSV)
    if actual != expected:
        for got, want in zip(actual, expected):
            if got != want:
                raise RuntimeError(f"Figure 10 mismatch:\nactual={got}\nexpected={want}")
        raise RuntimeError("Figure 10 row-count mismatch")
    print("Verified all 26 published Figure 10 coordinates exactly.")


def plot(rows):
    k = [int(row["sparsity_k"]) for row in rows]
    bram = [int(row["bram"]) for row in rows]
    accuracy = [float(row["total_accuracy_pct"]) for row in rows]

    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 9,
        "axes.labelsize": 10,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "lines.linewidth": 1.1,
        "lines.markersize": 5,
    })
    fig, ax = plt.subplots(figsize=(3.5, 2.0))
    ax.plot(k, bram, "-", marker="o", color="#0000b8")
    ax.set_xlabel(r"sparsity $k$")
    ax.set_ylabel("BRAM", color="#0000b8")
    ax.tick_params(axis="y", labelcolor="#0000b8")
    ax.set_xticks(k)
    ax.set_xlim(0.5, 13.5)
    ax.set_ylim(0, 16)
    ax.grid(True, color="0.85", linewidth=0.5)

    right = ax.twinx()
    right.plot(k, accuracy, "-", marker="s", color="#b80000")
    right.set_ylabel("Total Acc. (%)", color="#b80000")
    right.tick_params(axis="y", labelcolor="#b80000")
    right.set_ylim(76, 83)
    right.grid(False)

    fig.tight_layout()
    for extension in ("pdf", "png"):
        fig.savefig(HERE / f"figure10_sparsity_sweep.{extension}", dpi=300)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--plot-only",
        action="store_true",
        help="redraw from the committed Figure 10 value CSV",
    )
    args = parser.parse_args()

    if args.plot_only:
        values = read_csv(VALUES_CSV)
    else:
        values = build_values(evaluate_all())
        verify_values(values)
    plot(values)
    print(f"Wrote Figure 10 outputs under {HERE.relative_to(REPO)}/")


if __name__ == "__main__":
    main()
