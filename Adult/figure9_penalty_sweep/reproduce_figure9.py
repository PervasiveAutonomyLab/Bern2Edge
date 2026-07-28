#!/usr/bin/env python3
"""Re-evaluate the shipped Figure 9 rules/CART artifacts and redraw the figure.

Run from the repository root:

    python Adult/figure9_penalty_sweep/reproduce_figure9.py

The script writes a per-architecture metrics CSV, averages the two published
penalty slices across the five architectures, verifies the averages against the
committed paper values, and writes PDF/PNG plots.
"""

import argparse
import csv
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO))

from Adult.evaluate_rule_artifacts import (  # noqa: E402
    evaluate_rule_artifact,
    load_adult_test_split,
)


ARTIFACTS = HERE / "artifacts"
EXPECTED_CSV = HERE / "figure9_values.csv"
RAW_CSV = HERE / "figure9_metrics_by_arch.csv"
ARCHITECTURES = (
    "14x16x2",
    "14x32x2",
    "14x128x2",
    "14x16x8x2",
    "14x32x16x2",
)
NAME_RE = re.compile(
    r"^kd_fc_(?P<arch>.+?)_bern_deg3_.+_sca(?P<sca>[0-9.]+)_ca(?P<ca>[0-9.]+)\.json$"
)
RAW_FIELDS = (
    "arch",
    "same_cov_alpha",
    "conflict_alpha",
    "n_rules",
    "test_covered_pct",
    "test_covered_rule_acc",
    "test_rule_acc",
    "reevaluated_test_covered_pct",
    "reevaluated_test_rule_acc",
    "n_conflicts",
    "avg_conditions",
    "rule_json",
    "cart_npz",
)


def artifact_records():
    records = []
    for path in sorted(ARTIFACTS.glob("*.json")):
        match = NAME_RE.match(path.name)
        if not match:
            raise RuntimeError(f"unexpected artifact filename: {path.name}")
        records.append((
            match.group("arch"),
            float(match.group("sca")),
            float(match.group("ca")),
            path,
        ))
    expected = 5 * (11 + 11 - 1)
    if len(records) != expected:
        raise RuntimeError(f"expected {expected} JSON artifacts, found {len(records)}")
    found_arches = {r[0] for r in records}
    if found_arches != set(ARCHITECTURES):
        raise RuntimeError(f"architecture mismatch: {sorted(found_arches)}")
    return records


def evaluate_all():
    X_test, y_test = load_adult_test_split()
    rows = []
    for index, (arch, sca, ca, path) in enumerate(artifact_records(), start=1):
        evaluated = evaluate_rule_artifact(path, X_test, y_test)
        rows.append({
            "arch": arch,
            "same_cov_alpha": f"{sca:.1f}",
            "conflict_alpha": f"{ca:.1f}",
            **{key: evaluated[key] for key in RAW_FIELDS
               if key not in ("arch", "same_cov_alpha", "conflict_alpha")},
        })
        if index % 20 == 0 or index == 105:
            print(f"Evaluated {index}/105 artifacts", flush=True)

    with RAW_CSV.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=RAW_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    return rows


def average_slices(rows):
    output = []
    slices = (
        ("same_cov", "same_cov_alpha", "conflict_alpha", 0.5),
        ("conflict", "conflict_alpha", "same_cov_alpha", 0.5),
    )
    for sweep, x_key, fixed_key, fixed_value in slices:
        for x in (i / 10 for i in range(11)):
            selected = [
                row for row in rows
                if float(row[x_key]) == x and float(row[fixed_key]) == fixed_value
            ]
            if len(selected) != len(ARCHITECTURES):
                raise RuntimeError(f"{sweep}={x:.1f}: expected 5 rows, found {len(selected)}")
            output.append({
                "sweep": sweep,
                "parameter": f"{x:.1f}",
                "mean_test_covered_pct": f"{np.mean([r['test_covered_pct'] for r in selected]):.3f}",
                "mean_test_covered_rule_acc": f"{np.mean([r['test_covered_rule_acc'] for r in selected]):.3f}",
                "n_architectures": str(len(selected)),
            })
    return output


def read_expected():
    with EXPECTED_CSV.open(newline="") as f:
        return list(csv.DictReader(f))


def verify_values(actual):
    expected = read_expected()
    if actual != expected:
        for got, want in zip(actual, expected):
            if got != want:
                raise RuntimeError(f"Figure 9 mismatch:\nactual={got}\nexpected={want}")
        raise RuntimeError("Figure 9 row-count mismatch")
    print("Verified all 44 published Figure 9 coordinates exactly.")


def plot_panel(rows, sweep, output_stem):
    selected = [r for r in rows if r["sweep"] == sweep]
    x = np.asarray([float(r["parameter"]) for r in selected])
    cov = np.asarray([float(r["mean_test_covered_pct"]) for r in selected])
    cov_acc = np.asarray([float(r["mean_test_covered_rule_acc"]) for r in selected])

    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 9,
        "axes.labelsize": 10,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "lines.linewidth": 1.1,
        "lines.markersize": 5,
    })
    fig, ax = plt.subplots(figsize=(3.15, 1.65))
    ax.plot(x, cov, "-", marker="o", color="#0000b8")
    ax.set_xlabel(r"$\alpha_{sc}$" if sweep == "same_cov" else r"$\alpha_{conf}$")
    ax.set_ylabel("Cov. (%)", color="#0000b8")
    ax.tick_params(axis="y", labelcolor="#0000b8")
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(68, 102)
    ax.set_xticks(np.arange(0, 1.01, 0.2))
    ax.grid(True, color="0.85", linewidth=0.5)

    right = ax.twinx()
    right.plot(x, cov_acc, "-", marker="s", color="#b80000")
    right.set_ylabel("Cov. Acc. (%)", color="#b80000")
    right.tick_params(axis="y", labelcolor="#b80000")
    right.set_ylim(78.5, 85)
    right.grid(False)

    fig.tight_layout()
    for extension in ("pdf", "png"):
        fig.savefig(HERE / f"{output_stem}.{extension}", dpi=300)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--plot-only",
        action="store_true",
        help="redraw plots from the committed averaged CSV without evaluating artifacts",
    )
    args = parser.parse_args()

    if args.plot_only:
        averages = read_expected()
    else:
        rows = evaluate_all()
        averages = average_slices(rows)
        verify_values(averages)

    plot_panel(averages, "same_cov", "figure9_same_cov_sweep")
    plot_panel(averages, "conflict", "figure9_conflict_sweep")
    print(f"Wrote Figure 9 outputs under {HERE.relative_to(REPO)}/")


if __name__ == "__main__":
    main()
