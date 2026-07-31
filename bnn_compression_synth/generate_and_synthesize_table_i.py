"""Regenerate and verify all Table I Bernstein and ReLU HLS metrics.

One canonical fold is synthesized per dataset/architecture/activation row.
Run from the repository root after loading the required Vitis environment:

    python bnn_compression_synth/generate_and_synthesize_table_i.py
"""

from __future__ import annotations

import argparse
import csv
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(REPO))

from hls.bern2hls.core.collect import run_collect
from hls.bern2hls.core.synth import run_synth
from hls.bern2hls.fc.compile import compile_dataset


DATASET_KEYS = {"Adult": "adult", "HIGGS-Small": "higgs", "Covertype": "covertype"}
DEFAULT_BUILD = REPO / "build" / "table_i_hls"
DEFAULT_CSV = DEFAULT_BUILD / "fresh_hls_results.csv"


def canonical_checkpoints():
    with (HERE / "table_i_checkpoint_results.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    selected = [row for row in rows if int(row["fold_id"]) == 0]
    if len(selected) != 18:
        raise RuntimeError(f"Expected 18 canonical Table I checkpoints, found {len(selected)}")
    return selected


def compile_all(build_dir: Path):
    roots = {}
    for row in canonical_checkpoints():
        dataset = DATASET_KEYS[row["dataset"]]
        root = build_dir / dataset
        checkpoint = REPO / row["checkpoint_path"]
        compile_dataset(dataset, out_dir=str(root), pth_inputs=[str(checkpoint)])
        roots[dataset] = root
    return [roots[name] for name in ("adult", "higgs", "covertype")]


def verify_metrics(fresh_csv: Path):
    with (HERE / "table_i_hls_results.csv").open(newline="") as handle:
        expected = list(csv.DictReader(handle))
    with fresh_csv.open(newline="") as handle:
        actual = list(csv.DictReader(handle))
    actual_by_key = {
        (row["dataset"], row["arch"], row["act"]): row for row in actual
    }
    mismatches = []
    for row in expected:
        key = (DATASET_KEYS[row["dataset"]], row["architecture"], row["activation"])
        got = actual_by_key.get(key)
        if got is None:
            mismatches.append(f"{key}: missing fresh synthesis row")
            continue
        comparisons = {
            "latency_cycles": row["latency_cycles"],
            "DSP": row["dsp"],
            "BRAM_18K": row["bram_18k"],
            "LUT": row["lut"],
        }
        for field, value in comparisons.items():
            if str(got[field]) != str(value):
                mismatches.append(f"{key} {field}: fresh={got[field]} paper={value}")
    if len(actual) != 18:
        mismatches.append(f"Fresh CSV contains {len(actual)} rows; expected 18")
    if mismatches:
        print("\nTABLE I HARDWARE VERIFICATION: FAIL")
        for mismatch in mismatches:
            print(f"  {mismatch}")
        return False
    print("\nTABLE I HARDWARE VERIFICATION: PASS — all 18 rows match the paper")
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-dir", type=Path, default=DEFAULT_BUILD)
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--vitis-hls", default="vitis-run")
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--generate-only", action="store_true")
    parser.add_argument("--skip-csim", action="store_true")
    parser.add_argument("--clean", action="store_true")
    args = parser.parse_args()

    if args.clean and args.build_dir.exists():
        shutil.rmtree(args.build_dir)
    args.build_dir.mkdir(parents=True, exist_ok=True)
    roots = compile_all(args.build_dir)
    print(f"\nGenerated 18 Table I projects under {args.build_dir}")
    if args.generate_only:
        return 0
    if shutil.which(args.vitis_hls.split()[0]) is None:
        raise SystemExit(
            f"Cannot find {args.vitis_hls!r}. Load the required Vitis environment "
            "or pass --vitis-hls /path/to/vitis-run."
        )
    if not args.skip_csim:
        rc = run_synth([str(root) for root in roots], csim=True, jobs=args.jobs,
                       vitis_hls=args.vitis_hls)
        if rc:
            return rc
    rc = run_synth([str(root) for root in roots], jobs=args.jobs,
                   vitis_hls=args.vitis_hls)
    if rc:
        return rc
    args.csv.parent.mkdir(parents=True, exist_ok=True)
    rc = run_collect([str(root) for root in roots], csv_path=str(args.csv))
    return rc if rc else (0 if verify_metrics(args.csv) else 1)


if __name__ == "__main__":
    raise SystemExit(main())
