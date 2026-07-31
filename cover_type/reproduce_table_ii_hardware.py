"""Regenerate and verify the ten Covertype HLS designs used by Table II."""

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


DEFAULT_BUILD = REPO / "build" / "table_ii_hls"
DEFAULT_CSV = DEFAULT_BUILD / "fresh_hls_results.csv"


def canonical_checkpoints():
    with (HERE / "table_ii_checkpoint_results.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    selected = [row for row in rows if int(row["fold_id"]) == 0]
    if len(selected) != 10:
        raise RuntimeError(f"Expected 10 canonical Table II checkpoints, found {len(selected)}")
    return selected


def verify_metrics(fresh_csv: Path):
    with (HERE / "covertype_hls_results.csv").open(newline="") as handle:
        normalized = [
            {key.strip(): value.strip() for key, value in row.items()}
            for row in csv.DictReader(handle, skipinitialspace=True)
        ]
        expected = {row["model"]: row for row in normalized}
    with fresh_csv.open(newline="") as handle:
        actual = {row["model"]: row for row in csv.DictReader(handle)}
    models = {
        "bern_d3_54x8x7", "relu_54x8x7",
        "bern_d3_54x32x7", "relu_54x32x7",
        "bern_d3_54x64x32x7", "relu_54x64x32x7",
        "bern_d5_54x128x64x7", "relu_54x128x64x7",
        "bern_d5_54x256x128x7", "relu_54x256x128x7",
    }
    mismatches = []
    for model in sorted(models):
        if model not in actual:
            mismatches.append(f"{model}: missing fresh synthesis row")
            continue
        for field in ("latency_cycles", "BRAM_18K", "DSP", "FF", "LUT"):
            if actual[model][field] != expected[model][field]:
                mismatches.append(
                    f"{model} {field}: fresh={actual[model][field]} "
                    f"paper={expected[model][field]}"
                )
    if mismatches:
        print("\nTABLE II HARDWARE VERIFICATION: FAIL")
        for mismatch in mismatches:
            print(f"  {mismatch}")
        return False
    print("\nTABLE II HARDWARE VERIFICATION: PASS — all ten designs match")
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
    for row in canonical_checkpoints():
        compile_dataset(
            "covertype",
            out_dir=str(args.build_dir),
            pth_inputs=[str(REPO / row["checkpoint_path"])],
        )
    print(f"\nGenerated ten Table II projects under {args.build_dir}")
    if args.generate_only:
        return 0
    if shutil.which(args.vitis_hls.split()[0]) is None:
        raise SystemExit(
            f"Cannot find {args.vitis_hls!r}. Load the required Vitis environment "
            "or pass --vitis-hls /path/to/vitis-run."
        )
    if not args.skip_csim:
        rc = run_synth([str(args.build_dir)], csim=True, jobs=args.jobs,
                       vitis_hls=args.vitis_hls)
        if rc:
            return rc
    rc = run_synth([str(args.build_dir)], jobs=args.jobs, vitis_hls=args.vitis_hls)
    if rc:
        return rc
    args.csv.parent.mkdir(parents=True, exist_ok=True)
    rc = run_collect([str(args.build_dir)], csv_path=str(args.csv))
    return rc if rc else (0 if verify_metrics(args.csv) else 1)


if __name__ == "__main__":
    raise SystemExit(main())
