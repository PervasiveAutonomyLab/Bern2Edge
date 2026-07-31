"""Generate and optionally synthesize the four Table IX fallback variants."""

from __future__ import annotations

import argparse
import csv
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO))

from hls.bern2hls.core.collect import run_collect
from hls.bern2hls.core.synth import run_synth
from hls.bern2hls.rules.compile import compile_rule_model

DEFAULT_BUILD = REPO / "build" / "table_ix_fallback_hls"
DEFAULT_CSV = DEFAULT_BUILD / "fresh_hls_results.csv"
VARIANTS = {
    "lr": None,
    "network": "fallback_network.pth",
    "small_nn": "fallback_net.pt",
    "tree": "fallback_tree_int.npz",
}


def generate(build_dir):
    for variant, asset_name in VARIANTS.items():
        source = HERE / "artifacts" / variant
        asset = source / asset_name if asset_name else None
        for group, fallback_only in (("full", False), ("fallback_only", True)):
            compile_rule_model(
                source / "rules_int8.json",
                build_dir / group / variant,
                fallback_kind=variant,
                fallback_asset=asset,
                name=variant,
                fallback_only=fallback_only,
            )


def verify_metrics(fresh_csv):
    with (HERE / "hardware_results.csv").open(newline="") as stream:
        expected = {row["variant"]: row for row in csv.DictReader(stream)}
    with fresh_csv.open(newline="") as stream:
        rows = list(csv.DictReader(stream))

    actual = {}
    for row in rows:
        group = row["dataset"]
        actual[(group, Path(row["model"]).name)] = row

    mappings = {
        "full": {
            "tot_LUT": "LUT",
            "tot_FF": "FF",
            "tot_DSP": "DSP",
            "tot_BRAM": "BRAM_18K",
            "tot_lat": "latency_cycles",
        },
        "fallback_only": {
            "fb_LUT": "LUT",
            "fb_FF": "FF",
            "fb_DSP": "DSP",
            "fb_BRAM": "BRAM_18K",
            "fb_lat": "latency_cycles",
        },
    }
    mismatches = []
    for variant, paper in expected.items():
        for group, fields in mappings.items():
            fresh = actual.get((group, variant))
            if fresh is None:
                mismatches.append(f"{group}/{variant}: missing fresh synthesis row")
                continue
            for paper_key, fresh_key in fields.items():
                if paper[paper_key] != fresh[fresh_key]:
                    mismatches.append(
                        f"{group}/{variant} {fresh_key}: "
                        f"fresh={fresh[fresh_key]} paper={paper[paper_key]}"
                    )
    if mismatches:
        print("\nTABLE IX HARDWARE VERIFICATION: FAIL")
        for mismatch in mismatches:
            print(f"  {mismatch}")
        return False
    print("\nTABLE IX HARDWARE VERIFICATION: PASS")
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-dir", type=Path, default=DEFAULT_BUILD)
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--vitis-hls", default="vitis-run")
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--generate-only", action="store_true")
    parser.add_argument("--clean", action="store_true")
    args = parser.parse_args()

    if args.clean and args.build_dir.exists():
        shutil.rmtree(args.build_dir)
    args.build_dir.mkdir(parents=True, exist_ok=True)
    generate(args.build_dir)
    print("Generated four full and four fallback-only Table IX projects "
          f"under {args.build_dir}")
    if args.generate_only:
        return 0
    if shutil.which(args.vitis_hls.split()[0]) is None:
        raise SystemExit(
            f"Cannot find {args.vitis_hls!r}. Load the required Vitis environment."
        )
    roots = [args.build_dir / "full", args.build_dir / "fallback_only"]
    rc = run_synth([str(root) for root in roots], jobs=args.jobs,
                   vitis_hls=args.vitis_hls)
    if rc:
        return rc
    rc = run_collect([str(root) for root in roots], csv_path=str(args.csv))
    return rc if rc else (0 if verify_metrics(args.csv) else 1)


if __name__ == "__main__":
    raise SystemExit(main())
