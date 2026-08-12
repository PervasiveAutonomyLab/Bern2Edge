"""Generate and optionally synthesize the five Table IV rule classifiers.

Run from the repository root:

    python Adult/table4_rule_network_hardware/generate_and_synthesize_table_iv.py
"""

from __future__ import annotations

import argparse
import csv
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ADULT = HERE.parent
REPO = ADULT.parent
sys.path.insert(0, str(REPO))

from hls.bern2hls.core.collect import run_collect
from hls.bern2hls.core.compare import report
from hls.bern2hls.core.synth import run_synth
from hls.bern2hls.rules.compile import compile_rule_model

ARCHITECTURES = (
    "14x16x2",
    "14x32x2",
    "14x128x2",
    "14x16x8x2",
    "14x32x16x2",
)
DEFAULT_BUILD = REPO / "build" / "table_iv_rule_hls"
DEFAULT_CSV = DEFAULT_BUILD / "fresh_hls_results.csv"
# The packaged Adult held-out fold; the generated classifiers need it for csim.
TEST_VECTORS = REPO / "hls" / "test_vectors" / "adult9045"


def rule_dir(architecture):
    stem = (
        f"kd_fc_{architecture}_bern_deg3_alpha0.5_T2_"
        "lr0.006_wd0.0001_seed6_sca0.5_ca0.1"
    )
    return ADULT / "rule_jsons" / stem


def generate(build_dir):
    for architecture in ARCHITECTURES:
        source = rule_dir(architecture)
        compile_rule_model(
            source / "rules_int8.json",
            build_dir / architecture,
            fallback_kind="tree",
            fallback_asset=source / "fallback_tree_int.npz",
            name=architecture,
            test_data=TEST_VECTORS,
            # The paper's Table IV designs are the tree-architecture family:
            # the rule ROM is reshaped and pinned to block RAM so the five
            # widths share one memory layout.
            prose="tree_arch",
            arch=architecture,
            hls_proj=f"tree_{architecture}_hls",
        )


def verify_metrics(fresh_csv, strict=False):
    with (HERE / "hardware_results.csv").open(newline="") as stream:
        expected = {
            row["architecture"]: row
            for row in csv.DictReader(stream)
            if row["method"] == "Rules"
        }
    with fresh_csv.open(newline="") as stream:
        actual = {Path(row["model"]).name: row for row in csv.DictReader(stream)}
    mapping = {
        "latency_cycles": "latency_cycles",
        "dsp": "DSP",
        "bram": "BRAM_18K",
        "lut": "LUT",
        "ff": "FF",
    }
    entries, missing = [], []
    for architecture, paper in expected.items():
        fresh = actual.get(architecture)
        if fresh is None:
            missing.append(f"{architecture}: missing fresh synthesis row")
            continue
        for paper_key, fresh_key in mapping.items():
            entries.append(
                (architecture, fresh_key, fresh[fresh_key], paper[paper_key])
            )
    return report("TABLE IV RULE-HARDWARE VERIFICATION (5 classifiers)", entries,
                  missing=missing, strict=strict)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-dir", type=Path, default=DEFAULT_BUILD)
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--vitis-hls", default="vitis-run")
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--generate-only", action="store_true")
    parser.add_argument("--clean", action="store_true")
    parser.add_argument("--strict", action="store_true",
                        help="require every resource metric to match exactly")
    args = parser.parse_args()

    if args.clean and args.build_dir.exists():
        shutil.rmtree(args.build_dir)
    args.build_dir.mkdir(parents=True, exist_ok=True)
    generate(args.build_dir)
    print(f"Generated five Table IV rule projects under {args.build_dir}")
    if args.generate_only:
        return 0
    if shutil.which(args.vitis_hls.split()[0]) is None:
        raise SystemExit(
            f"Cannot find {args.vitis_hls!r}. Load the required Vitis environment."
        )
    rc = run_synth([str(args.build_dir)], jobs=args.jobs, vitis_hls=args.vitis_hls)
    if rc:
        return rc
    rc = run_collect([str(args.build_dir)], csv_path=str(args.csv))
    return rc if rc else (0 if verify_metrics(args.csv, args.strict) else 1)


if __name__ == "__main__":
    raise SystemExit(main())
