"""Generate and optionally synthesize the five Table XII encoder-layer designs."""

from __future__ import annotations

import argparse
import csv
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(REPO))

from hls.bern2hls.bert.collect_modules import module_reports
from hls.bern2hls.bert.compile import compile_checkpoint
from hls.bern2hls.core.synth import run_synth

DEFAULT_BUILD = REPO / "build" / "table_xii_transformer_hls"
DEFAULT_CSV = DEFAULT_BUILD / "fresh_table_xii_hls.csv"
MODELS = {
    "teacher": HERE / "models" / "teacher_gelu_9037.pt",
    "gelu_h600": HERE / "models" / "release_gelu_h600.pt",
    "bern_h600": HERE / "models" / "release_bern_h600.pt",
    "gelu_h312": HERE / "models" / "release_gelu_h312.pt",
    "bern_h312": HERE / "models" / "release_bern_h312.pt",
}
PROJECTS = {
    "teacher": ("h600", "gelu_poly_layer"),
    "gelu_h600": ("h600", "gelu_poly_layer_h600"),
    "bern_h600": ("h600", "bern_layer_lut50"),
    "gelu_h312": ("h312", "gelu_poly_layer"),
    "bern_h312": ("h312", "bern_layer_lut50"),
}
METRICS = ("latency", "dsp", "bram", "ff", "lut")


def generate(build_dir):
    for checkpoint in MODELS.values():
        compile_checkpoint(checkpoint, build_dir, scope="layer")


def report_dir(build_dir, tag):
    bundle, project = PROJECTS[tag]
    matches = list(
        (build_dir / bundle / "layer" / project / "script").glob(
            "*_hls/solution1/syn/report"
        )
    )
    if len(matches) != 1:
        raise FileNotFoundError(f"{tag}: expected one synthesis report, found {matches}")
    return matches[0]


def measured_values(build_dir):
    values = {}
    for tag in MODELS:
        modules = module_reports(report_dir(build_dir, tag))
        top = modules["layer_top"]
        attention = modules["attention_sublayer"]
        values[("full", "latency", tag)] = top["lat"] * 4
        values[("ffn", "latency", tag)] = (top["lat"] - attention["lat"]) * 4
        for metric, report_key in (
            ("dsp", "DSP"),
            ("bram", "BRAM_18K"),
            ("ff", "FF"),
            ("lut", "LUT"),
        ):
            values[("full", metric, tag)] = top[report_key]
            values[("ffn", metric, tag)] = top[report_key] - attention[report_key]
    return values


def write_and_verify(build_dir, csv_path):
    values = measured_values(build_dir)
    columns = ("scope", "metric", *MODELS)
    rows = []
    for scope in ("ffn", "full"):
        for metric in METRICS:
            rows.append({
                "scope": scope,
                "metric": metric,
                **{tag: values[(scope, metric, tag)] for tag in MODELS},
            })
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)

    with (HERE / "results" / "table_xii_hls.csv").open(newline="") as stream:
        expected = {
            (row["scope"], row["metric"]): row
            for row in csv.DictReader(line for line in stream if not line.startswith("#"))
        }
    mismatches = []
    for row in rows:
        paper = expected[(row["scope"], row["metric"])]
        for tag in MODELS:
            if str(row[tag]) != paper[tag]:
                mismatches.append(
                    f"{row['scope']}/{row['metric']}/{tag}: "
                    f"fresh={row[tag]} paper={paper[tag]}"
                )
    if mismatches:
        print("\nTABLE XII HARDWARE VERIFICATION: FAIL")
        for mismatch in mismatches:
            print(f"  {mismatch}")
        return False
    print("\nTABLE XII HARDWARE VERIFICATION: PASS")
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
    print(f"Generated five Table XII encoder-layer projects under {args.build_dir}")
    if args.generate_only:
        return 0
    if shutil.which(args.vitis_hls.split()[0]) is None:
        raise SystemExit(
            f"Cannot find {args.vitis_hls!r}. Load the required Vitis environment."
        )
    roots = [
        args.build_dir / "h312" / "layer",
        args.build_dir / "h600" / "layer",
    ]
    rc = run_synth([str(root) for root in roots], jobs=args.jobs,
                   vitis_hls=args.vitis_hls)
    if rc:
        return rc
    return 0 if write_and_verify(args.build_dir, args.csv) else 1


if __name__ == "__main__":
    raise SystemExit(main())
