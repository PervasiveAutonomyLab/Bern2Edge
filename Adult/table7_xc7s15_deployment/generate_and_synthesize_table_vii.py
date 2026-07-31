"""Generate, synthesize, and optionally route the eight Table VII designs."""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ADULT = HERE.parent
REPO = ADULT.parent
sys.path.insert(0, str(REPO))

from hls.bern2hls.core.collect import run_collect
from hls.bern2hls.core.ooc import run_ooc
from hls.bern2hls.core.synth import run_synth
from hls.bern2hls.fc.compile import compile_dataset
from hls.bern2hls.rules.compile import compile_rule_model

CHECKPOINTS = (
    "kd_fc_14x4x2_bern_deg3_alpha0.5_T2_lr0.006_wd0.0001_seed9.pth",
    "kd_fc_14x8x2_bern_deg3_alpha0.85_T2_lr0.006_wd0.0001_seed7.pth",
    "kd_fc_14x16x2_bern_deg3_alpha0.85_T2_lr0.006_wd0.0001_seed8.pth",
    "kd_fc_14x32x2_bern_deg3_alpha0.5_T2_lr0.006_wd0.0001_seed7.pth",
    "kd_fc_14x64x2_bern_deg3_alpha0_T1_lr0.006_wd0.0001_seed9.pth",
    "kd_fc_14x128x2_bern_deg3_alpha0_T1_lr0.006_wd0.0001_seed7.pth",
)
DEFAULT_BUILD = REPO / "build/table_vii_xc7s15"


def generate(build):
    bnn = build / "bnn"
    for index, name in enumerate(CHECKPOINTS):
        profile = "lp_quant_dspbind" if index == 0 else "lp_per_neuron"
        compile_dataset("adult", out_dir=str(bnn),
                        pth_inputs=[str(ADULT / "student_model_weights" / name)],
                        profile=profile)
    rules = build / "rules"
    r50 = ADULT / "table9_fallback_ablation/artifacts"
    compile_rule_model(r50 / "tree/rules_int8.json", rules / "R50",
                       fallback_kind="tree",
                       fallback_asset=r50 / "tree/fallback_tree_int.npz",
                       profile="lp_rules_dspopt_lutram", name="R50",
                       hls_proj="fb_tree_dspopt_${tag}_hls", force_dense=True)
    r29 = ADULT / ("rule_jsons/kd_fc_14x16x8x2_bern_deg3_alpha0.5_T2_"
                   "lr0.006_wd0.0001_seed6_sca0.5_ca0.1")
    compile_rule_model(r29 / "rules_int8.json", rules / "R29",
                       fallback_kind="tree",
                       fallback_asset=r29 / "fallback_tree_int.npz",
                       profile="lp_rules_dspopt", name="R29",
                       prose="tree_arch", arch="14x16x8x2",
                       hls_proj="tree_14x16x8x2_dspopt_${tag}_hls")
    return [bnn, rules]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--build-dir", type=Path, default=DEFAULT_BUILD)
    p.add_argument("--vitis-hls", default="vitis-run")
    p.add_argument("--vivado", default="vivado")
    p.add_argument("--jobs", type=int, default=1)
    p.add_argument("--generate-only", action="store_true")
    p.add_argument("--skip-route", action="store_true")
    p.add_argument("--clean", action="store_true")
    args = p.parse_args()
    if args.clean and args.build_dir.exists():
        shutil.rmtree(args.build_dir)
    args.build_dir.mkdir(parents=True, exist_ok=True)
    roots = generate(args.build_dir)
    print(f"Generated all eight Table VII projects under {args.build_dir}")
    if args.generate_only:
        return 0
    env = ["QW=18", "QI=8", "HLS_PART=xc7s15ftgb196-1",
           "HLS_PERIOD=15", "HLS_TAG=xc7s15", "HLS_SKIP_CSIM=1"]
    rc = run_synth([str(r) for r in roots], jobs=args.jobs,
                   vitis_hls=args.vitis_hls, env=env)
    if rc or args.skip_route:
        return rc
    rc = run_ooc([str(r) for r in roots], jobs=args.jobs,
                 part="xc7s15ftgb196-1", clk_ns=15, vivado=args.vivado)
    if rc:
        return rc
    return run_collect([str(r) for r in roots],
                       csv_path=str(args.build_dir / "fresh_reports.csv"),
                       with_ooc=True, accuracy=True)


if __name__ == "__main__":
    raise SystemExit(main())
