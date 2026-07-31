"""Command-line interface for Bern2Edge BNN and rule-network HLS generation."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


DATASETS = ("adult", "higgs", "covertype")
RULE_FALLBACKS = ("none", "lr", "tree", "network", "small_nn")


def main(argv=None):
    parser = argparse.ArgumentParser(prog="bern2hls", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    compile_parser = sub.add_parser("compile", help="Compile .pth checkpoints to HLS projects")
    compile_parser.add_argument("--dataset", choices=DATASETS, required=True)
    compile_parser.add_argument("--pth", nargs="+", required=True)
    compile_parser.add_argument("--out", required=True)
    compile_parser.add_argument("--profile", default="kv260_default")
    compile_parser.add_argument("--part")
    compile_parser.add_argument("--clock", type=float)

    quantize_parser = sub.add_parser(
        "quantize-rules",
        help="Convert a float rule JSON (and optional CART sidecar) for HLS",
    )
    quantize_parser.add_argument("--rules", required=True)
    quantize_parser.add_argument("--output", required=True)
    quantize_parser.add_argument("--cart")
    quantize_parser.add_argument("--cart-output")

    rules_parser = sub.add_parser(
        "compile-rules", help="Compile one HLS-ready rules JSON to an HLS project"
    )
    rules_parser.add_argument("--rules", required=True)
    rules_parser.add_argument("--fallback", choices=RULE_FALLBACKS, default="none")
    rules_parser.add_argument("--fallback-model")
    rules_parser.add_argument("--out", required=True)
    rules_parser.add_argument("--name")
    rules_parser.add_argument("--profile", default="kv260_default")
    rules_parser.add_argument("--test-data")
    rules_parser.add_argument("--num-test", type=int, default=9045)
    rules_parser.add_argument(
        "--fallback-only",
        action="store_true",
        help="Generate only the selected fallback kernel (Table IX resources)",
    )

    transformer_parser = sub.add_parser(
        "compile-transformer",
        help="Compile a Bern2Edge Transformer release to a paper HLS project",
    )
    transformer_parser.add_argument("--model", required=True)
    transformer_parser.add_argument("--scope", choices=("ffn", "layer"), default="layer")
    transformer_parser.add_argument("--layer", type=int, default=0)
    transformer_parser.add_argument("--out", required=True)
    transformer_parser.add_argument("--with-data", action="store_true")

    synth_parser = sub.add_parser("synth", help="Run Vitis HLS csim or csynth")
    synth_parser.add_argument("--root", nargs="+", required=True)
    synth_parser.add_argument("--csim", action="store_true")
    synth_parser.add_argument("--only")
    synth_parser.add_argument("--jobs", type=int, default=1)
    synth_parser.add_argument("--vitis-hls", default="vitis-run")

    collect_parser = sub.add_parser("collect", help="Collect metrics from Vitis reports")
    collect_parser.add_argument("--root", nargs="+", required=True)
    collect_parser.add_argument("--csv", required=True)
    collect_parser.add_argument("--accuracy", action="store_true")

    ooc_parser = sub.add_parser("ooc", help="Optional Vivado out-of-context implementation")
    ooc_parser.add_argument("--root", nargs="+", required=True)
    ooc_parser.add_argument("--jobs", type=int, default=1)
    ooc_parser.add_argument("--part", default="xck26-sfvc784-2LV-c")
    ooc_parser.add_argument("--clock", type=float, default=10.0)
    ooc_parser.add_argument("--vivado", default="vivado")
    ooc_parser.add_argument("--force", action="store_true")

    args = parser.parse_args(argv)
    if args.command == "compile":
        from .fc.compile import compile_dataset

        compile_dataset(
            args.dataset,
            out_dir=args.out,
            pth_inputs=args.pth,
            profile=args.profile,
            part=args.part,
            clock_ns=args.clock,
        )
        return 0
    if args.command == "quantize-rules":
        from bern2edge.rule_extraction.quantize import (
            quantize_cart_thresholds,
            quantize_rule_json,
        )

        if bool(args.cart) != bool(args.cart_output):
            parser.error("--cart and --cart-output must be supplied together")
        rules_path = Path(args.rules)
        output_path = Path(args.output)
        with rules_path.open() as stream:
            quantized = quantize_rule_json(json.load(stream))
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w") as stream:
            json.dump(quantized, stream, indent=2)
            stream.write("\n")
        print(f"Wrote HLS-ready rules to {output_path}")

        if args.cart:
            import numpy as np

            with np.load(args.cart) as data:
                arrays = {key: data[key] for key in data.files}
            cart_output = Path(args.cart_output)
            cart_output.parent.mkdir(parents=True, exist_ok=True)
            np.savez(cart_output, **quantize_cart_thresholds(arrays))
            print(f"Wrote fix<16,8> CART fallback to {cart_output}")
        return 0
    if args.command == "compile-rules":
        from .rules.compile import compile_rule_model

        compile_rule_model(
            args.rules,
            args.out,
            fallback_kind=args.fallback,
            fallback_asset=args.fallback_model,
            profile=args.profile,
            test_data=args.test_data,
            name=args.name,
            num_test=args.num_test,
            fallback_only=args.fallback_only,
        )
        return 0
    if args.command == "compile-transformer":
        from .bert.compile import compile_checkpoint

        compile_checkpoint(
            args.model,
            args.out,
            scope=args.scope,
            layer=args.layer,
            with_data=args.with_data,
        )
        return 0
    if args.command == "synth":
        from .core.synth import run_synth

        return run_synth(
            args.root,
            only=args.only,
            csim=args.csim,
            jobs=args.jobs,
            vitis_hls=args.vitis_hls,
        )
    if args.command == "collect":
        from .core.collect import run_collect

        return run_collect(args.root, csv_path=args.csv, accuracy=args.accuracy)
    if args.command == "ooc":
        from .core.ooc import run_ooc

        return run_ooc(
            args.root,
            jobs=args.jobs,
            part=args.part,
            clk_ns=args.clock,
            vivado=args.vivado,
            force=args.force,
        )
    return 1


if __name__ == "__main__":
    sys.exit(main())
