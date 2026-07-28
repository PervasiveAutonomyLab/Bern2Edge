"""
Build a release checkpoint from a Stage-3 training checkpoint, for either a
Bernstein or a GeLU substituted variant.

A Stage-3 checkpoint is the monkey-patched training object: it holds the full
BertForSequenceClassification state (including the now-dead original GeLU FFN
weights) plus one FCModel state per replaced layer. This script folds those into
the clean `BernBertForSequenceClassification` module, verifies the clean module
reproduces the SST-2 validation accuracy, and writes `<name>.pt` + `<name>.meta.json`
-- the format the four shipped TABLE XII variants in models/ use.

Usage:
  python Transformer/build_release.py \
      --in Transformer/models/bern_4layer_kd_h312_best.pt \
      --out-dir Transformer/models --name release_bern_h312 \
      --expected-acc 0.9002

Add --npz to also emit a torch-free numpy copy of the weights (hardware handoff;
not shipped in this artifact).
"""

import argparse
import json
import os
import sys

import numpy as np
import torch

# This directory and the repository root on the path, so both `bernbert` and
# `bern2edge` resolve regardless of CWD.
_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_HERE)
for _p in (_HERE, _REPO_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from bernbert import BernBertForSequenceClassification, param_summary  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build a clean release into a bundle dir")
    p.add_argument("--in", dest="inp", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--name", required=True, help="basename, e.g. release_bern_h312")
    p.add_argument("--expected-acc", type=float, default=None)
    p.add_argument("--tol", type=float, default=0.002)
    p.add_argument("--no-verify", action="store_true")
    p.add_argument("--npz", action="store_true",
                   help="also write a torch-free numpy copy of the weights")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--max-length", type=int, default=64)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 64)
    print(f"Building release '{args.name}' from: {args.inp}")
    model = BernBertForSequenceClassification.from_stage3_checkpoint(
        args.inp, map_location="cpu").to(device)
    info = param_summary(model)
    print(f"  act             : {model.act}")
    print(f"  replaced layers : {info['replaced_layers']}")
    print(f"  hidden/degree   : {info['hidden']} / {info['degree']}")
    print(f"  total params    : {info['total_params']:,}")
    print(f"  sub FFN params  : {info['bern_ffn_params']:,}")

    verified_acc = None
    if not args.no_verify:
        import shared_utils as U
        tokenizer = U.build_tokenizer()
        _, val_loader = U.make_sst2_loaders(
            tokenizer, args.batch_size, args.max_length, with_labels=True)
        verified_acc = U.evaluate_classifier(model, val_loader, device)
        print(f"  clean-module SST-2 val acc: {verified_acc:.4f}")
        if args.expected_acc is not None:
            diff = abs(verified_acc - args.expected_acc)
            status = "OK" if diff <= args.tol else "MISMATCH"
            print(f"  vs expected {args.expected_acc:.4f}: |diff|={diff:.4f} -> {status}")
            assert diff <= args.tol, "clean module does not reproduce expected accuracy"

    meta = {
        "source_checkpoint": os.path.basename(args.inp),
        "model": "BernBertForSequenceClassification",
        "act": model.act,
        "base": "huawei-noah/TinyBERT_General_4L_312D",
        "replaced_layers": info["replaced_layers"],
        "hidden": info["hidden"],
        "degree": info["degree"] if model.act == "bern" else None,
        "total_params": info["total_params"],
        "sub_ffn_params": info["bern_ffn_params"],
        "verified_val_acc": verified_acc,
    }

    state = {k: v.cpu() for k, v in model.state_dict().items()}
    pt_path = os.path.join(args.out_dir, args.name + ".pt")
    meta_path = os.path.join(args.out_dir, args.name + ".meta.json")
    torch.save({"state_dict": state, "meta": meta}, pt_path)
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    print("=" * 64)
    print(f"Saved: {pt_path}")
    print(f"       {meta_path}")
    if args.npz:
        npz_path = os.path.join(args.out_dir, args.name + "_weights.npz")
        np.savez(npz_path, **{k: v.numpy() for k, v in state.items()})
        print(f"       {npz_path}  ({len(state)} tensors)")


if __name__ == "__main__":
    main()
