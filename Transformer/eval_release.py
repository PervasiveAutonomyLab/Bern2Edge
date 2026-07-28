"""
Recompute the SST-2 accuracy row of paper TABLE XII from the shipped weights.

Evaluates the GeLU TinyBERT4 teacher and the four substituted-FFN variants
(Bernstein / GeLU x width 312 / 600) on the SST-2 validation split, compares each
against the accuracy recorded when the release was built, and writes
results/table_xii_acc.csv -- the file make_table_xii.py renders.

This is the artifact's reproducibility claim for TABLE XII: the numbers come from
the weights, not from a log. A mismatch is a hard failure (non-zero exit).

  python Transformer/eval_release.py                 # all five, GPU if present
  python Transformer/eval_release.py --variants bern_h312
  python Transformer/eval_release.py --device cpu

SST-2 is downloaded once via `datasets`; the stock BERT parts come from
huawei-noah/TinyBERT_General_4L_312D.
"""

import argparse
import csv
import json
import os
import sys

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import shared_utils as U  # noqa: E402
from bernbert import BernBertForSequenceClassification, param_summary  # noqa: E402

# The four substituted variants, in the column order paper TABLE XII uses.
VARIANTS = ["gelu_h600", "bern_h600", "gelu_h312", "bern_h312"]

# The teacher is a stock BertForSequenceClassification, not a release module, so
# it is handled separately. Its accuracy is the TABLE XII "TinyBERT4" column.
# The four substituted variants carry their expected accuracy in their .meta.json;
# the teacher has none, so it is pinned here: 788 of the 872 SST-2 dev examples,
# which the paper rounds to 90.37%.
TEACHER_CKPT = "teacher_gelu_9037.pt"
TEACHER_EXPECTED = 788 / 872

# The releases were verified under these tokenization/batching settings; changing
# them changes the accuracy (max_length in particular).
BATCH_SIZE = 32
MAX_LENGTH = 64


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Recompute the TABLE XII SST-2 accuracy row from the shipped weights")
    p.add_argument("--variants", nargs="+", default=VARIANTS, choices=VARIANTS,
                   help="substituted variants to evaluate (default: all four)")
    p.add_argument("--no-teacher", action="store_true",
                   help="skip the TinyBERT4 teacher column")
    p.add_argument("--device", default=None, choices=["cpu", "cuda"],
                   help="default: cuda when available, else cpu")
    p.add_argument("--tol", type=float, default=1e-4,
                   help="max |recomputed - expected| before this run fails")
    p.add_argument("--write-csv", default=None,
                   help="default: results/table_xii_acc.csv")
    return p.parse_args()


def evaluate_teacher(val_loader, device: torch.device) -> tuple:
    """Accuracy + parameter count of the GeLU TinyBERT4 teacher."""
    ckpt = U.models_path(TEACHER_CKPT)
    model = U.load_finetuned_teacher(ckpt, device, freeze=True, num_labels=2)
    n_params = sum(p.numel() for p in model.parameters())
    # 4 encoder layers x (intermediate.dense + output.dense), weights and biases.
    # LayerNorm is excluded -- see the ffn_params note in main().
    ffn_params = sum(
        p.numel()
        for layer in U.encoder_layers(model)
        for mod in (layer.intermediate.dense, layer.output.dense)
        for p in mod.parameters()
    )
    acc = U.evaluate_classifier(model, val_loader, device)
    return acc, n_params, ffn_params


def ffn_params_no_ln(model) -> int:
    """Parameters of the replaced FFN blocks, excluding their LayerNorm.

    `param_summary` counts each block's LayerNorm too, but LayerNorm is unchanged
    by the substitution and is excluded from the paper's FFN budget (SPEC.md 7),
    so it is dropped here to keep every row of the CSV comparable.
    """
    return sum(p.numel()
               for ffn in model.bern_ffns
               for name, p in ffn.named_parameters()
               if not name.startswith("ln."))


def evaluate_variant(name: str, val_loader, device: torch.device) -> tuple:
    """Accuracy + parameter counts of one substituted-FFN release."""
    base = U.models_path(f"release_{name}")
    with open(base + ".meta.json") as f:
        meta = json.load(f)
    model = BernBertForSequenceClassification(
        meta["replaced_layers"], hidden=meta["hidden"],
        degree=meta["degree"] or 0, act=meta["act"])
    model.load_state_dict(torch.load(base + ".pt", map_location="cpu")["state_dict"])
    model.to(device).eval()
    info = param_summary(model)
    acc = U.evaluate_classifier(model, val_loader, device)
    return acc, meta, info["total_params"], ffn_params_no_ln(model)


def main() -> None:
    args = parse_args()
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    out_csv = args.write_csv or U.results_path("table_xii_acc.csv")
    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)

    print("=" * 72)
    print("TABLE XII -- SST-2 accuracy recomputed from the shipped weights")
    print(f"  device={device}  batch_size={BATCH_SIZE}  max_length={MAX_LENGTH}")
    print("=" * 72)

    tokenizer = U.build_tokenizer()
    _, val_loader = U.make_glue_loaders(
        tokenizer, "sst2", BATCH_SIZE, MAX_LENGTH, with_labels=True, num_workers=2)

    rows = []
    failures = []

    if not args.no_teacher:
        acc, n_params, ffn_params = evaluate_teacher(val_loader, device)
        diff = abs(acc - TEACHER_EXPECTED)
        rows.append({
            "variant": "teacher", "act": "gelu", "hidden": 1200,
            "acc": f"{acc:.6f}", "expected_acc": f"{TEACHER_EXPECTED:.6f}",
            "abs_diff": f"{diff:.6f}", "total_params": n_params,
            "ffn_params": ffn_params,
        })
        status = "OK" if diff <= args.tol else "MISMATCH"
        print(f"  {'teacher (TinyBERT4)':<22} acc={acc:.4f}  "
              f"expected={TEACHER_EXPECTED:.4f}  |diff|={diff:.6f}  {status}")
        if diff > args.tol:
            failures.append(f"teacher: {acc:.6f} != {TEACHER_EXPECTED:.6f}")

    for name in args.variants:
        acc, meta, n_params, ffn_params = evaluate_variant(name, val_loader, device)
        expected = meta["verified_val_acc"]
        diff = abs(acc - expected)
        rows.append({
            "variant": name, "act": meta["act"], "hidden": meta["hidden"],
            "acc": f"{acc:.6f}", "expected_acc": f"{expected:.6f}",
            "abs_diff": f"{diff:.6f}", "total_params": n_params,
            "ffn_params": ffn_params,
        })
        status = "OK" if diff <= args.tol else "MISMATCH"
        print(f"  {name:<22} acc={acc:.4f}  expected={expected:.4f}  "
              f"|diff|={diff:.6f}  {status}")
        if diff > args.tol:
            failures.append(f"{name}: {acc:.6f} != {expected:.6f}")

    # ffn_params is the 4x FFN-block budget EXCLUDING LayerNorm (unchanged by the
    # substitution) -- the same convention as SPEC.md 7 and the paper.
    # Stable column order so the CSV regenerates byte-identically.
    fields = ["variant", "act", "hidden", "acc", "expected_acc", "abs_diff",
              "total_params", "ffn_params"]
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    print("=" * 72)
    print(f"Wrote {out_csv}")
    if failures:
        raise SystemExit(
            "FAIL -- the shipped weights did not reproduce their recorded accuracy "
            f"(tol={args.tol}):\n  " + "\n  ".join(failures))
    print(f"OK -- all {len(rows)} accuracies reproduced within tol={args.tol}")
    print("Render the table with: python Transformer/make_table_xii.py")


if __name__ == "__main__":
    main()
