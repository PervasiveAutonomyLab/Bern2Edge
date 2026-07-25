"""
run_rule_extraction.py
----------------------
Extract interpretable rules from the trained Bernstein Adult students and write
the per-config metrics used to build the paper's rule-extraction table.

For each architecture and each (same_cov_alpha, conflict_alpha) penalty combo it:
  * loads the student `.pth` and the matching data fold (identical splits to
    training, via `data.adult_ordinal_dataloaders_kfold`),
  * runs `rule_extraction.extract_rules` with the paper hyperparameters
    (grid=5, depth=3, p90 purity cascade, CART fallback, dense),
  * writes `rules_float.json` + `rules_int8.json` (+ CART sidecars) per run,
  * appends one row to `rule_results.csv`.

The defaults reproduce TABLE III (dense, CART fallback, sca in {0.5, 0.1},
ca=0.1). Other fallbacks are available via --fallback for later tables.

Run:
    cd Bern2Edge
    python Adult/run_rule_extraction.py                     # all 5 archs, both sca
    python Adult/run_rule_extraction.py --arch 14x16x2      # one arch
    python Adult/run_rule_extraction.py --fallback lr       # different fallback
"""

import argparse
import csv
import os
import sys

import numpy as np
import torch

# Make the repo-root modules (data, models, rule_extraction) importable.
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, os.pardir))
sys.path.insert(0, REPO)

from bern2edge.data import (adult_ordinal_dataloaders_kfold,                      # noqa: E402
                  ADULT_NUM_FEATURES, ADULT_CAT_FEATURES)
from bern2edge.models import FCModel                                              # noqa: E402
from bern2edge.rule_extraction import ExtractionConfig, prepare_context, extract_rules  # noqa: E402
from bern2edge.rule_extraction.quantize import (quantize_rule_json,               # noqa: E402
                                      quantize_cart_thresholds, cart_arrays)

# ── Paths / fixed protocol ───────────────────────────────────────────────────
TEACHER_CKPT = os.path.join(HERE, "adult_teacher_ordinal.pt")
CKPT_DIR     = os.path.join(HERE, "rule_checkpoints")
JSON_DIR     = os.path.join(HERE, "rule_jsons")
RESULTS_CSV  = os.path.join(HERE, "rule_results.csv")
DATA_SEED    = 6            # student seed 6 == fold 0
FEATURE_NAMES = list(ADULT_NUM_FEATURES) + list(ADULT_CAT_FEATURES)

# Architectures and penalty combos for the paper table (strictest sca first, to
# match the published row order).
ARCHS = ["14x16x2", "14x32x2", "14x128x2", "14x16x8x2", "14x32x16x2"]
SAME_COV_ALPHAS = [0.5, 0.1]
CONFLICT_ALPHAS = [0.1]

CSV_FIELDS = [
    "rule_json_file_name", "log_file_name",
    "ckpt", "arch", "degree", "model_test_acc",
    "base_purity", "purity_stages", "max_depth",
    "fallback_mode", "sparsity_k",
    "conflict_alpha", "same_cov_alpha",
    "n_active_neurons", "avg_regimes_per_neuron", "total_candidates",
    "n_rules",
    "train_covered_pct", "test_covered_pct", "test_covered_rule_acc",
    "test_rule_acc", "test_fidelity", "test_covered_fidelity",
    "n_conflicts", "avg_conditions",
]


def _fold_arrays(folds, fold_idx):
    """Reconstruct the extractor's train/test arrays from the k-fold loaders.

    The extractor is trained on the fold's full development split (train + val,
    transformed with the fold's own preprocessor) and evaluated on the fixed test
    split — exactly how the research pipeline builds them.
    """
    train_loader, val_loader, test_loader = folds[fold_idx]
    X_tr, y_tr = (t.numpy() for t in train_loader.dataset.tensors)
    X_va, y_va = (t.numpy() for t in val_loader.dataset.tensors)
    X_te, y_te = (t.numpy() for t in test_loader.dataset.tensors)
    X_train = np.vstack([X_tr, X_va]).astype(np.float32)
    y_train = np.concatenate([y_tr, y_va]).astype(np.int64)
    return X_train, y_train, X_te.astype(np.float32), y_te.astype(np.int64)


def _load_student(ckpt_path):
    """Load a Bernstein FCModel student checkpoint (returns model, raw dict)."""
    raw = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model = FCModel(layer_sizes=raw["arch"], degree=raw["degree"],
                    act=raw.get("activation", "bern"), last_bern=False, dropout=0.0)
    model.load_state_dict(raw["state_dict"])
    model.eval()
    return model, raw


def _write_artifacts(run_dir, result):
    """Persist rules_float.json + rules_int8.json and any fallback sidecars."""
    import json
    os.makedirs(run_dir, exist_ok=True)
    with open(os.path.join(run_dir, "rules_float.json"), "w") as f:
        json.dump(result.rules_json, f, indent=2)
    with open(os.path.join(run_dir, "rules_int8.json"), "w") as f:
        json.dump(quantize_rule_json(result.rules_json), f, indent=2)
    # CART fallback: float + fix<16,8>-threshold structure for hardware eval.
    if result.fb_tree is not None:
        arrays = cart_arrays(result.fb_tree)
        np.savez(os.path.join(run_dir, "fallback_tree_float.npz"), **arrays)
        np.savez(os.path.join(run_dir, "fallback_tree_int.npz"),
                 **quantize_cart_thresholds(arrays))
    # small_nn fallback: the trained net's weights.
    if result.fb_model is not None:
        torch.save({"state_dict": result.fb_model.state_dict()},
                   os.path.join(run_dir, "fallback_net_float.pt"))


def run(archs, cfg_base, same_cov_alphas, conflict_alphas,
        out_csv=RESULTS_CSV, out_json_dir=JSON_DIR):
    """Run the extraction sweep. `out_csv` / `out_json_dir` let callers (e.g. the
    penalty sweep) write to a separate location without mixing outputs."""
    os.makedirs(out_json_dir, exist_ok=True)
    print("Loading Adult folds (same splits as training) ...", flush=True)
    folds, _, _ = adult_ordinal_dataloaders_kfold(TEACHER_CKPT, k=5, seed=DATA_SEED)

    write_header = not os.path.exists(out_csv)
    with open(out_csv, "a", newline="") as csv_f:
        writer = csv.DictWriter(csv_f, fieldnames=CSV_FIELDS)
        if write_header:
            writer.writeheader()

        for arch in archs:
            fname = f"kd_fc_{arch}_bern_deg3_alpha0.5_T2_lr0.006_wd0.0001_seed6.pth"
            ckpt_path = os.path.join(CKPT_DIR, fname)
            if not os.path.exists(ckpt_path):
                print(f"  SKIP {arch}: checkpoint not found ({fname})")
                continue

            print(f"\n=== {arch} ===", flush=True)
            model, raw = _load_student(ckpt_path)
            fold_idx = int(raw.get("seed", DATA_SEED)) - DATA_SEED
            X_train, y_train, X_test, y_test = _fold_arrays(folds, fold_idx)
            model_test_acc = round(float(raw.get("test_acc", raw.get("val_acc", 0.0))), 4)

            # Candidates depend only on the network + grid/depth/purity, so build
            # them once per checkpoint and reuse across penalty combos.
            context = prepare_context(model, X_train, y_train, X_test, y_test, cfg_base)

            for sca in same_cov_alphas:
                for ca in conflict_alphas:
                    from dataclasses import replace
                    cfg = replace(cfg_base, same_cov_alpha=sca, conflict_alpha=ca)
                    print(f"  -- sca={sca} ca={ca} fallback={cfg.fallback_mode} "
                          f"{'dense' if cfg.sparsity_k is None else f'k{cfg.sparsity_k}'} --",
                          flush=True)
                    result = extract_rules(cfg, os.path.basename(ckpt_path),
                                           raw["arch"], FEATURE_NAMES, context=context)

                    run_name = f"{os.path.splitext(fname)[0]}_sca{sca}_ca{ca}"
                    _write_artifacts(os.path.join(out_json_dir, run_name), result)

                    row = {
                        "rule_json_file_name": f"{run_name}/rules_float.json",
                        "log_file_name": "",
                        "ckpt": os.path.basename(ckpt_path),
                        "arch": str(raw["arch"]),
                        "degree": raw.get("degree", ""),
                        "model_test_acc": model_test_acc,
                        "base_purity": cfg.base_purity,
                        "purity_stages": ";".join(f"{p}:{c}" for p, c in cfg.purity_stages),
                        "max_depth": cfg.max_depth,
                        "fallback_mode": cfg.fallback_mode,
                        "sparsity_k": cfg.sparsity_k,
                        "conflict_alpha": ca,
                        "same_cov_alpha": sca,
                        **result.metrics,
                    }
                    writer.writerow(row)
                    csv_f.flush()

    print(f"\nDone. Results -> {out_csv}\n      JSONs   -> {out_json_dir}/")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arch", nargs="+", default=ARCHS,
                    help="architecture subset (default: all 5 paper archs)")
    ap.add_argument("--fallback", choices=["tree", "lr", "network", "small_nn"],
                    default="tree", help="fallback for uncovered inputs (default: tree)")
    ap.add_argument("--sparsity-k", type=int, default=None,
                    help="fixed-k condition sparsification (default: dense)")
    ap.add_argument("--same-cov", type=float, nargs="+", default=SAME_COV_ALPHAS,
                    help="same-coverage penalty values to sweep (default: 0.5 0.1)")
    ap.add_argument("--conflict", type=float, nargs="+", default=CONFLICT_ALPHAS,
                    help="conflict penalty values to sweep (default: 0.1)")
    ap.add_argument("--out-csv", default=RESULTS_CSV,
                    help="results CSV path (default: Adult/rule_results.csv)")
    ap.add_argument("--out-json-dir", default=JSON_DIR,
                    help="directory for per-run rule JSONs (default: Adult/rule_jsons)")
    args = ap.parse_args()

    cfg_base = ExtractionConfig(fallback_mode=args.fallback, sparsity_k=args.sparsity_k)
    run(args.arch, cfg_base, args.same_cov, args.conflict,
        out_csv=args.out_csv, out_json_dir=args.out_json_dir)


if __name__ == "__main__":
    main()
