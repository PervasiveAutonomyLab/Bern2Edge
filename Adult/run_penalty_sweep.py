"""
run_penalty_sweep.py
--------------------
Joint penalty sweep for one architecture (paper TABLE VIII: 14x32x2, dense, CART
fallback). It runs rule extraction over a grid of the two greedy-cover penalties
and writes one metrics row per (conflict_alpha, same_cov_alpha) combo.

  * conflict_alpha (alpha_conf) — penalty for re-covering an OPPOSITE-label point
  * same_cov_alpha (alpha_sc)   — penalty for re-covering a SAME-label point

------------------------------------------------------------------------------
TO CHANGE THE SWEEP: just edit the four constants below (or use the CLI flags).
------------------------------------------------------------------------------
"""

import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, os.pardir)))   # repo root

# run_rule_extraction lives next to this file; import its shared sweep runner.
sys.path.insert(0, HERE)
import run_rule_extraction as rre                                    # noqa: E402
from bern2edge.rule_extraction import ExtractionConfig                         # noqa: E402

# ── Sweep configuration — EDIT THESE ─────────────────────────────────────────
ARCH            = "14x32x2"                 # architecture to sweep (h = 32)
CONFLICT_ALPHAS = [0.1, 1.0]                # alpha_conf values
SAME_COV_ALPHAS = [0.1, 0.3, 0.5, 1.0]      # alpha_sc values
FALLBACK        = "tree"                    # uncovered-input fallback (CART)
SPARSITY_K      = None                      # None = dense rules
# ─────────────────────────────────────────────────────────────────────────────

# Outputs (kept separate from the TABLE III run so nothing is mixed).
OUT_CSV      = os.path.join(HERE, "penalty_sweep_14x32x2_results.csv")
OUT_JSON_DIR = os.path.join(HERE, "penalty_sweep_14x32x2_jsons")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arch", default=ARCH, help=f"architecture (default: {ARCH})")
    ap.add_argument("--conflict", type=float, nargs="+", default=CONFLICT_ALPHAS,
                    help=f"alpha_conf values (default: {CONFLICT_ALPHAS})")
    ap.add_argument("--same-cov", type=float, nargs="+", default=SAME_COV_ALPHAS,
                    help=f"alpha_sc values (default: {SAME_COV_ALPHAS})")
    ap.add_argument("--fallback", choices=["tree", "lr", "network", "small_nn"],
                    default=FALLBACK, help=f"fallback (default: {FALLBACK})")
    ap.add_argument("--out-csv", default=OUT_CSV)
    ap.add_argument("--out-json-dir", default=OUT_JSON_DIR)
    args = ap.parse_args()

    n = len(args.conflict) * len(args.same_cov)
    print(f"Penalty sweep on {args.arch}: {len(args.conflict)} x {len(args.same_cov)} "
          f"= {n} combos (fallback={args.fallback}, dense)\n"
          f"  conflict_alpha: {args.conflict}\n"
          f"  same_cov_alpha: {args.same_cov}\n", flush=True)

    cfg_base = ExtractionConfig(fallback_mode=args.fallback, sparsity_k=SPARSITY_K)
    rre.run([args.arch], cfg_base,
            same_cov_alphas=args.same_cov, conflict_alphas=args.conflict,
            out_csv=args.out_csv, out_json_dir=args.out_json_dir)


if __name__ == "__main__":
    main()
