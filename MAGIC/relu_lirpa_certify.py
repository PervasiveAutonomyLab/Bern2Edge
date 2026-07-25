"""
relu_lirpa_certify.py  (MAGIC certified robustness — paper TABLE X)
==================================================================
Generates the full TABLE X row set for one architecture (default 10x64x32x2, seed 42):

  Empirical (K random per-feature noise draws):  Rule Acc.,  Fidelity
  Certified-stable % (worst-case over the box):  Rules (= decision-flip),  BNN,  ReLU

Certificates share one semantics (the binary margin logit1 - logit0 over the per-feature
noise box; label certified-stable iff the margin's sign is provably constant):
  - Rules  : rule-system decision, sound interval propagation on the slabs (`rule_certify`).
  - BNN    : the Bernstein student, via Bern-IBP (`bern_net.nn_certify_box`, subinterval_bounds).
  - ReLU   : the matched-arch ReLU student, via standard IBP in **auto_LiRPA** — the
             algorithmically matched equivalent of Bern-IBP (isolates the activation).

Soundness (certified ⊆ empirically-stable, 0 violations) and monotonicity of every
certified column in c are asserted at runtime. The NN-accuracy and Boundary-Exit /
per-feature-tolerance analyses are intentionally excluded (not part of TABLE X).

auto_LiRPA is needed ONLY here (the ReLU column). Install from GitHub so it does not
downgrade torch:  pip install --no-deps git+https://github.com/Verified-Intelligence/auto_LiRPA.git

Usage (from MAGIC/):
    RF=rule_jsons/magic_rules_kd_fc_10x64x32x2_bern_deg3_alpha0.5_T2_lr0.003_wd0.0001_seed42_g5_p85_mc5_d2_adaptive.json
    python relu_lirpa_certify.py --arch 10x64x32x2 --rule-file $RF --write-csv results/table_x.csv
"""
import os
import sys
import argparse
import contextlib
import warnings

import numpy as np
import torch

warnings.filterwarnings("ignore")

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import bern_net as bn  # noqa: E402  (local model core: Bern-IBP + ReLU IBP)
from rule_certify import (  # noqa: E402  (local rule-cert + data primitives)
    load_checkpoint, load_rule, build_rule_system, predict_system_detail,
    predict_system, certify_decision_stable, noise_box,
)

# auto_LiRPA method name -> compute_bounds(method=...)
_LIRPA_METHOD = {
    "ibp": "IBP",                     # matched to Bern-IBP (the headline ReLU column)
    "crown-ibp": "CROWN-IBP",         # optional stronger-relaxation footnote
    "alpha-crown": "CROWN-Optimized",
}


@torch.no_grad()
def certify_relu_lirpa(model, z_lo, z_hi, method="ibp", batch=1024, device="cpu"):
    """auto_LiRPA certified label for each input box [z_lo, z_hi] (np arrays [N, D]).

    Bounds the binary margin (logit1 - logit0) via the requested LiRPA `method` over the
    per-feature box, returns {1, 0, -1} with the SAME semantics as bern_net.nn_certify_box.
    """
    from auto_LiRPA import BoundedModule, BoundedTensor
    from auto_LiRPA.perturbations import PerturbationLpNorm

    lirpa_method = _LIRPA_METHOD[method]
    lo = np.asarray(z_lo, dtype=np.float32)
    hi = np.asarray(z_hi, dtype=np.float32)
    N = lo.shape[0]
    out = np.full(N, -1, dtype=np.int64)

    for s in range(0, N, batch):
        e = min(s + batch, N)
        lo_t = torch.as_tensor(lo[s:e], device=device)
        hi_t = torch.as_tensor(hi[s:e], device=device)
        x0 = (lo_t + hi_t) / 2
        bm = BoundedModule(model, torch.empty_like(x0), device=device)
        bm.eval()
        ptb = PerturbationLpNorm(norm=float("inf"), x_L=lo_t, x_U=hi_t)
        bx = BoundedTensor(x0, ptb)
        C = torch.tensor([[[-1.0, 1.0]]], device=device).expand(x0.shape[0], 1, 2)
        sink = open(os.devnull, "w")
        with contextlib.redirect_stdout(sink):
            lb, ub = bm.compute_bounds(x=(bx,), C=C, method=lirpa_method)
        sink.close()
        m_lo = lb.squeeze(1).detach().cpu().numpy()
        m_hi = ub.squeeze(1).detach().cpu().numpy()
        cert = np.full(e - s, -1, dtype=np.int64)
        cert[m_lo > 0] = 1
        cert[m_hi < 0] = 0
        out[s:e] = cert
    return out


def _derive_ckpt(arch_str, act):
    """Default checkpoint path for an arch string (e.g. '10x64x32x2') and activation."""
    tag = "bern_deg3" if act == "bern" else "relu_degNA"
    return os.path.join(HERE, "student_model_weights",
                        f"kd_fc_{arch_str}_{tag}_alpha0.5_T2_lr0.003_wd0.0001_seed42.pth")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arch", default="10x64x32x2",
                    help="arch tag used to derive default checkpoints (e.g. 10x64x32x2)")
    ap.add_argument("--bern-ckpt", default=None, help="override Bernstein student path")
    ap.add_argument("--relu-ckpt", default=None, help="override ReLU student path")
    ap.add_argument("--rule-file", required=True,
                    help="rule JSON for this arch (defines the covered-correct cohort)")
    ap.add_argument("--methods", nargs="+", default=["ibp"], choices=list(_LIRPA_METHOD),
                    help="LiRPA method(s) for the ReLU column. Default 'ibp' is the matched "
                         "equivalent to Bern-IBP; add crown-ibp/alpha-crown for the footnote.")
    ap.add_argument("--c-sweep", type=float, nargs="+", default=[0.0, 0.01, 0.03, 0.05, 0.10])
    ap.add_argument("--k-samples", type=int, default=20)
    ap.add_argument("--min-coverage", type=float, default=500)
    ap.add_argument("--min-purity", type=float, default=0.95)
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--write-csv", default=None, help="write the TABLE X rows to this CSV")
    ap.add_argument("--outdir", default=HERE)
    args = ap.parse_args()
    device = "cpu"

    # ---- data ----
    qt, X_raw, y_raw, idx_train, idx_test, feat_names = load_checkpoint()
    X_test = qt.transform(X_raw[idx_test]).astype(np.float64)
    X_raw_te = X_raw[idx_test]
    y_test = y_raw[idx_test]
    std_raw = X_raw[idx_train].std(axis=0)
    print(f"[data] test set: {X_test.shape[0]} points, {len(feat_names)} features")

    # ---- rule system + covered-correct cohort ----
    rule_file = args.rule_file
    if not os.path.exists(rule_file):
        for cand in (os.path.join(HERE, rule_file), os.path.join(HERE, "rule_jsons", rule_file)):
            if os.path.exists(cand):
                rule_file = cand
                break
    data, _rule, _ridx = load_rule(rule_file, None, args.min_coverage, args.min_purity, feat_names)
    system = build_rule_system(data)
    pred_all, pstar_all, widx_all = predict_system_detail(X_test, system)
    cohort = (widx_all >= 0) & (pred_all == y_test)
    nq = int(cohort.sum())
    print(f"[rule] file = {os.path.basename(rule_file)}")
    print(f"[cohort] covered-correct = {nq} ({100*cohort.mean():.1f}% of test)")

    # ---- models ----
    bern_ck = args.bern_ckpt or _derive_ckpt(args.arch, "bern")
    relu_ck = args.relu_ckpt or _derive_ckpt(args.arch, "relu")
    bern_model, bern_raw = bn.load_student(bern_ck)
    relu_model, relu_raw = bn.load_student(relu_ck)
    bern_nom = 100.0 * (bn.nn_predict(bern_model, X_test) == y_test).mean()
    relu_nom = 100.0 * (bn.nn_predict(relu_model, X_test) == y_test).mean()
    print(f"[nn] Bern {bern_raw['arch']} acc {bern_nom:.2f}%  |  ReLU {relu_raw['arch']} acc {relu_nom:.2f}%")
    assert bern_raw["arch"] == relu_raw["arch"], "Bernstein and ReLU archs must match"

    Xc_raw = X_raw_te[cohort]
    yc = y_test[cohort]
    rule_lab_nom = pred_all[cohort]
    bern_lab_nom = bn.nn_predict(bern_model, X_test)[cohort]
    relu_lab_nom = bn.nn_predict(relu_model, X_test)[cohort]

    rng = np.random.default_rng(0)
    rows = []
    for c in args.c_sweep:
        z_lo, z_hi = noise_box(qt, Xc_raw, std_raw, c)

        # ---- certified columns ----
        dec_stable, _ = certify_decision_stable(z_lo, z_hi, system, rule_lab_nom,
                                                pstar_all[cohort], widx_all[cohort])
        bern_stable = (bn.nn_certify_box(bern_model, z_lo, z_hi) == bern_lab_nom)
        relu_handibp = (bn.nn_certify_box(relu_model, z_lo, z_hi) == relu_lab_nom)
        relu_stable = {m: (certify_relu_lirpa(relu_model, z_lo, z_hi, m, args.batch, device)
                           == relu_lab_nom) for m in args.methods}
        if "ibp" in relu_stable:  # plumbing cross-check: LiRPA-IBP == hand-rolled IBP
            mism = int((relu_stable["ibp"] != relu_handibp).sum())
            assert mism == 0, f"LiRPA-IBP disagrees with hand-rolled IBP on {mism} pts (c={c})"

        # ---- empirical columns (K uniform draws in the raw box) ----
        budget = c * std_raw
        rc = ag = 0
        n_draw = max(1, args.k_samples) if c > 0 else 1
        dec_viol = bern_viol = 0
        relu_viol = {m: 0 for m in args.methods}
        for _ in range(n_draw):
            dz = rng.uniform(-budget, budget, size=Xc_raw.shape) if c > 0 else np.zeros_like(Xc_raw)
            Zp = qt.transform(Xc_raw + dz)
            rp = predict_system(Zp, system)
            npd = bn.nn_predict(bern_model, Zp)
            relu_npd = bn.nn_predict(relu_model, Zp)
            rc += int((rp == yc).sum())
            ag += int((rp == npd).sum())
            dec_viol += int((dec_stable & (rp != rule_lab_nom)).sum())
            bern_viol += int((bern_stable & (npd != bern_lab_nom)).sum())
            for m in args.methods:
                relu_viol[m] += int((relu_stable[m] & (relu_npd != relu_lab_nom)).sum())
        denom = nq * n_draw
        assert dec_viol == 0, f"UNSOUND: {dec_viol} decision-certified flipped (c={c})"
        assert bern_viol == 0, f"UNSOUND: {bern_viol} Bern-certified flipped (c={c})"
        for m in args.methods:
            assert relu_viol[m] == 0, f"UNSOUND: {relu_viol[m]} ReLU-{m}-certified flipped (c={c})"

        row = {
            "c": c,
            "rule_acc": 100 * rc / denom, "fidelity": 100 * ag / denom,
            "decision_flip": 100 * dec_stable.mean(),
            "nn_stable_bern": 100 * bern_stable.mean(),
        }
        for m in args.methods:
            row[f"relu_{m}"] = 100 * relu_stable[m].mean()
        rows.append(row)

    # monotonicity of every certified column
    cert_keys = ["decision_flip", "nn_stable_bern"] + [f"relu_{m}" for m in args.methods]
    for key in cert_keys:
        for a, b in zip(rows, rows[1:]):
            assert b[key] <= a[key] + 1e-6, f"certified {key} not monotone in c"
    # relaxation ordering across LiRPA methods (stronger >= weaker), if present
    order = [m for m in ["ibp", "crown-ibp", "alpha-crown"] if m in args.methods]
    for r in rows:
        for w0, w1 in zip(order, order[1:]):
            assert r[f"relu_{w1}"] >= r[f"relu_{w0}"] - 1e-6, \
                f"{w1} certified < {w0} at c={r['c']} (relaxation ordering violated)"

    _print_table(rows, args.methods, bern_raw["arch"])
    if args.write_csv:
        _write_csv(args.write_csv, rows, args.methods)
    print("\n[ok] soundness + monotonicity asserts passed.")


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #
_M_LABEL = {"ibp": "ReLU", "crown-ibp": "ReLU-CRWNIBP", "alpha-crown": "ReLU-aCROWN"}


def _print_table(rows, methods, arch):
    print("\n" + "=" * 66)
    print(f"TABLE X  (arch {arch}; certified = worst-case over the noise box)")
    print("=" * 66)
    print("  Empirical (%)          Certified-stable (%)")
    mcols = "".join(f" {_M_LABEL[m]:>12s}" for m in methods)
    print(f"  {'c':>5s} {'RuleAcc':>8s} {'Fidlty':>7s} {'Rules':>7s} {'BNN':>7s}{mcols}")
    for r in rows:
        mvals = "".join(f" {r[f'relu_{m}']:>12.1f}" for m in methods)
        print(f"  {r['c']:>5g} {r['rule_acc']:>8.1f} {r['fidelity']:>7.1f} "
              f"{r['decision_flip']:>7.1f} {r['nn_stable_bern']:>7.1f}{mvals}")


def _write_csv(path, rows, methods):
    import csv
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    fields = ["c", "rule_acc", "fidelity", "decision_flip", "nn_stable_bern"] + \
             [f"relu_{m}" for m in methods]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: (round(r[k], 4) if isinstance(r[k], float) else r[k])
                        for k in fields})
    print(f"[csv] wrote {path}")


if __name__ == "__main__":
    main()
