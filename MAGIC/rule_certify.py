"""Sound rule-system certification primitives for MAGIC (paper TABLE X).

Interval-propagation certificates for the symbolic rule system: given a per-feature
noise box, decide whether a point's rule-system *decision* is provably unchanged
(Decision-Flip / "Rules" column). Rules are ANDs of linear "slabs"
``band_lo <= w·z < band_hi`` on first-layer pre-activations; the system predicts by
max-purity with a nearest-centroid fallback (mirrors ``extract_rules.py``).

Self-contained: numpy + the saved ``QuantileTransformer`` (via ``teacher_magic.pt``);
torch only to read the checkpoint. Trimmed to exactly what the TABLE X driver
(``relu_lirpa_certify.py``) imports — the per-feature-tolerance / margin-bracket
side-analyses are intentionally omitted.
"""
import json
import os

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TEACHER_PT = os.path.join(SCRIPT_DIR, "teacher_magic.pt")


# --------------------------------------------------------------------------- #
# Interval bounds on a rule's slab pre-activations over an axis-aligned box.
# --------------------------------------------------------------------------- #
def _preact_interval(w, z_lo, z_hi):
    a = z_lo * w
    b = z_hi * w
    return np.minimum(a, b).sum(axis=1), np.maximum(a, b).sum(axis=1)


def rule_certified_in(conds, z_lo, z_hi):
    """Box entirely satisfies ALL of the rule's conditions (rule certainly fires)."""
    ok = np.ones(z_lo.shape[0], dtype=bool)
    for w, lo, hi in conds:
        plo, phi = _preact_interval(w, z_lo, z_hi)
        if lo is not None:
            ok &= plo >= lo
        if hi is not None:
            ok &= phi < hi
    return ok


def rule_can_fire(conds, z_lo, z_hi):
    """Conservatively True unless the box CERTIFIES the rule cannot fire (some condition
    is entirely outside its band over the whole box). Sound for exclusion."""
    ok = np.ones(z_lo.shape[0], dtype=bool)
    for w, lo, hi in conds:
        plo, phi = _preact_interval(w, z_lo, z_hi)
        sat = np.ones(z_lo.shape[0], dtype=bool)
        if lo is not None:
            sat &= phi >= lo          # some box point can reach >= band_lo
        if hi is not None:
            sat &= plo < hi           # some box point can reach < band_hi
        ok &= sat
    return ok


# --------------------------------------------------------------------------- #
# Rule-system prediction (max-purity + nearest-centroid fallback).
# --------------------------------------------------------------------------- #
def predict_system_detail(Z, system):
    """predict_system + the winner's purity and rule index (fallback -> widx=-1, pstar=-1)."""
    compiled, c0, c1 = system
    N = Z.shape[0]
    pred = np.full(N, -1, dtype=int)
    pstar = np.full(N, -1.0)
    widx = np.full(N, -1, dtype=int)
    for ri, (label, purity, conds) in enumerate(compiled):
        m = np.ones(N, dtype=bool)
        for w, lo, hi in conds:
            d = Z @ w
            if lo is not None:
                m &= d >= lo
            if hi is not None:
                m &= d < hi
        up = m & (purity > pstar)
        pred[up] = label
        pstar[up] = purity
        widx[up] = ri
    uncov = pred == -1
    if uncov.any():
        Zr = Z[uncov]
        d0 = np.linalg.norm(Zr - c0, axis=1)
        d1 = np.linalg.norm(Zr - c1, axis=1)
        pred[uncov] = (d1 < d0).astype(int)
    return pred, pstar, widx


def predict_system(Z, system):
    """Final predicted label of the full rule system (max-purity; uncovered -> centroid)."""
    compiled, c0, c1 = system
    N = Z.shape[0]
    pred = np.full(N, -1, dtype=int)
    pred_purity = np.full(N, -1.0)
    for label, purity, conds in compiled:
        m = np.ones(N, dtype=bool)
        for w, lo, hi in conds:
            d = Z @ w
            if lo is not None:
                m &= d >= lo
            if hi is not None:
                m &= d < hi
        upgrade = m & (purity > pred_purity)
        pred[upgrade] = label
        pred_purity[upgrade] = purity
    uncov = pred == -1
    if uncov.any():
        Zr = Z[uncov]
        d0 = np.linalg.norm(Zr - c0, axis=1)
        d1 = np.linalg.norm(Zr - c1, axis=1)
        pred[uncov] = (d1 < d0).astype(int)
    return pred


def certify_decision_stable(z_lo, z_hi, system, pred, pstar, widx):
    """Sound per-point certificate that the predicted label is unchanged over the box.

    Certified iff (1) the winning rule stays firing over the WHOLE box, and (2) no
    different-label rule with purity >= the winner's can be entered anywhere in the box.
    Only defined for covered points (widx >= 0); fallback points return False.
    Returns (stable, covered).
    """
    compiled, _, _ = system
    N = z_lo.shape[0]
    covered = widx >= 0
    winner_stays = np.zeros(N, dtype=bool)
    for ri, (_, _, conds) in enumerate(compiled):
        sel = covered & (widx == ri)
        if sel.any():
            winner_stays[sel] = rule_certified_in(conds, z_lo[sel], z_hi[sel])
    no_flip = np.ones(N, dtype=bool)
    for (label, purity, conds) in compiled:
        cf = rule_can_fire(conds, z_lo, z_hi)
        threat = covered & cf & (label != pred) & (purity >= pstar)
        no_flip &= ~threat
    stable = covered & winner_stays & no_flip
    return stable, covered


def fires(conds, Z):
    """Boolean mask: which rows of Z satisfy all (w, lo, hi) conditions (rule fires)."""
    m = np.ones(Z.shape[0], dtype=bool)
    for w, lo, hi in conds:
        d = Z @ w
        if lo is not None:
            m &= d >= lo
        if hi is not None:
            m &= d < hi
    return m


def noise_box(qt, X_raw_pts, std_raw, c):
    """Per-feature raw noise box [x-c*std, x+c*std] mapped to normalized space via the
    monotone quantile transform. Returns (z_lo, z_hi)."""
    budget = c * std_raw
    z_lo = qt.transform(X_raw_pts - budget)   # monotone increasing -> lower edges
    z_hi = qt.transform(X_raw_pts + budget)
    return z_lo, z_hi


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
def load_checkpoint():
    """Load the fixed split + fitted preprocessor + raw arrays from teacher_magic.pt."""
    import torch  # local import; only needed to read the .pt
    ckpt = torch.load(TEACHER_PT, weights_only=False, map_location="cpu")
    for req in ("preprocessor", "idx_test", "idx_train", "X_raw", "y_raw"):
        assert req in ckpt, f"checkpoint missing key {req!r}"
    qt = ckpt["preprocessor"]
    X_raw = np.asarray(ckpt["X_raw"], dtype=np.float64)
    y_raw = np.asarray(ckpt["y_raw"]).astype(int)
    idx_test = np.asarray(ckpt["idx_test"])
    idx_train = np.asarray(ckpt["idx_train"])
    feat_names = list(ckpt["feature_names"])
    return qt, X_raw, y_raw, idx_train, idx_test, feat_names


def load_rule(rule_file, rule_index, min_cov, min_pur, feat_names):
    """Load the rule JSON and attach numpy weight vectors to each condition."""
    with open(rule_file) as f:
        data = json.load(f)
    assert data["feature_names"] == feat_names, (
        "feature order mismatch between rule JSON and checkpoint:\n"
        f"  json: {data['feature_names']}\n  ckpt: {feat_names}"
    )
    rules = data["rules"]
    if rule_index is None:
        chosen = None
        for i, r in enumerate(rules):
            if r["coverage"] > min_cov and r["purity"] > min_pur:
                chosen = i
                break
        rule_index = chosen if chosen is not None else 0
    rule = rules[rule_index]
    for c in rule["conditions"]:
        c["w"] = np.asarray(c["weight_vector"], dtype=np.float64)
        c["w_inf"] = float(np.max(np.abs(c["w"])))
    return data, rule, rule_index


def build_rule_system(data):
    """Compile all rules (JSON order) + purity + centroid fallback into arrays."""
    compiled = []
    for r in data["rules"]:
        if not r["conditions"]:
            continue
        conds = [(np.asarray(c["weight_vector"], dtype=np.float64),
                  c["band_lo"], c["band_hi"]) for c in r["conditions"]]
        compiled.append((int(r["label"]), float(r["purity"]), conds))
    fb = data["fallback"]
    c0 = np.asarray(fb["c0"], dtype=np.float64)
    c1 = np.asarray(fb["c1"], dtype=np.float64)
    return compiled, c0, c1
