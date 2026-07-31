"""Golden reference for the rule classifier: max-purity match, then fallback.

Written to mirror what the HLS kernel computes, so that csim comparing against
`data/test_output_ref.txt` is a meaningful check rather than a restatement of
the same code. It stays in float — the fixed-point datapath is what csim is
measuring the deviation *of*.

One trap worth naming: which weight vector a condition dots with is a
per-suite choice. `fallback_4_variance` projects with `sparse_weights` (the
retained top-k, missing names treated as zero) while the dense `tree_arch`
suite projects with the full `weight_vector`. Getting it backwards produces a
plausible-looking reference that silently disagrees with the hardware.
"""

import numpy as np

from . import fallback as fb


def rule_predict(model, X, dot_source='sparse_weights'):
    """Max-purity rule match. Returns labels, with -1 for uncovered inputs."""
    n = len(X)
    best_label = np.full(n, -1, dtype=int)
    best_purity = np.full(n, -1.0)
    for r in model.rules:
        fires = np.ones(n, dtype=bool)
        for c in r['conditions']:
            w = np.array(model.sparse_weight(c) if dot_source == 'sparse_weights'
                         else model.dense_weight(c))
            z = X @ w
            if c['band_lo'] is not None:
                fires &= (z >= c['band_lo'])
            if c['band_hi'] is not None:
                fires &= (z < c['band_hi'])
        upd = fires & (r['purity'] > best_purity)
        best_label[upd] = r['label']
        best_purity[upd] = r['purity']
    return best_label


def golden_predict(model, X, fallback_kind, fallback_params,
                   dot_source='sparse_weights'):
    """Full pipeline: rules first, fallback only on the uncovered inputs."""
    pred = rule_predict(model, X, dot_source)
    if fallback_kind == 'none':
        return pred
    uncovered = pred == -1
    if uncovered.any():
        pred = pred.copy()
        pred[uncovered] = fb.predict(fallback_kind, fallback_params, X[uncovered])
    return pred


def coverage_stats(model, X, y, dot_source='sparse_weights'):
    """Rule coverage and covered-accuracy, for cross-checking against the
    `metrics` block of the rule JSON and the shipped manifest."""
    pred = rule_predict(model, X, dot_source)
    covered = pred != -1
    n = len(X)
    return dict(
        covered_pct=100.0 * covered.sum() / n,
        covered_acc=(100.0 * (pred[covered] == y[covered]).sum() / covered.sum()
                     if covered.any() else float('nan')),
    )
