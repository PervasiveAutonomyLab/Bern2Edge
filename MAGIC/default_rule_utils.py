"""Nearest-centroid fallback rule for points no extracted rule covers (local, MAGIC).

Vendored verbatim from the code that produced the shipped rule JSONs. The extractor
appends this as the rule with empty ``conditions``; at prediction time an uncovered
point is assigned the label of the nearer class centroid (in normalized feature space).
"""
import numpy as np


def build_default_rule(X_train, y_train, uncov_mask):
    """Centroid-based fallback for uncovered points."""
    c0 = X_train[y_train == 0].mean(axis=0)
    c1 = X_train[y_train == 1].mean(axis=0)
    N  = len(y_train)

    if uncov_mask.sum() == 0:
        return {
            'conditions':  {},
            'c0':          c0,
            'c1':          c1,
            'purity':      1.0,
            'coverage':    N,
            'n_uncovered': 0,
        }

    X_uncov = X_train[uncov_mask]
    y_uncov = y_train[uncov_mask]
    d0 = np.linalg.norm(X_uncov - c0, axis=1)
    d1 = np.linalg.norm(X_uncov - c1, axis=1)
    assignments = (d1 < d0).astype(int)
    purity = float((assignments == y_uncov).mean())

    return {
        'conditions':  {},
        'c0':          c0,
        'c1':          c1,
        'purity':      purity,
        'coverage':    N,
        'n_uncovered': int(uncov_mask.sum()),
    }
