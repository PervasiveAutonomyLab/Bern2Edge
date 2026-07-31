"""Phase-3 fallback models: what runs when no rule fires.

Four fallback variants ship, and the point of the ablation is that they differ
widely
in area for near-identical accuracy:

  network      full-precision Bernstein MLP (H=32), the source network itself
  small_nn     int8-quantized Bernstein MLP (H=4)
  tree         CART, depth <= 4 — comparisons only, no multipliers
  lr           logistic regression, folded to int8 weights in the rule JSON

Assets are loaded from .npz where possible so that this front-end needs only
numpy; torch is imported lazily and only for the two .pt/.pth checkpoints.
"""

import json
import os

import numpy as np

BERN_DEGREE = 3
NCK = [1, 3, 3, 1]          # C(3, k)

# state_dict key prefixes differ between the two training scripts
_NET_KEYS = ('net.0', 'net.1', 'net.2')
_LAYER_KEYS = ('layers.0', 'layers.1', 'layers.2')


def _load_bern_state(path, keys):
    import torch
    sd = torch.load(path, map_location='cpu', weights_only=False)['state_dict']
    a, b, c = keys
    return dict(W0=sd[f'{a}.weight'].numpy(), b0=sd[f'{a}.bias'].numpy(),
                coeffs=sd[f'{b}.bern_coeffs'].numpy(),
                bounds=sd[f'{b}.input_bounds'].numpy(),
                W2=sd[f'{c}.weight'].numpy(), b2=sd[f'{c}.bias'].numpy())


def _load_bern_npz(path):
    z = np.load(path)
    return {k: z[k] for k in ('W0', 'b0', 'coeffs', 'bounds', 'W2', 'b2')}


def load_bern(path):
    """Load a Bernstein-MLP fallback from either the packaged .npz or a .pt."""
    if path.endswith('.npz'):
        return _load_bern_npz(path)
    keys = _NET_KEYS if os.path.basename(path).startswith('fallback_network') \
        else _LAYER_KEYS
    return _load_bern_state(path, keys)


def load_tree(path):
    t = np.load(path)
    return dict(feat=t['feature'].astype(int), thr=t['threshold'].astype(float),
                left=t['children_left'].astype(int),
                right=t['children_right'].astype(int),
                value=t['value'].astype(float))


def load_lr(path):
    """The LR fallback lives inside the rule JSON, not a separate asset."""
    with open(path) as f:
        fb = json.load(f)['fallback']
    return dict(w=np.array(fb['w_eff']), b=float(fb['b_eff']),
                scale=fb['int8_scale'])


# ---- float golden predictors ------------------------------------------

def bern_forward(X, W0, b0, coeffs, bounds, W2, b2):
    """deg-3 Bernstein MLP forward; returns argmax labels."""
    z0 = X @ W0.T + b0
    lo, hi = bounds[:, 0], bounds[:, 1]
    u = np.clip((z0 - lo) / (hi - lo + 1e-8), 0.0, 1.0)
    k = np.arange(BERN_DEGREE + 1)
    basis = np.array(NCK) * (u[..., None] ** k) * ((1 - u)[..., None] ** (BERN_DEGREE - k))
    return ((basis * coeffs).sum(-1) @ W2.T + b2).argmax(1)


def tree_predict(X, T):
    feat, thr, left, right = T['feat'], T['thr'], T['left'], T['right']
    leaf_label = T['value'].argmax(1)
    out = np.empty(len(X), dtype=int)
    for i, x in enumerate(X):
        node = 0
        while feat[node] != -2:            # -2 marks a leaf in sklearn's dump
            node = left[node] if x[feat[node]] <= thr[node] else right[node]
        out[i] = leaf_label[node]
    return out


def lr_predict(X, L):
    return (X @ L['w'] + L['b'] > 0).astype(int)


def load(kind, asset_path):
    """Load whichever asset `kind` needs. Returns None for 'none'."""
    if kind == 'none':
        return None
    if kind == 'tree':
        return load_tree(asset_path)
    if kind == 'lr':
        return load_lr(asset_path)
    if kind in ('network', 'small_nn'):
        return load_bern(asset_path)
    raise ValueError(f'unknown fallback kind: {kind}')


def predict(kind, params, X):
    if kind == 'tree':
        return tree_predict(X, params)
    if kind == 'lr':
        return lr_predict(X, params)
    if kind in ('network', 'small_nn'):
        return bern_forward(X, **params)
    raise ValueError(f'unknown fallback kind: {kind}')
