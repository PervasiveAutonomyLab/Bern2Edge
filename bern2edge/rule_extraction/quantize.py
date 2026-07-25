"""
quantize.py
-----------
Fixed-point quantization of extracted rules for hardware deployment.

The float rules use real-valued weight vectors and band thresholds. On an FPGA
these become:

  * rule weight vectors  -> per-vector symmetric int8 (one shared scale each)
  * band thresholds      -> fix<16,8> (8 integer + 8 fractional bits)
  * CART split thresholds -> fix<16,8>  (feature indices / leaf labels stay int)

`quantize_rule_json` produces the `rules_int8.json` companion to `rules_float.json`;
`quantize_cart_thresholds` produces the fix<16,8> tree structure. This mirrors the
research bundle exactly (weights int8 per-vector, thresholds fix<16,8>).

Ported from make_synthesis_bundle.py + quantize_int8_save_k7.py (numpy only).
"""

import copy

import numpy as np


def quantize_int8_sym(arr):
    """Per-vector symmetric int8: scale = max(|arr|) / 127.

    Returns (dequantized_values, scale) where dequantized_values are the int8
    grid points as floats (the values a fixed-point unit would actually use).
    """
    arr = np.asarray(arr, dtype=np.float64)
    amax = np.max(np.abs(arr))
    if amax == 0:
        return arr.copy(), 0.0
    scale = amax / 127.0
    q_int = np.round(arr / scale).astype(np.int8)
    return q_int.astype(np.float64) * scale, scale


def quantize_fix16_8_scalar(val):
    """fix<16,8> for a threshold: 8 integer bits, 8 fractional bits."""
    return float(np.clip(np.round(val * 256) / 256, -128.0, 128.0 - 1.0 / 256))


def quantize_rule_json(data):
    """Return an int8-quantized copy of a rules JSON dict.

    Each condition's weight_vector becomes per-vector int8 (with `int8_scale`)
    and its band_lo/band_hi become fix<16,8>. An LR fallback's w_eff/b_eff are
    quantized the same way. All other fields are left untouched.
    """
    q = copy.deepcopy(data)
    q['quantization'] = {'weights': 'int8_symmetric_per_vector',
                         'thresholds': 'fix16_8',
                         'fallback': 'int8/fix16_8 if linear'}
    for r in q['rules']:
        for c in r['conditions']:
            w_q, scale = quantize_int8_sym(np.asarray(c['weight_vector'], dtype=np.float64))
            c['weight_vector'] = [round(float(v), 8) for v in w_q]
            c['int8_scale'] = round(float(scale), 10)
            if c.get('band_lo') is not None:
                c['band_lo'] = quantize_fix16_8_scalar(c['band_lo'])
            if c.get('band_hi') is not None:
                c['band_hi'] = quantize_fix16_8_scalar(c['band_hi'])
    fb = q.get('fallback', {})
    if 'w_eff' in fb:                       # LR fallback
        w_q, scale = quantize_int8_sym(np.asarray(fb['w_eff'], dtype=np.float64))
        fb['w_eff'] = [round(float(v), 8) for v in w_q]
        fb['int8_scale'] = round(float(scale), 10)
        fb['b_eff'] = quantize_fix16_8_scalar(fb['b_eff'])
    return q


def cart_arrays(tree):
    """Flatten a fitted sklearn DecisionTreeClassifier into the arrays a hardware
    evaluator needs: children, split feature/threshold, and per-node class votes."""
    t = tree.tree_
    return {
        'children_left':  t.children_left,
        'children_right': t.children_right,
        'feature':        t.feature,
        'threshold':      t.threshold,
        'value':          t.value.reshape(t.value.shape[0], -1),
    }


def quantize_cart_thresholds(arrays):
    """Return a copy of the CART arrays with split thresholds cast to fix<16,8>
    (leaf nodes, marked by children_left == -1, keep their sentinel threshold)."""
    q = {k: np.array(v, copy=True) for k, v in arrays.items()}
    thr = np.array([quantize_fix16_8_scalar(t) if q['children_left'][i] != -1 else t
                    for i, t in enumerate(arrays['threshold'])])
    q['threshold'] = thr
    return q
