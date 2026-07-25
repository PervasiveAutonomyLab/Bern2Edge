"""
bern_regimes.py
---------------
Activation-geometry analysis of a trained Bernstein layer.

Each Bernstein neuron applies a learned degree-`d` polynomial to its (normalised)
pre-activation.  The *shape* of that polynomial — where it rises, falls, peaks, or
bends — partitions the neuron's input range into a handful of monotone "regimes".
These regimes are the atoms the rule extractor turns into human-readable
conditions of the form ``band_lo <= w . x < band_hi``.

This module is the single source of truth for the regime geometry:

  * ``N_FIXED_GRID``            - the fixed grid resolution (shared by the rule
                                  extractor AND the neuron visualiser, so the two
                                  always agree).
  * ``classify_motif``          - label a neuron's coefficient vector by shape.
  * ``compute_regime_breakpoints`` - interior breakpoints in normalised t-space.
  * ``build_neuron_regimes``    - per-neuron regimes in pre-activation (z) space.

Ported from the research pipeline (activation_distill.py + the two_moons motif
classifier), cleaned up and de-duplicated for publication.
"""

import numpy as np


# Fixed grid resolution.  Every non-flat neuron always receives these uniform
# interior breakpoints {k / (N_FIXED_GRID + 1) : k = 1..N_FIXED_GRID}, on top of
# any analytic turning/inflection points.  Grid=5 -> {1/6, 2/6, ..., 5/6}.
# Keep this value in one place: both extraction and visualisation import it.
N_FIXED_GRID = 5


# ─── Motif classification ────────────────────────────────────────────────────

def classify_motif(coeffs, global_coeff_std):
    """Classify one neuron's Bernstein coefficient vector by activation shape.

    Returns one of: 'flat', 'bump', 'valley', 'monotone_up', 'monotone_down',
    'other'.  `global_coeff_std` is the std of all coefficients in the layer and
    sets the 'flat' threshold.
    """
    c = coeffs
    c_range = c.max() - c.min()

    # flat: coefficient range is negligible relative to the layer.
    if c_range < 0.1 * (global_coeff_std + 1e-8):
        return 'flat'

    mono_tol = 0.05 * c_range

    # bump: an interior coefficient rises above both endpoints.
    if len(c) > 2 and c[1:-1].max() > max(c[0], c[-1]) + mono_tol:
        return 'bump'
    # valley: an interior coefficient dips below both endpoints.
    if len(c) > 2 and c[1:-1].min() < min(c[0], c[-1]) - mono_tol:
        return 'valley'

    diffs = np.diff(c)
    if np.all(diffs >= -mono_tol):
        return 'monotone_up'
    if np.all(diffs <= mono_tol):
        return 'monotone_down'
    return 'other'


def classify_motif_extended(coeffs, global_coeff_std):
    """Refine the 'other' bucket of `classify_motif` into sigmoid_up /
    sigmoid_down / complex using the sign of the two second differences."""
    base = classify_motif(coeffs, global_coeff_std)
    if base != 'other':
        return base
    d2_0 = coeffs[2] - 2 * coeffs[1] + coeffs[0]
    d2_1 = coeffs[3] - 2 * coeffs[2] + coeffs[1]
    if d2_0 * d2_1 < 0:   # one inflection -> sigmoid
        return 'sigmoid_up' if coeffs[-1] > coeffs[0] else 'sigmoid_down'
    return 'complex'


# ─── Analytic turning / inflection points (degree-3 only) ────────────────────

def bern_deriv_roots(coeffs):
    """Real roots of p'(t) in (0, 1) for a degree-3 Bernstein polynomial."""
    if len(coeffs) != 4:
        raise ValueError("bern_deriv_roots supports degree-3 (4 coefficients) only")
    d = np.diff(coeffs)
    A = 3 * (d[0] - 2 * d[1] + d[2])
    B = 6 * (d[1] - d[0])
    C = 3 * d[0]
    if abs(A) < 1e-8:                     # p' is (near) linear
        if abs(B) < 1e-8:
            return []
        t = -C / B
        return [float(t)] if 0.0 < t < 1.0 else []
    disc = B ** 2 - 4 * A * C
    if disc < 0:
        return []
    roots = [(-B + np.sqrt(disc)) / (2 * A), (-B - np.sqrt(disc)) / (2 * A)]
    return sorted(float(r) for r in roots if 0.0 < r < 1.0)


def bern_second_deriv_root(coeffs):
    """Root of p''(t) in (0, 1) for a degree-3 Bernstein polynomial (or None)."""
    d2_0 = coeffs[2] - 2 * coeffs[1] + coeffs[0]
    d2_1 = coeffs[3] - 2 * coeffs[2] + coeffs[1]
    denom = d2_0 - d2_1
    if abs(denom) < 1e-8:
        return None
    t_inf = d2_0 / denom
    return float(t_inf) if 0.0 < t_inf < 1.0 else None


def compute_regime_breakpoints(coeffs, motif, n_fixed_grid=N_FIXED_GRID):
    """Interior breakpoints (in normalised t-space) that split a neuron's curve
    into monotone regimes: a fixed uniform grid plus, for degree-3 neurons, the
    analytic turning and inflection points.  Flat neurons get no breakpoints.
    """
    if motif == 'flat':
        return []
    pts = set()
    # Always-present uniform grid.
    for k in range(1, n_fixed_grid + 1):
        pts.add(k / (n_fixed_grid + 1))
    # Shape-dependent points (degree-3 has closed-form roots).
    if len(coeffs) == 4:
        for root in bern_deriv_roots(coeffs):
            if 0.02 < root < 0.98:
                pts.add(root)
        t_inf = bern_second_deriv_root(coeffs)
        if t_inf is not None and 0.02 < t_inf < 0.98:
            pts.add(t_inf)
    return sorted(pts)


# ─── Per-neuron regimes in pre-activation space ──────────────────────────────

def build_neuron_regimes(model, n_fixed_grid=N_FIXED_GRID):
    """Extract per-neuron regime information from the model's FIRST Bernstein layer.

    Reads the first Linear layer (weights W0, bias b0) and the first Bernstein
    layer (per-neuron coefficients and calibrated input_bounds).  For each
    non-flat neuron it maps the t-space breakpoints into pre-activation z-space
    (z = w . x + b) and returns the resulting regime intervals.

    Returns a list of dicts, one per non-flat neuron:
        idx, w (H-vector), b, lo, hi, coeffs, motif, regimes, n_regimes
    where `regimes` is a list of (z_lo, z_hi) half-open intervals spanning
    (-inf, +inf), with the bound edges lo/hi inserted so out-of-range (clipped)
    inputs form their own regimes.
    """
    W0 = model.layers[0].weight.detach().cpu().numpy()   # (H, D)
    b0 = model.layers[0].bias.detach().cpu().numpy()     # (H,)
    bl = model.get_bern_layers()[0]
    bounds     = bl.input_bounds.detach().cpu().numpy()  # (H, 2)
    coeffs_all = bl.bern_coeffs.detach().cpu().numpy()    # (H, degree+1)
    global_std = coeffs_all.std()

    neurons = []
    for i in range(W0.shape[0]):
        lo, hi = float(bounds[i, 0]), float(bounds[i, 1])
        motif = classify_motif_extended(coeffs_all[i], global_std)
        if motif == 'flat':
            continue
        breaks = compute_regime_breakpoints(coeffs_all[i], motif, n_fixed_grid)
        z_breaks = [lo + t * (hi - lo) for t in breaks]
        # Insert the bound edges so clipped (out-of-range) inputs are separated
        # from the polynomial interior, then bracket with +/- inf.
        z_breaks = sorted(set([lo] + z_breaks + [hi]))
        z_edges = [-np.inf] + z_breaks + [np.inf]
        regimes = [(z_edges[j], z_edges[j + 1]) for j in range(len(z_edges) - 1)]
        neurons.append({
            'idx':       i,
            'w':         W0[i],
            'b':         b0[i],
            'lo':        lo,
            'hi':        hi,
            'coeffs':    coeffs_all[i],
            'motif':     motif,
            'regimes':   regimes,
            'n_regimes': len(regimes),
        })
    return neurons


def eval_bern_poly(coeffs, t):
    """Evaluate a Bernstein polynomial with the given coefficients at t in [0, 1]."""
    from math import comb
    n = len(coeffs) - 1
    result = np.zeros_like(t, dtype=float)
    for k, c in enumerate(coeffs):
        result += c * comb(n, k) * (t ** k) * ((1 - t) ** (n - k))
    return result
