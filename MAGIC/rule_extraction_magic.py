"""MAGIC rule-extraction primitives (self-contained, local).

Bernstein-neuron regime geometry + candidate-rule generation for the MAGIC rule
extractor (`extract_rules.py`). This is the "different method" the paper's TABLE V
uses: rules are linear "slab" conditions on **first-hidden-layer** Bernstein
pre-activations (``z = w·x + b``), not the repo-wide ``rule_extraction/`` package.

Vendored VERBATIM (via runtime introspection) from the code that produced the shipped
rule JSONs — ``activation_distill.py`` / ``coeff_analysis.classify_motif`` — so
re-extraction reproduces the published rules exactly. Kept local (not merged into the
shared ``rule_extraction/`` package) because the two extraction methods differ.

The module-level hyper-parameters below are the defaults; ``extract_rules.py`` overrides
them per run (e.g. ``rule_extraction_magic.N_FIXED_GRID = args.grid``), and every function
reads them as module globals at call time — matching the original driver's behaviour.
"""
import numpy as np

# ── Hyper-parameters (overridden per run by extract_rules.py) ─────────────────
PURITY_THRESHOLD = 0.95   # min purity to accept a rule
MIN_COVERAGE     = 10     # min training points per rule
MAX_DEPTH        = 3      # max conditions per rule (rule "depth")
REGIME_DELTA     = 0.15   # half-width of bump/valley peak sub-interval (t-space)
N_FIXED_GRID     = 7      # uniform grid -> (N_FIXED_GRID+1) regimes per monotone neuron


# ── Motif classification ──────────────────────────────────────────────────────
def classify_motif(coeffs: np.ndarray, global_coeff_std: float) -> str:
    """
    Classify a single neuron's Bernstein coefficient vector into one of:
      flat, bump, valley, monotone_up, monotone_down, other

    Args:
        coeffs: 1-D array of shape (degree+1,)
        global_coeff_std: std of all coefficients across the whole layer,
                          used for the 'flat' threshold
    Returns:
        motif label string
    """
    c = coeffs
    c_range = c.max() - c.min()

    # ── flat: range is negligible ─────────────────────────────────────────────
    flat_tol = 0.1 * (global_coeff_std + 1e-8)
    if c_range < flat_tol:
        return 'flat'

    mono_tol = 0.05 * c_range

    # ── bump: interior max rises above both endpoints ─────────────────────────
    if len(c) > 2:
        interior_max = c[1:-1].max()
        endpoints_max = max(c[0], c[-1])
        if interior_max > endpoints_max + mono_tol:
            return 'bump'

    # ── valley: interior min dips below both endpoints ────────────────────────
    if len(c) > 2:
        interior_min = c[1:-1].min()
        endpoints_min = min(c[0], c[-1])
        if interior_min < endpoints_min - mono_tol:
            return 'valley'

    # ── monotone_up: all successive diffs non-negative ────────────────────────
    diffs = np.diff(c)
    if np.all(diffs >= -mono_tol):
        return 'monotone_up'

    # ── monotone_down: all successive diffs non-positive ─────────────────────
    if np.all(diffs <= mono_tol):
        return 'monotone_down'

    return 'other'


def classify_motif_extended(coeffs, global_std):
    """
    Wraps classify_motif and refines 'other':
      sigmoid_up / sigmoid_down / complex
    """
    base = classify_motif(coeffs, global_std)
    if base != 'other':
        return base
    d2_0 = coeffs[2] - 2*coeffs[1] + coeffs[0]
    d2_1 = coeffs[3] - 2*coeffs[2] + coeffs[1]
    if d2_0 * d2_1 < 0:
        return 'sigmoid_up' if coeffs[-1] > coeffs[0] else 'sigmoid_down'
    return 'complex'


# ── Bernstein activation geometry (degree-3) ──────────────────────────────────
def bern_deriv_roots(coeffs):
    """Analytic real roots of p'(t) for degree-3 Bernstein in t ∈ (0, 1)."""
    if len(coeffs) != 4:
        raise ValueError("bern_deriv_roots only supports degree-3 (4 coefficients)")
    d = np.diff(coeffs)
    A = 3 * (d[0] - 2*d[1] + d[2])
    B = 6 * (d[1] - d[0])
    C = 3 * d[0]
    if abs(A) < 1e-8:
        if abs(B) < 1e-8:
            return []
        t = -C / B
        return [float(t)] if 0.0 < t < 1.0 else []
    disc = B**2 - 4*A*C
    if disc < 0:
        return []
    roots = [(-B + np.sqrt(disc)) / (2*A),
             (-B - np.sqrt(disc)) / (2*A)]
    return sorted([float(r) for r in roots if 0.0 < r < 1.0])


def bern_second_deriv_root(coeffs):
    """
    Root of p''(t) in (0, 1) for degree-3 Bernstein.
    Returns t_inf if root in (0, 1), else None.
    """
    d2_0 = coeffs[2] - 2*coeffs[1] + coeffs[0]
    d2_1 = coeffs[3] - 2*coeffs[2] + coeffs[1]
    denom = d2_0 - d2_1
    if abs(denom) < 1e-8:
        return None
    t_inf = d2_0 / denom
    return float(t_inf) if 0.0 < t_inf < 1.0 else None


def compute_regime_breakpoints(coeffs, motif, delta=REGIME_DELTA):
    """Returns interior t-space breakpoints from activation shape.

    Fine version: always uses uniform grid {0.25, 0.50, 0.75} + derivative
    roots + inflection root, independent of motif.  Monotone neurons get
    4 regimes (3 breakpoints) instead of the old 2 regimes (1 breakpoint).
    """
    if motif == 'flat':
        return []
    pts = set()
    # Fixed uniform grid (always present)
    for k in range(1, N_FIXED_GRID + 1):
        pts.add(k / (N_FIXED_GRID + 1))
    # Derivative roots (local extrema)
    for root in bern_deriv_roots(coeffs):
        if 0.02 < root < 0.98:
            pts.add(root)
    # Second-derivative root (inflection point)
    t_inf = bern_second_deriv_root(coeffs)
    if t_inf is not None and 0.02 < t_inf < 0.98:
        pts.add(t_inf)
    return sorted(pts)


def build_neuron_regimes(model):
    """
    Extract per-neuron regime information from the FIRST hidden Bernstein layer.
    W0: (H, D), b0: (H,), bounds: (H, 2), coeffs: (H, degree+1).
    Flat neurons are skipped.
    """
    W0 = model.layers[0].weight.detach().numpy()
    b0 = model.layers[0].bias.detach().numpy()
    bl = model.get_bern_layers()[0]
    bounds     = bl.input_bounds.detach().numpy()
    coeffs_all = bl.bern_coeffs.detach().numpy()
    global_std = coeffs_all.std()

    neurons = []
    for i in range(W0.shape[0]):
        lo, hi = float(bounds[i, 0]), float(bounds[i, 1])
        motif  = classify_motif_extended(coeffs_all[i], global_std)
        if motif == 'flat':
            continue
        breaks   = compute_regime_breakpoints(coeffs_all[i], motif)
        z_breaks = [lo + t * (hi - lo) for t in breaks]
        # Add lo and hi as explicit breakpoints to separate the
        # out-of-bounds clipped regions from the polynomial interior
        z_breaks = sorted(set([lo] + z_breaks + [hi]))
        z_edges  = [-np.inf] + z_breaks + [np.inf]
        regimes  = [(z_edges[j], z_edges[j+1]) for j in range(len(z_edges) - 1)]
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


# ── Rule evaluation + candidate generation ────────────────────────────────────
def rule_mask(conditions, Z):
    """Boolean mask of rows in Z satisfying all conditions."""
    mask = np.ones(len(Z), dtype=bool)
    for idx, (z_lo, z_hi) in conditions.items():
        lo = z_lo if not np.isinf(z_lo) else -1e18
        hi = z_hi if not np.isinf(z_hi) else  1e18
        mask &= (Z[:, idx] >= lo) & (Z[:, idx] < hi)
    return mask


def score_rule(conditions, Z, y):
    """Evaluate a candidate rule. Returns None if coverage < MIN_COVERAGE."""
    m   = rule_mask(conditions, Z)
    cov = int(m.sum())
    if cov < MIN_COVERAGE:
        return None
    label  = int(np.bincount(y[m]).argmax())
    purity = float((y[m] == label).mean())
    return {
        'conditions': conditions,
        'label':      label,
        'coverage':   cov,
        'purity':     purity,
        'mask':       m,
    }


def _extend_one_level(parents, neurons, Z, y_train):
    """Extend each parent rule with one additional neuron condition."""
    import time as _time
    new_accepted  = []
    for_extension = []
    n_parents = len(parents)
    t_start   = _time.time()
    for pi, parent in enumerate(parents):
        best_pure   = None
        best_impure = None
        for nrn in neurons:
            if nrn['idx'] in parent['conditions']:
                continue
            for regime in nrn['regimes']:
                cond = {**parent['conditions'], nrn['idx']: regime}
                res  = score_rule(cond, Z, y_train)
                if res is None:
                    continue
                if res['purity'] >= PURITY_THRESHOLD:
                    if best_pure is None or res['coverage'] > best_pure['coverage']:
                        best_pure = res
                else:
                    if best_impure is None or res['coverage'] > best_impure['coverage']:
                        best_impure = res
        if best_pure is not None:
            new_accepted.append(best_pure)
        if best_impure is not None:
            for_extension.append(best_impure)
        # Progress ticker every 50 parents or at end
        if (pi + 1) % 50 == 0 or (pi + 1) == n_parents:
            elapsed = _time.time() - t_start
            eta     = elapsed / (pi + 1) * (n_parents - pi - 1)
            print(f"      extend {pi+1}/{n_parents}  pure={len(new_accepted)}  "
                  f"impure={len(for_extension)}  {elapsed:.0f}s elapsed  ETA {eta:.0f}s",
                  flush=True)
    return new_accepted, for_extension


def generate_candidate_rules(neurons, Z, y_train):
    """
    Generate candidate rules up to depth MAX_DEPTH.
    Depth-1: single-neuron regime conditions.
    Depth-2+: greedily extend impure rules with the best additional condition.
    """
    accepted = []

    depth1_impure = []
    for nrn in neurons:
        for regime in nrn['regimes']:
            res = score_rule({nrn['idx']: regime}, Z, y_train)
            if res is None:
                continue
            if res['purity'] >= PURITY_THRESHOLD:
                accepted.append(res)
            else:
                depth1_impure.append(res)

    print(f"    Depth-1: {len(accepted)} pure, {len(depth1_impure)} impure candidates", flush=True)

    impure = depth1_impure
    for depth in range(2, MAX_DEPTH + 1):
        if not impure:
            break
        print(f"    Depth-{depth}: extending {len(impure)} impure rules ...", flush=True)
        new_acc, impure = _extend_one_level(impure, neurons, Z, y_train)
        accepted.extend(new_acc)
        print(f"    Depth-{depth}: {len(new_acc)} pure added, {len(impure)} impure candidates", flush=True)

    return accepted
