"""
extract_rules.py  (MAGIC rule extraction — paper TABLE V)
========================================================
Bernstein-neuron rule extraction + adaptive-k sparsification for MAGIC. Rules are
linear "slab" conditions on first-hidden-layer Bernstein pre-activations (z = w·x + b),
selected by a purity-staged, conflict-penalized greedy cover, then sparsified per rule.

Self-contained: all extraction primitives are local (`rule_extraction_magic`,
`bern_net`, `default_rule_utils`); nothing is imported from sibling repos. Reproduces
the shipped `rule_jsons/*.json` and the metrics rows in `5_fold_results.csv`.

Usage (from the MAGIC/ folder):
    python -u extract_rules.py \
        --ckpt student_model_weights/kd_fc_10x64x32x2_bern_deg3_alpha0.5_T2_lr0.003_wd0.0001_seed42.pth \
        --grid 5 --purity 0.85 --min_cov 5 --depth 2 --conflict_alpha 0.1 --same_cov_alpha 0.1
"""

import argparse
import csv
import json
import os
import pickle
import time

import numpy as np
import torch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

import rule_extraction_magic as ad
from bern_net import FCModel, BernsteinLayer
from default_rule_utils import build_default_rule

MAGIC_FEATURES = [
    'fLength', 'fWidth', 'fSize', 'fConc', 'fConc1',
    'fAsym', 'fM3Long', 'fM3Trans', 'fAlpha', 'fDist',
]
CLASS_NAMES = ['hadron', 'gamma']


# ─── Rule-system evaluation helpers (prediction, fidelity, coverage) ─────────
# Pure functions: rule prediction (early-stop / max-purity) + centroid fallback.

def _apply_centroid_fallback(default_rule, X_val, pred):
    """For uncovered samples (pred == -1), assign class by nearest centroid in X-space."""
    uncov = pred == -1
    if not uncov.any():
        return
    X_uncov = X_val[uncov]
    d0 = np.linalg.norm(X_uncov - default_rule['c0'], axis=1)
    d1 = np.linalg.norm(X_uncov - default_rule['c1'], axis=1)
    pred[uncov] = (d1 < d0).astype(int)


def _predict_with_rules(rules, Z_val, X_val=None, strategy='early_stop'):
    N    = len(Z_val)
    pred = np.full(N, -1, dtype=int)
    default_rule = None
    real_rules = [r for r in rules if r['conditions']]
    for r in rules:
        if not r['conditions']:
            default_rule = r
            break

    if strategy == 'early_stop':
        real_rules_ord = sorted(real_rules, key=lambda r: r.get('coverage', 0), reverse=True)
        remaining = np.ones(N, dtype=bool)
        for r in real_rules_ord:
            if not remaining.any():
                break
            m = ad.rule_mask(r['conditions'], Z_val) & remaining
            pred[m] = r['label']
            remaining[m] = False
    else:  # max_purity
        pred_purity = np.full(N, -1.0, dtype=float)
        for r in real_rules:
            m = ad.rule_mask(r['conditions'], Z_val)
            upgrade = m & (r['purity'] > pred_purity)
            pred[upgrade]        = r['label']
            pred_purity[upgrade] = r['purity']

    if default_rule is not None and X_val is not None:
        _apply_centroid_fallback(default_rule, X_val, pred)
    return pred


def eval_on_val(rules, Z_val, y_val, X_val=None, y_model=None):
    real_rules = [r for r in rules if r['conditions']]
    covered_correct = np.zeros(len(y_val), dtype=bool)
    covered_wrong   = np.zeros(len(y_val), dtype=bool)
    for r in real_rules:
        m = ad.rule_mask(r['conditions'], Z_val)
        covered_correct |= (m & (y_val == r['label']))
        covered_wrong   |= (m & (y_val != r['label']))
    uncovered   = ~covered_correct & ~covered_wrong
    n_conflicts = int((covered_correct & covered_wrong).sum())
    avg_depth   = float(np.mean([len(r['conditions']) for r in real_rules])) if real_rules else 0.0
    pred        = _predict_with_rules(rules, Z_val, X_val)
    rule_accuracy = 100.0 * float((pred == y_val).mean())
    covered_mask  = covered_correct | covered_wrong
    cov_rule_acc  = (100.0 * float(covered_correct[covered_mask].mean())
                     if covered_mask.any() else 0.0)
    if y_model is not None:
        cov_fidelity   = (100.0 * float((pred[covered_mask] == y_model[covered_mask]).mean())
                          if covered_mask.any() else 0.0)
        total_fidelity = 100.0 * float((pred == y_model).mean())
    else:
        cov_fidelity = total_fidelity = None
    return {
        'n_val':                   len(y_val),
        'pct_covered_correct':     100.0 * covered_correct.mean(),
        'pct_covered_wrong':       100.0 * covered_wrong.mean(),
        'pct_uncovered':           100.0 * uncovered.mean(),
        'avg_conditions_per_rule': avg_depth,
        'rule_accuracy':           rule_accuracy,
        'cov_rule_accuracy':       cov_rule_acc,
        'n_conflicts':             n_conflicts,
        'cov_fidelity':            cov_fidelity,
        'total_fidelity':          total_fidelity,
    }


def cascade_greedy_cover(all_rules, y_train, purity_stages, min_cov,
                         conflict_alpha=1.0, min_cov_exp=3.0, same_cov_alpha=0.2):
    """Greedy cover in stages: highest purity first, conflict-penalized scoring."""
    N       = len(y_train)
    n_class = int(y_train.max()) + 1
    uncovered        = np.ones(N, dtype=bool)
    total_covered    = np.zeros(N, dtype=bool)
    covered_by_label = [np.zeros(N, dtype=bool) for _ in range(n_class)]
    selected         = []

    for purity_thresh in sorted(purity_stages, reverse=True):
        stage_min_cov = max(2, round(min_cov * (0.95 / purity_thresh) ** min_cov_exp))
        stage_rules   = [r for r in all_rules if r['purity'] >= purity_thresh]
        n_before      = len(selected)
        n_iter        = 0

        while uncovered.any():
            best, best_score, best_gain = None, -np.inf, 0
            for r in stage_rules:
                gain = int((r['mask'] & uncovered).sum())
                if gain < stage_min_cov:
                    continue
                same_overlap  = int((r['mask'] & covered_by_label[r['label']]).sum())
                other_overlap = int((r['mask'] & (total_covered & ~covered_by_label[r['label']])).sum())
                conflicts     = same_overlap * same_cov_alpha + other_overlap * conflict_alpha
                score         = gain - conflicts
                if score > best_score:
                    best_score, best_gain, best = score, gain, r

            if best is None or best_gain < stage_min_cov or best_score < 0:
                break

            selected.append({**best, 'new_coverage': best_gain, 'stage_purity': purity_thresh})
            uncovered                       &= ~best['mask']
            covered_by_label[best['label']] |= best['mask']
            total_covered                   |= best['mask']
            n_iter += 1
            if n_iter % 20 == 0:
                pct_done = 100.0 * (1 - uncovered.mean())
                print(f"    iter {n_iter:4d}: covered {pct_done:.1f}%  "
                      f"gain={best_gain}  score={best_score:.0f}  "
                      f"label={best['label']}  purity={best['purity']:.3f}", flush=True)

        n_added = len(selected) - n_before
        pct_cov = 100.0 * (1 - uncovered.mean())
        print(f"  Stage purity>={purity_thresh:.2f} min_cov={stage_min_cov} (exp={min_cov_exp:.1f}): "
              f"{len(stage_rules)} eligible  +{n_added} selected  "
              f"total={len(selected)}  covered={pct_cov:.1f}%  "
              f"uncovered={int(uncovered.sum())}", flush=True)

        if not uncovered.any():
            print("  All training samples covered -- stopping cascade early.", flush=True)
            break

    return selected, uncovered


def _sparse_mask(rule, X, b0):
    mask = np.ones(len(X), dtype=bool)
    for nidx, w_sp in rule['sparse_weights'].items():
        z_lo, z_hi = rule['conditions'][nidx]
        z_sp = X @ w_sp + float(b0[nidx])
        lo = z_lo if not np.isinf(z_lo) else -1e18
        hi = z_hi if not np.isinf(z_hi) else  1e18
        mask &= (z_sp >= lo) & (z_sp < hi)
    return mask


def sparsify_rule_min_k(rule, X_train, y_labels, W0, b0, purity_thresh, min_cov):
    """Find minimum k per rule (by |w| magnitude) that meets purity+cov thresholds."""
    if not rule['conditions']:
        return rule, None

    k_candidates = list(range(1, W0.shape[1])) + [W0.shape[1]]
    for k in k_candidates:
        cond_sparse, new_conds = {}, {}
        for nidx, (z_lo, z_hi) in rule['conditions'].items():
            w_i     = W0[nidx]
            top_idx = np.argsort(np.abs(w_i))[::-1][:k]
            w_sp    = np.zeros_like(w_i)
            w_sp[top_idx] = w_i[top_idx]
            cond_sparse[nidx] = w_sp
            new_conds[nidx]   = (z_lo, z_hi)

        sr      = {**rule, 'conditions': new_conds, 'sparse_weights': cond_sparse, 'k_used': k}
        sp_mask = _sparse_mask(sr, X_train, b0)
        cov     = int(sp_mask.sum())
        if cov < min_cov:
            continue
        purity = float((y_labels[sp_mask] == rule['label']).mean())
        if purity < purity_thresh - 1e-6:
            continue
        return {**sr, 'purity': purity, 'coverage': cov}, k

    return None, None


def eval_sparse_on_val(sparse_rules, X_val, y_val, b0, strategy='early_stop', y_model=None):
    real_rules    = [r for r in sparse_rules if     r['conditions']]
    default_rules = [r for r in sparse_rules if not r['conditions']]
    covered_correct = np.zeros(len(y_val), dtype=bool)
    covered_wrong   = np.zeros(len(y_val), dtype=bool)
    for r in real_rules:
        m = _sparse_mask(r, X_val, b0)
        covered_correct |= (m & (y_val == r['label']))
        covered_wrong   |= (m & (y_val != r['label']))
    uncovered   = ~covered_correct & ~covered_wrong
    n_conflicts = int((covered_correct & covered_wrong).sum())
    avg_depth   = float(np.mean([len(r['conditions']) for r in real_rules])) if real_rules else 0.0
    N    = len(y_val)
    pred = np.full(N, -1, dtype=int)
    if strategy == 'early_stop':
        remaining = np.ones(N, dtype=bool)
        real_rules_sorted = sorted(real_rules, key=lambda r: r.get('coverage', 0), reverse=True)
        for r in real_rules_sorted:
            if not remaining.any():
                break
            m = _sparse_mask(r, X_val, b0) & remaining
            pred[m] = r['label']
            remaining[m] = False
    else:  # max_purity
        pred_purity = np.full(N, -1.0)
        for r in real_rules:
            m = _sparse_mask(r, X_val, b0)
            upgrade = m & (r['purity'] > pred_purity)
            pred[upgrade] = r['label']
            pred_purity[upgrade] = r['purity']
    covered_mask = covered_correct | covered_wrong
    cov_rule_acc = (100.0 * float(covered_correct[covered_mask].mean())
                    if covered_mask.any() else 0.0)
    if default_rules:
        _apply_centroid_fallback(default_rules[0], X_val, pred)
    rule_accuracy = 100.0 * float((pred == y_val).mean())
    if y_model is not None:
        cov_fidelity   = (100.0 * float((pred[covered_mask] == y_model[covered_mask]).mean())
                          if covered_mask.any() else 0.0)
        total_fidelity = 100.0 * float((pred == y_model).mean())
    else:
        cov_fidelity = total_fidelity = None
    return {
        'n_rules':             len(real_rules),
        'pct_covered_correct': 100.0 * covered_correct.mean(),
        'pct_covered_wrong':   100.0 * covered_wrong.mean(),
        'pct_uncovered':       100.0 * uncovered.mean(),
        'rule_accuracy':       rule_accuracy,
        'cov_rule_accuracy':   cov_rule_acc,
        'n_conflicts':         n_conflicts,
        'avg_conditions':      avg_depth,
        'cov_fidelity':        cov_fidelity,
        'total_fidelity':      total_fidelity,
    }


def compute_expected_mults(rules, X_val, b0, input_dim):
    real_rules = sorted(
        [r for r in rules if r['conditions']],
        key=lambda r: -r['coverage'],
    )
    if not real_rules:
        return 0.0
    N = len(X_val)
    already_fired    = np.zeros(N, dtype=bool)
    sample_mults     = np.zeros(N, dtype=float)
    cumulative_mults = 0
    for r in real_rules:
        cumulative_mults += r.get('k_used', input_dim) * len(r['conditions'])
        m         = _sparse_mask(r, X_val, b0)
        new_fires = m & ~already_fired
        sample_mults[new_fires] = cumulative_mults
        already_fired |= new_fires
    sample_mults[~already_fired] = cumulative_mults
    return float(sample_mults.mean())


def rule_memory_floats(rules, input_dim):
    real = [r for r in rules if r['conditions']]
    cond_floats = sum(len(r['conditions']) * (r.get('k_used', input_dim) + 2) for r in real)
    rule_floats = len(real) * 2
    return cond_floats + rule_floats + 2 * input_dim   # +2*D for centroid fallback


# ─── Args ─────────────────────────────────────────────────────────────────────

parser = argparse.ArgumentParser()
parser.add_argument('--ckpt', required=True)
parser.add_argument('--grid',    type=int,   default=3,    help='N_FIXED_GRID')
parser.add_argument('--purity',  type=float, default=0.85, help='PURITY_THRESHOLD')
parser.add_argument('--min_cov', type=int,   default=5,    help='MIN_COVERAGE')
parser.add_argument('--depth',   type=int,   default=2,    help='MAX_DEPTH')
parser.add_argument('--conflict_alpha', type=float, default=0.1)
parser.add_argument('--same_cov_alpha', type=float, default=0.5)
parser.add_argument('--min_cov_exp', type=float, default=3.0)
parser.add_argument('--use_model_purity', action=argparse.BooleanOptionalAction, default=True)
parser.add_argument('--teacher-pt', dest='teacher_pt', default='teacher_magic.pt',
                    help='checkpoint providing preprocessor + idx_train/idx_test + X_raw/y_raw '
                         '(default teacher_magic.pt; override for a covariate-restricted shift split)')
args = parser.parse_args()

_ALL_STAGES   = [0.95, 0.90, 0.85, 0.80, 0.75, 0.70]
PURITY_STAGES = [p for p in _ALL_STAGES if p >= args.purity - 1e-6]
if not PURITY_STAGES:
    PURITY_STAGES = [args.purity]

CKPT = os.path.join(SCRIPT_DIR, args.ckpt)

# ─── Hyperparams ──────────────────────────────────────────────────────────────

ad.N_FIXED_GRID     = args.grid
ad.PURITY_THRESHOLD = args.purity
ad.MIN_COVERAGE     = args.min_cov
ad.MAX_DEPTH        = args.depth

# ─── Load model ───────────────────────────────────────────────────────────────

raw  = torch.load(CKPT, map_location='cpu', weights_only=False)
_act = raw.get('activation', 'bern')

model = FCModel(
    layer_sizes=raw['arch'],
    degree=raw['degree'] if raw['degree'] is not None else 3,
    act=_act,
    last_bern=False,
    dropout=0.0,
)
model.load_state_dict(raw['state_dict'])
model.eval()

n_class   = model.layers[-1].weight.shape[0]
n_hidden  = len([m for m in model.layers if isinstance(m, BernsteinLayer)])

mode_str = []
if args.use_model_purity:
    mode_str.append('Mode A: model-purity')
mode_label = ' + '.join(mode_str) if mode_str else 'ground-truth purity only'

print("=" * 72)
print(f"MAGIC Distillation  ({mode_label})")
print("=" * 72)
print(f"  ckpt:     {os.path.basename(CKPT)}")
print(f"  grid={ad.N_FIXED_GRID}  purity={ad.PURITY_THRESHOLD}  "
      f"min_cov={ad.MIN_COVERAGE}  depth={ad.MAX_DEPTH}")
print(f"  cascade stages: {PURITY_STAGES}  conflict_alpha={args.conflict_alpha}  "
      f"same_cov_alpha={args.same_cov_alpha}")
print(f"\n  arch={raw['arch']}  act={_act}  n_class={n_class}  "
      f"n_hidden_layers={n_hidden}")
print(f"  test_acc={raw.get('test_acc', 0):.4f}")

# ─── Load data ────────────────────────────────────────────────────────────────

TEACHER_PT = args.teacher_pt if os.path.isabs(args.teacher_pt) \
             else os.path.join(SCRIPT_DIR, args.teacher_pt)
ckpt_t     = torch.load(TEACHER_PT, map_location='cpu', weights_only=False)
print(f"  split/preprocessor source: {os.path.basename(TEACHER_PT)}")
qt         = ckpt_t['preprocessor']
X_raw      = ckpt_t['X_raw']
y_raw      = ckpt_t['y_raw']
idx_train  = ckpt_t['idx_train']
idx_test   = ckpt_t['idx_test']

X_train = qt.transform(X_raw[idx_train]).astype(np.float32)
X_test  = qt.transform(X_raw[idx_test]).astype(np.float32)
y_train = y_raw[idx_train]
y_test  = y_raw[idx_test]

input_dim = X_train.shape[1]
print(f"\n  Train N={len(X_train)}  Test N={len(X_test)}  input_dim={input_dim}")

# ─── Pre-activations (first layer only) ──────────────────────────────────────

W0 = model.layers[0].weight.detach().numpy()  # (H1, input_dim)
b0 = model.layers[0].bias.detach().numpy()    # (H1,)
Z_train = X_train @ W0.T + b0
Z_test  = X_test  @ W0.T + b0

# NN predictions on test set (for fidelity evaluation)
with torch.no_grad():
    y_model_test = model(torch.tensor(X_test, dtype=torch.float32)).argmax(dim=1).numpy()

# ─── Build neuron regimes (first hidden layer) ───────────────────────────────

neurons  = ad.build_neuron_regimes(model)
n_active = len(neurons)
avg_reg  = float(np.mean([n['n_regimes'] for n in neurons])) if neurons else 0
print(f"\n  Active neurons: {n_active}/{W0.shape[0]}  "
      f"avg regimes/neuron: {avg_reg:.2f}  (first hidden layer, act={_act})")

# ─── Generate candidate rules (ground-truth purity) — with disk cache ────────

_CACHE_DIR  = os.path.join(SCRIPT_DIR, 'candidates_cache')
os.makedirs(_CACHE_DIR, exist_ok=True)
_ckpt_stem  = os.path.splitext(os.path.basename(CKPT))[0]
_cache_key  = f"{_ckpt_stem}_grid{ad.N_FIXED_GRID}_depth{ad.MAX_DEPTH}"
_cache_path = os.path.join(_CACHE_DIR, f"{_cache_key}.pkl")

t0 = time.time()
if os.path.exists(_cache_path):
    print(f"\n  Loading candidates from cache: {os.path.basename(_cache_path)}", flush=True)
    with open(_cache_path, 'rb') as _f:
        all_rules = pickle.load(_f)
    print(f"  Candidates (gt-purity >= {ad.PURITY_THRESHOLD}): {len(all_rules)}  "
          f"(cached, {time.time()-t0:.1f}s)")
else:
    print("\n  Generating candidates ...", flush=True)
    all_rules = ad.generate_candidate_rules(neurons, Z_train, y_train)
    print(f"  Candidates (gt-purity >= {ad.PURITY_THRESHOLD}): {len(all_rules)}")
    with open(_cache_path, 'wb') as _f:
        pickle.dump(all_rules, _f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"  Candidates cached -> {_cache_path}")

total_candidates = len(all_rules)

# ─── Mode A: recompute purity using model predictions ─────────────────────────

if args.use_model_purity:
    print("  Mode A: computing model-prediction purity ...", flush=True)
    with torch.no_grad():
        logits  = model(torch.tensor(X_train, dtype=torch.float32))
        y_model = logits.argmax(dim=1).numpy()

    n_before = len(all_rules)
    model_pure_rules = []
    for r in all_rules:
        pts = r['mask']
        if pts.sum() == 0:
            continue
        model_purity = float((y_model[pts] == r['label']).mean())
        r['model_purity'] = model_purity
        r['gt_purity']    = r['purity']
        if model_purity >= ad.PURITY_THRESHOLD:
            r['purity'] = model_purity
            model_pure_rules.append(r)
    all_rules = model_pure_rules
    print(f"  Candidates after model-purity filter: {len(all_rules)}  "
          f"(removed {n_before - len(all_rules)} with model_purity < {ad.PURITY_THRESHOLD})")
else:
    y_model = y_train
    for r in all_rules:
        r['model_purity'] = None
        r['gt_purity']    = r['purity']

# ─── Cascade greedy set cover ─────────────────────────────────────────────────

selected, uncov_train = cascade_greedy_cover(
    all_rules, y_train, PURITY_STAGES, ad.MIN_COVERAGE,
    conflict_alpha=args.conflict_alpha,
    min_cov_exp=args.min_cov_exp,
    same_cov_alpha=args.same_cov_alpha)
train_covered_pct = 100.0 * (1 - uncov_train.mean())
print(f"  Selected rules: {len(selected)}")
print(f"  Train covered:  {train_covered_pct:.1f}%")

default_rule = build_default_rule(X_train, y_train, uncov_train)
print(f"  Default rule:   centroid (X-space)  "
      f"purity={default_rule['purity']:.3f}  n_uncov={default_rule['n_uncovered']}")
selected = selected + [default_rule]

# ─── Evaluate dense rules on test ────────────────────────────────────────────

test_stats = eval_on_val(selected, Z_test, y_test, X_test, y_model=y_model_test)

rule_accuracy_test = test_stats['rule_accuracy']
test_covered_pct   = 100.0 - test_stats['pct_uncovered']
test_correct_pct   = test_stats['pct_covered_correct']
test_wrong_pct     = test_stats['pct_covered_wrong']
n_conflicts        = test_stats['n_conflicts']
avg_cond           = test_stats['avg_conditions_per_rule']

print(f"\n  Dense rules (test set):")
print(f"  {'covered%':20s}  {test_covered_pct:>10.1f}")
print(f"  {'cov_rule_acc%':20s}  {test_stats['cov_rule_accuracy']:>10.2f}")
print(f"  {'cov_fidelity%':20s}  {test_stats['cov_fidelity']:>10.2f}")
print(f"  {'total_fidelity%':20s}  {test_stats['total_fidelity']:>10.2f}")
print(f"  {'total_acc%':20s}  {rule_accuracy_test:>10.2f}")
print(f"  {'conflicts':20s}  {n_conflicts:>10}")
print(f"  {'avg_cond/rule':20s}  {avg_cond:>10.2f}")

# ─── Adaptive-k sparsification ────────────────────────────────────────────────

print(f"\n{'='*72}")
print(f"  Adaptive-k sparsification  (base_purity={ad.PURITY_THRESHOLD}  "
      f"min_cov>={ad.MIN_COVERAGE})")

rules_to_sparse = [r for r in selected if r['conditions']]
default_rules   = [r for r in selected if not r['conditions']]

y_labels = y_model if args.use_model_purity else y_train

adaptive_sparse = []
k_used_list     = []
n_fallback      = 0

for rule in rules_to_sparse:
    _min_cov_rule = max(ad.MIN_COVERAGE, int(0.9 * rule.get('coverage', ad.MIN_COVERAGE)))
    sp_rule, k = sparsify_rule_min_k(
        rule, X_train, y_labels, W0, b0,
        rule.get('stage_purity', ad.PURITY_THRESHOLD),
        _min_cov_rule)
    if sp_rule is None:
        _full_sparse_weights = {nidx: W0[nidx] for nidx in rule['conditions']}
        adaptive_sparse.append({**rule, 'k_used': W0.shape[1],
                                 'sparse_weights': _full_sparse_weights})
        k_used_list.append(W0.shape[1])
        n_fallback += 1
    else:
        adaptive_sparse.append(sp_rule)
        k_used_list.append(k)

adaptive_with_default = adaptive_sparse + default_rules

print(f"  Rules: {len(rules_to_sparse)} dense -> {len(adaptive_sparse)} adaptive-k  "
      f"(sparsified={len(adaptive_sparse)-n_fallback}  fallback-dense={n_fallback})")

if k_used_list:
    k_arr = np.array(k_used_list)
    print(f"  k distribution:  min={k_arr.min()}  "
          f"p25={int(np.percentile(k_arr, 25))}  "
          f"median={int(np.median(k_arr))}  "
          f"p75={int(np.percentile(k_arr, 75))}  "
          f"max={k_arr.max()}  mean={k_arr.mean():.1f}")

ts_sp_es = eval_sparse_on_val(adaptive_with_default, X_test, y_test, b0,
                               strategy='early_stop', y_model=y_model_test)
ts_sp_mp = eval_sparse_on_val(adaptive_with_default, X_test, y_test, b0,
                               strategy='max_purity', y_model=y_model_test)

wst_mults_dense  = W0.shape[0] * W0.shape[1]
wst_mults_sparse = sum(r.get('k_used', W0.shape[1]) * len(r['conditions'])
                       for r in adaptive_sparse)
exp_mults        = compute_expected_mults(adaptive_with_default, X_test, b0, input_dim)
mem_dense        = rule_memory_floats(selected, input_dim)
mem_sparse       = rule_memory_floats(adaptive_with_default, input_dim)

print(f"\n  Sparse rules (test set):")
print(f"  {'':25s}  {'early_stop':>10}  {'max_purity':>10}")
print(f"  {'n_rules':25s}  {ts_sp_es['n_rules']:>10}  {ts_sp_mp['n_rules']:>10}")
print(f"  {'test_cov%':25s}  {100.0-ts_sp_es['pct_uncovered']:>10.1f}  {100.0-ts_sp_mp['pct_uncovered']:>10.1f}")
print(f"  {'cov_rule_acc%':25s}  {ts_sp_es['cov_rule_accuracy']:>10.2f}  {ts_sp_mp['cov_rule_accuracy']:>10.2f}")
print(f"  {'cov_fidelity%':25s}  {ts_sp_es['cov_fidelity']:>10.2f}  {ts_sp_mp['cov_fidelity']:>10.2f}")
print(f"  {'total_fidelity%':25s}  {ts_sp_es['total_fidelity']:>10.2f}  {ts_sp_mp['total_fidelity']:>10.2f}")
print(f"  {'total_acc%':25s}  {ts_sp_es['rule_accuracy']:>10.2f}  {ts_sp_mp['rule_accuracy']:>10.2f}")
print(f"  {'conflicts':25s}  {ts_sp_es['n_conflicts']:>10}  {ts_sp_mp['n_conflicts']:>10}")
print(f"  {'wst_mults_sparse':25s}  {wst_mults_sparse:>10}  {wst_mults_sparse:>10}")
print(f"  {'exp_mults':25s}  {exp_mults:>10.1f}  {'N/A':>10}")
print(f"  {'mem_floats':25s}  {mem_sparse:>10}  {mem_sparse:>10}")

elapsed = time.time() - t0
print(f"\n  Elapsed: {elapsed:.1f}s")
print("=" * 72)

# ─── Save results to CSV ──────────────────────────────────────────────────────

RESULTS_CSV = os.path.join(SCRIPT_DIR, 'extraction_results.csv')
CSV_FIELDS  = [
    'ckpt', 'arch', 'degree', 'n_hidden_layers', 'model_test_acc',
    'n_fixed_grid', 'purity_threshold', 'min_coverage', 'max_depth',
    'conflict_alpha', 'same_cov_alpha',
    'n_active_neurons', 'avg_regimes_per_neuron', 'total_candidates',
    'n_rules', 'train_covered_pct',
    'test_covered_pct', 'test_covered_rule_acc',
    'test_rule_acc_no_sparse', 'test_rule_acc_mp_sparse',
    'test_cov_fidelity', 'test_total_fidelity',
    'n_conflicts', 'avg_conditions',
    'wst_mult_before_sparse', 'wst_mult_after_sparse',
    'mem_floats_before_sparse', 'mem_floats_after_sparse',
]

n_real_dense = len([r for r in selected if r['conditions']])
row = {
    'ckpt':                    os.path.basename(CKPT),
    'arch':                    str(raw['arch']),
    'degree':                  raw['degree'],
    'n_hidden_layers':         n_hidden,
    'model_test_acc':          round(raw.get('test_acc', 0), 4),
    'n_fixed_grid':            ad.N_FIXED_GRID,
    'purity_threshold':        ad.PURITY_THRESHOLD,
    'min_coverage':            ad.MIN_COVERAGE,
    'max_depth':               ad.MAX_DEPTH,
    'conflict_alpha':          args.conflict_alpha,
    'same_cov_alpha':          args.same_cov_alpha,
    'n_active_neurons':        n_active,
    'avg_regimes_per_neuron':  round(avg_reg, 2),
    'total_candidates':        total_candidates,
    'n_rules':                 n_real_dense,
    'train_covered_pct':       round(train_covered_pct, 2),
    'test_covered_pct':        round(test_covered_pct, 2),
    'test_covered_rule_acc':   round(test_stats['cov_rule_accuracy'], 2),
    'test_rule_acc_no_sparse': round(rule_accuracy_test, 2),
    'test_rule_acc_mp_sparse': round(ts_sp_mp['rule_accuracy'], 2),
    'test_cov_fidelity':       round(test_stats['cov_fidelity'], 2),
    'test_total_fidelity':     round(test_stats['total_fidelity'], 2),
    'n_conflicts':             n_conflicts,
    'avg_conditions':          round(avg_cond, 2),
    'wst_mult_before_sparse':  wst_mults_dense,
    'wst_mult_after_sparse':   wst_mults_sparse,
    'mem_floats_before_sparse': mem_dense,
    'mem_floats_after_sparse':  mem_sparse,
}

write_header = not os.path.exists(RESULTS_CSV)
with open(RESULTS_CSV, 'a', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
    if write_header:
        writer.writeheader()
    writer.writerow(row)
print(f"  Results appended -> {RESULTS_CSV}")

# ─── Save rules to JSON ──────────────────────────────────────────────────────

feature_names = MAGIC_FEATURES


def _band(lo_z, hi_z, w_i, b_i):
    lo_x = None if np.isinf(lo_z) and lo_z < 0 else float(lo_z - b_i)
    hi_x = None if np.isinf(hi_z) and hi_z > 0 else float(hi_z - b_i)
    abs_w = np.abs(w_i)
    top_idx = np.argsort(abs_w)[::-1][:5]
    top_features = [
        {'feature': feature_names[j] if j < len(feature_names) else f'f{j}',
         'weight': round(float(w_i[j]), 6)}
        for j in top_idx if abs_w[j] > 1e-9
    ]
    return {
        'band_lo':       lo_x,
        'band_hi':       hi_x,
        'weight_vector': [round(float(v), 6) for v in w_i],
        'top5_features': top_features,
    }


def _serialize_rules(rules_list, is_sparse_rules):
    out = []
    for r in rules_list:
        if not r['conditions']:
            continue
        conditions_x = []
        for nidx, (lo_z, hi_z) in r['conditions'].items():
            b_i = float(b0[nidx])
            w_i = r['sparse_weights'][nidx] if (is_sparse_rules and 'sparse_weights' in r) else W0[nidx]
            cond = {'neuron': int(nidx), **_band(lo_z, hi_z, w_i, b_i)}
            if is_sparse_rules and 'sparse_weights' in r:
                nonzero_idx = np.where(np.abs(w_i) > 1e-9)[0]
                cond['sparse_weights'] = {
                    (feature_names[j] if j < len(feature_names) else f'f{j}'): round(float(w_i[j]), 6)
                    for j in nonzero_idx
                }
                cond['k']            = int(nonzero_idx.shape[0])
                cond['sparsity_pct'] = round(100.0 * (1 - nonzero_idx.shape[0] / len(w_i)), 1)
                cond['weight_vector'] = [round(float(v), 6) for v in w_i]
            conditions_x.append(cond)
        entry = {
            'label':       r['label'],
            'label_name':  CLASS_NAMES[r['label']],
            'coverage':    r['coverage'],
            'purity':      round(r['purity'], 4),
            'gt_purity':   round(r.get('gt_purity', r['purity']), 4),
            'n_cond':      len(conditions_x),
            'conditions':  conditions_x,
        }
        out.append(entry)
    return out


def _save_rules_json(rules_list, is_sparse_rules, k_tag, test_s):
    ckpt_stem  = os.path.splitext(os.path.basename(CKPT))[0]
    config_tag = (f"g{ad.N_FIXED_GRID}_p{int(ad.PURITY_THRESHOLD*100)}_"
                  f"mc{ad.MIN_COVERAGE}_d{ad.MAX_DEPTH}_{k_tag}")
    out_dir = os.path.join(SCRIPT_DIR, 'rule_jsons')
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"magic_rules_{ckpt_stem}_{config_tag}.json")
    desc_kind  = f'top-{k_tag} sparse' if is_sparse_rules else 'dense'
    real_rules = [r for r in rules_list if r['conditions']]
    with open(path, 'w') as f:
        json.dump({
            'description':   f'{desc_kind} rules in input (X) space. '
                             'band_lo/band_hi are bias-absorbed: check band_lo <= w*x <= band_hi. '
                             'null=unbounded. Fallback: nearest-centroid in X-space.',
            'feature_names': feature_names,
            'config': {
                'ckpt': os.path.basename(CKPT), 'arch': raw['arch'],
                'n_hidden_layers': n_hidden,
                'n_fixed_grid': ad.N_FIXED_GRID,
                'purity_threshold': ad.PURITY_THRESHOLD,
                'min_coverage': ad.MIN_COVERAGE, 'max_depth': ad.MAX_DEPTH, 'k': k_tag,
                'use_model_purity': args.use_model_purity,
            },
            'fallback': {
                'type':              'centroid_x_space',
                'c0':                default_rule['c0'].tolist(),
                'c1':                default_rule['c1'].tolist(),
                'purity':            round(default_rule['purity'], 4),
                'n_uncovered_train': default_rule['n_uncovered'],
            },
            'metrics': {
                'n_rules':          len(real_rules),
                'test_covered_pct': round(100.0 - test_s['pct_uncovered'], 2),
                'test_rule_acc':    round(test_s['rule_accuracy'], 2),
                'n_conflicts':      test_s['n_conflicts'],
                'avg_conditions':   round(test_s.get('avg_conditions', avg_cond), 2),
            },
            'rules': _serialize_rules(rules_list, is_sparse_rules),
        }, f, indent=2)
    print(f"  Rules saved -> {path}  ({len(real_rules)} rules + default, {k_tag})")


_save_rules_json(selected, False, 'dense', test_stats)
_save_rules_json(adaptive_with_default, True, 'adaptive', ts_sp_mp)
