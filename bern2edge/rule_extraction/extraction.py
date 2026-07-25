"""
extraction.py
-------------
Turn a trained Bernstein student network into a small set of interpretable rules.

Pipeline (all measured against the *network's own predictions*, so the rules
distil the network, not the raw labels):

  1. Build per-neuron activation regimes         (bern_regimes.build_neuron_regimes)
  2. Generate candidate rules up to `max_depth`   conditions of the form
     ``band_lo <= w . x < band_hi``               (generate_candidate_rules)
  3. Select a covering subset with a purity       (cascade_greedy_cover)
     cascade + conflict/same-coverage penalties
  4. Handle the leftover uncovered inputs with a  (build_default_rule_lr / CART /
     fallback: LR | CART tree | network | small NN  network / small Bernstein net)
  5. (optional) sparsify each condition to top-k weights
  6. Score coverage / accuracy / fidelity and serialise to JSON

`ExtractionConfig` collects every hyperparameter; its defaults reproduce the
paper's dense CART-fallback setting (grid=5, depth=3, p90 cascade, CART depth 4).

Ported and consolidated from the research pipeline
(run_all_architectures_network_purity.py + activation_distill.py +
default_rule_utils.py); only the code paths actually used to produce the paper
results are kept, and module-global state is replaced by an explicit config.
"""

from dataclasses import dataclass, field
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

from .bern_regimes import build_neuron_regimes


# ─── Configuration ───────────────────────────────────────────────────────────

@dataclass
class ExtractionConfig:
    """Every rule-extraction hyperparameter. Defaults reproduce paper TABLE III
    (dense rules, CART fallback, grid=5, depth=3, p90 purity cascade)."""

    # Regime geometry (fixed grid resolution shared with the visualiser).
    n_fixed_grid: int = 5

    # Candidate generation: max conditions per rule, and the purity cascade of
    # (purity_threshold, min_new_coverage) stages processed strictest-first.
    max_depth: int = 3
    purity_stages: Tuple[Tuple[float, int], ...] = ((1.0, 2), (0.95, 3), (0.90, 5))

    # Greedy-cover penalties: discourage re-covering points already claimed by a
    # same-label rule (same_cov_alpha) or an opposite-label rule (conflict_alpha).
    conflict_alpha: float = 0.1
    same_cov_alpha: float = 0.5

    # Fallback for inputs no rule covers.
    #   'lr'       logistic regression folded to X-space (network-free, cheap)
    #   'tree'     shallow CART (comparisons only) -- the paper's headline setting
    #   'network'  defer to the original Bernstein network's prediction
    #   'small_nn' a fresh small Bernstein net trained on the uncovered residual
    fallback_mode: str = 'tree'
    fb_train_labels: str = 'gt'          # 'gt' = ground truth, 'network' = model output

    # CART fallback depth (kept shallow for interpretability + cheap hardware).
    fb_tree_depth: int = 4

    # Small-Bernstein-net fallback hyperparameters (fallback_mode == 'small_nn').
    fb_hidden: int = 4
    fb_degree: int = 3
    fb_lr: float = 5e-3
    fb_batch_size: int = 256
    fb_warmup_epochs: int = 2
    fb_max_epochs: int = 400
    fb_patience: int = 40
    fb_range_pen_w: float = 1e-2

    # Condition sparsification: None = dense (full weight vector per condition);
    # int k = keep only each condition's top-k weights by magnitude.
    sparsity_k: Optional[int] = None

    @property
    def base_purity(self) -> float:
        """Lowest cascade purity = the candidate-generation purity threshold."""
        return min(p for p, _ in self.purity_stages)

    @property
    def gen_min_cov(self) -> int:
        """Smallest stage min-coverage = the candidate-generation min coverage."""
        return min(c for _, c in self.purity_stages)


@dataclass
class ExtractionResult:
    """Everything a run produces: the rule set, its metrics, and the artefacts
    the caller needs to persist (JSON dict + optional fallback sidecars)."""
    rules_json: dict                     # the full rules_float.json content
    metrics: dict                        # numeric columns for the results CSV
    selected: list                       # selected real rules (in-memory)
    default_rule: dict                   # the fallback descriptor
    fb_model: object = None              # trained small Bernstein net (small_nn)
    fb_tree: object = None               # fitted DecisionTreeClassifier (tree)
    fb_train: object = None              # fallback predictions on train (int array)
    fb_test: object = None               # fallback predictions on test (int array)


# ─── Rule masks and scoring ──────────────────────────────────────────────────

def rule_mask(conditions, Z):
    """Boolean mask of rows in Z (pre-activations) satisfying all conditions.

    conditions maps neuron index -> (z_lo, z_hi) half-open band on that neuron's
    pre-activation z = w . x + b.  +/- inf bounds mean unbounded on that side.
    """
    mask = np.ones(len(Z), dtype=bool)
    for idx, (z_lo, z_hi) in conditions.items():
        lo = z_lo if not np.isinf(z_lo) else -1e18
        hi = z_hi if not np.isinf(z_hi) else 1e18
        mask &= (Z[:, idx] >= lo) & (Z[:, idx] < hi)
    return mask


def score_rule(conditions, Z, y, min_coverage):
    """Evaluate a candidate rule; return None if it covers < min_coverage points.

    The rule's label is the majority class among covered points and its purity is
    the fraction of covered points with that label (measured against `y`, which
    for distillation is the network's own predictions).
    """
    m = rule_mask(conditions, Z)
    cov = int(m.sum())
    if cov < min_coverage:
        return None
    label = int(np.bincount(y[m]).argmax())
    purity = float((y[m] == label).mean())
    return {'conditions': conditions, 'label': label,
            'coverage': cov, 'purity': purity, 'mask': m}


# ─── Candidate rule generation ───────────────────────────────────────────────

def _extend_one_level(parents, neurons, Z, y, purity_threshold, min_coverage):
    """Extend each impure parent rule by one extra neuron condition, keeping the
    highest-coverage pure extension (accepted) and the highest-coverage impure
    one (queued for further extension)."""
    new_accepted, for_extension = [], []
    for parent in parents:
        best_pure, best_impure = None, None
        for nrn in neurons:
            if nrn['idx'] in parent['conditions']:
                continue
            for regime in nrn['regimes']:
                cond = {**parent['conditions'], nrn['idx']: regime}
                res = score_rule(cond, Z, y, min_coverage)
                if res is None:
                    continue
                if res['purity'] >= purity_threshold:
                    if best_pure is None or res['coverage'] > best_pure['coverage']:
                        best_pure = res
                else:
                    if best_impure is None or res['coverage'] > best_impure['coverage']:
                        best_impure = res
        if best_pure is not None:
            new_accepted.append(best_pure)
        if best_impure is not None:
            for_extension.append(best_impure)
    return new_accepted, for_extension


def generate_candidate_rules(neurons, Z, y, max_depth, purity_threshold, min_coverage):
    """Generate candidate rules up to `max_depth` conditions.

    Depth 1 tries every single-neuron regime; deeper levels greedily extend the
    still-impure rules with the best additional condition.  Returns all rules
    that reached the purity threshold at any depth.
    """
    accepted, impure = [], []
    for nrn in neurons:
        for regime in nrn['regimes']:
            res = score_rule({nrn['idx']: regime}, Z, y, min_coverage)
            if res is None:
                continue
            (accepted if res['purity'] >= purity_threshold else impure).append(res)
    print(f"    depth 1: {len(accepted)} pure, {len(impure)} impure", flush=True)

    for depth in range(2, max_depth + 1):
        if not impure:
            break
        new_acc, impure = _extend_one_level(
            impure, neurons, Z, y, purity_threshold, min_coverage)
        accepted.extend(new_acc)
        print(f"    depth {depth}: +{len(new_acc)} pure, {len(impure)} impure", flush=True)

    return accepted


# ─── Greedy covering with a purity cascade ───────────────────────────────────

def cascade_greedy_cover(all_rules, y, purity_stages, conflict_alpha, same_cov_alpha):
    """Greedy set cover over purity stages (strictest purity first).

    Within a stage, repeatedly pick the rule maximising
        (new points covered) - same_cov_alpha * (same-label re-coverage)
                             - conflict_alpha  * (opposite-label re-coverage)
    among rules meeting the stage purity threshold and min-coverage.  Preferring
    exact rules first, then relaxing purity while demanding more coverage, keeps
    the rule set small and low-conflict.  Returns (selected_rules, uncovered_mask).
    """
    N = len(y)
    n_class = int(y.max()) + 1
    uncovered = np.ones(N, dtype=bool)
    total_covered = np.zeros(N, dtype=bool)
    covered_by_label = [np.zeros(N, dtype=bool) for _ in range(n_class)]
    selected = []

    for purity_thresh, stage_min_cov in sorted(purity_stages, key=lambda s: -s[0]):
        stage_rules = [r for r in all_rules if r['purity'] >= purity_thresh]
        n_before = len(selected)

        while uncovered.any():
            best, best_score, best_gain = None, -np.inf, 0
            for r in stage_rules:
                gain = int((r['mask'] & uncovered).sum())
                if gain < stage_min_cov:
                    continue
                same_overlap = int((r['mask'] & covered_by_label[r['label']]).sum())
                other_overlap = int((r['mask'] &
                                     (total_covered & ~covered_by_label[r['label']])).sum())
                score = gain - same_overlap * same_cov_alpha - other_overlap * conflict_alpha
                if score > best_score:
                    best_score, best_gain, best = score, gain, r

            if best is None or best_gain < stage_min_cov or best_score < 0:
                break

            selected.append({**best, 'new_coverage': best_gain, 'stage_purity': purity_thresh})
            uncovered &= ~best['mask']
            covered_by_label[best['label']] |= best['mask']
            total_covered |= best['mask']

        pct_cov = 100.0 * (1 - uncovered.mean())
        print(f"    stage purity>={purity_thresh:.2f} min_cov={stage_min_cov}: "
              f"+{len(selected) - n_before} rules, covered {pct_cov:.1f}%", flush=True)
        if not uncovered.any():
            break

    return selected, uncovered


# ─── Optional fixed-k sparsification (keep every rule) ───────────────────────

def _sparse_mask(rule, X, b0):
    """Boolean mask for a sparsified rule: each condition uses its top-k weight
    vector directly on X (rather than the dense pre-activation Z)."""
    mask = np.ones(len(X), dtype=bool)
    for nidx, w_sp in rule['sparse_weights'].items():
        z_lo, z_hi = rule['conditions'][nidx]
        z_sp = X @ w_sp + float(b0[nidx])
        lo = z_lo if not np.isinf(z_lo) else -1e18
        hi = z_hi if not np.isinf(z_hi) else 1e18
        mask &= (z_sp >= lo) & (z_sp < hi)
    return mask


def sparsify_rule_fixed_k_keepall(rule, X_train, y, W0, b0, k):
    """Keep only each condition's top-k weights by magnitude; keep the rule
    regardless of the resulting purity.  Purity/coverage are recomputed against
    the sparse approximation (vs `y` = network output)."""
    cond_sparse, new_conds = {}, {}
    for nidx, (z_lo, z_hi) in rule['conditions'].items():
        w_i = W0[nidx]
        top_idx = np.argsort(np.abs(w_i))[::-1][:k]
        w_sp = np.zeros_like(w_i)
        w_sp[top_idx] = w_i[top_idx]
        cond_sparse[nidx] = w_sp
        new_conds[nidx] = (z_lo, z_hi)
    sr = {**rule, 'conditions': new_conds, 'sparse_weights': cond_sparse, 'k_used': k}
    sp_mask = _sparse_mask(sr, X_train, b0)
    cov = int(sp_mask.sum())
    purity = float((y[sp_mask] == rule['label']).mean()) if cov > 0 else 0.0
    return {**sr, 'purity': purity, 'coverage': cov}


# ─── Default rule and fallbacks for uncovered inputs ─────────────────────────

def build_default_rule_lr(X_train, y_train, uncov_mask, W0, b0):
    """Logistic-regression catch-all, folded from Z-space back into X-space.

    Trains LR on Z = X @ W0.T + b0 (all training points, labels = network output)
    then folds the weights so inference needs no Z:
        w_eff = W0.T @ w_lr,   b_eff = w_lr . b0 + b_lr,   predict 1 iff w_eff.x + b_eff > 0.
    Purity is reported on the uncovered training points only.
    """
    N = len(y_train)
    Z_train = X_train @ W0.T + b0
    lr = LogisticRegression(max_iter=1000, C=1.0, solver='lbfgs')
    lr.fit(Z_train, y_train)
    w_lr = lr.coef_[0]
    b_lr = float(lr.intercept_[0])
    w_eff = W0.T @ w_lr
    b_eff = float(w_lr @ b0) + b_lr

    if uncov_mask.sum() == 0:
        purity = 1.0
    else:
        pred = (X_train[uncov_mask] @ w_eff + b_eff > 0).astype(int)
        purity = float((pred == y_train[uncov_mask]).mean())

    return {'conditions': {}, 'w_eff': w_eff, 'b_eff': b_eff,
            'w_lr': w_lr, 'b_lr': b_lr, 'purity': purity,
            'coverage': N, 'n_uncovered': int(uncov_mask.sum())}


def train_fallback_bern(X_uncov, y_uncov, input_dim, cfg, device='cpu'):
    """Train a small `fb_hidden`-wide Bernstein net on the residual points.

    Warmup with wide bounds -> calibrate bounds from data -> fine-tune with early
    stopping on a held-out split.  Returns (model, best_val_acc).
    """
    from ..models import FCModel
    layer_sizes = [input_dim, cfg.fb_hidden, 2]
    n_uncov = len(y_uncov)

    stratify = y_uncov if (n_uncov >= 10 and len(np.unique(y_uncov)) > 1) else None
    X_tr, X_vl, y_tr, y_vl = train_test_split(
        X_uncov, y_uncov, test_size=0.2, stratify=stratify, random_state=42)

    train_loader = DataLoader(
        TensorDataset(torch.tensor(X_tr, dtype=torch.float32),
                      torch.tensor(y_tr, dtype=torch.long)),
        batch_size=cfg.fb_batch_size, shuffle=True)
    val_loader = DataLoader(
        TensorDataset(torch.tensor(X_vl, dtype=torch.float32),
                      torch.tensor(y_vl, dtype=torch.long)),
        batch_size=cfg.fb_batch_size)

    model = FCModel(layer_sizes, degree=cfg.fb_degree, act='bern',
                    last_bern=False, dropout=0.0).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.fb_lr)
    criterion = nn.CrossEntropyLoss()

    # Phase 1: warmup with deliberately wide bounds.
    for bl in model.get_bern_layers():
        bl.input_bounds.data[..., 0] = -5.0
        bl.input_bounds.data[..., 1] = 5.0
        bl.use_bounds = True
    for _ in range(cfg.fb_warmup_epochs):
        model.train()
        for X, yb in train_loader:
            X, yb = X.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(X), yb) + cfg.fb_range_pen_w * model.range_penalty()
            loss.backward()
            optimizer.step()

    # Phase 2: calibrate Bernstein bounds from the data distribution.
    for layer_idx in range(len(model.get_bern_layers())):
        model.calibrate_one_bern_layer_from_data(
            train_loader, device, layer_idx, p_lo=0.01, p_hi=0.99, min_width=0.2)

    # Phase 3: fine-tune with early stopping on validation accuracy.
    best_val_acc, best_state, no_improve = 0.0, None, 0
    for _ in range(cfg.fb_max_epochs):
        model.train()
        for X, yb in train_loader:
            X, yb = X.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(X), yb) + cfg.fb_range_pen_w * model.range_penalty()
            loss.backward()
            optimizer.step()
        model.eval()
        correct = total = 0
        with torch.no_grad():
            for X, yb in val_loader:
                X, yb = X.to(device), yb.to(device)
                correct += (model(X).argmax(1) == yb).sum().item()
                total += len(yb)
        val_acc = correct / total if total else 0.0
        if val_acc > best_val_acc:
            best_val_acc, no_improve = val_acc, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            no_improve += 1
        if no_improve >= cfg.fb_patience:
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return model, best_val_acc


# ─── Prediction and evaluation ───────────────────────────────────────────────

def _apply_fallback(pred, fallback_pred, default_rule, X):
    """Fill still-uncovered points (pred == -1). A supplied fallback prediction
    array (network / tree / small_nn) takes precedence; otherwise apply the LR
    default rule directly on X."""
    uncov = pred == -1
    if not uncov.any():
        return
    if fallback_pred is not None:
        pred[uncov] = fallback_pred[uncov]
    elif default_rule is not None and X is not None:
        scores = X[uncov] @ default_rule['w_eff'] + default_rule['b_eff']
        pred[uncov] = (scores > 0).astype(int)


def _predict_dense(rules, Z, X, fallback_pred):
    """Predict with dense rules (highest-coverage rule wins), then fallback."""
    N = len(Z)
    pred = np.full(N, -1, dtype=int)
    default_rule = next((r for r in rules if not r['conditions']), None)
    real_rules = [r for r in rules if r['conditions']]
    remaining = np.ones(N, dtype=bool)
    for r in sorted(real_rules, key=lambda r: r.get('coverage', 0), reverse=True):
        if not remaining.any():
            break
        m = rule_mask(r['conditions'], Z) & remaining
        pred[m] = r['label']
        remaining[m] = False
    _apply_fallback(pred, fallback_pred, default_rule, X)
    return pred


def _coverage_stats(covered_correct, covered_wrong, pred, y, y_model):
    """Shared metric assembly for dense and sparse evaluation."""
    uncovered = ~covered_correct & ~covered_wrong
    covered_mask = covered_correct | covered_wrong
    cov_rule_acc = (100.0 * float(covered_correct[covered_mask].mean())
                    if covered_mask.any() else 0.0)
    rule_accuracy = 100.0 * float((pred == y).mean())
    if y_model is not None:
        cov_fidelity = (100.0 * float((pred[covered_mask] == y_model[covered_mask]).mean())
                        if covered_mask.any() else 0.0)
        total_fidelity = 100.0 * float((pred == y_model).mean())
    else:
        cov_fidelity = total_fidelity = None
    return {
        'pct_uncovered':     100.0 * uncovered.mean(),
        'rule_accuracy':     rule_accuracy,
        'cov_rule_accuracy': cov_rule_acc,
        'n_conflicts':       int((covered_correct & covered_wrong).sum()),
        'cov_fidelity':      cov_fidelity,
        'total_fidelity':    total_fidelity,
    }


def eval_on_val(rules, Z, y, X, y_model, fallback_pred):
    """Evaluate dense rules on a data split. Returns coverage/accuracy/fidelity."""
    real_rules = [r for r in rules if r['conditions']]
    covered_correct = np.zeros(len(y), dtype=bool)
    covered_wrong = np.zeros(len(y), dtype=bool)
    for r in real_rules:
        m = rule_mask(r['conditions'], Z)
        covered_correct |= (m & (y == r['label']))
        covered_wrong |= (m & (y != r['label']))
    avg_depth = (float(np.mean([len(r['conditions']) for r in real_rules]))
                 if real_rules else 0.0)
    pred = _predict_dense(rules, Z, X, fallback_pred)
    stats = _coverage_stats(covered_correct, covered_wrong, pred, y, y_model)
    stats['avg_conditions_per_rule'] = avg_depth
    return stats


def eval_sparse_on_val(rules, X, y, b0, y_model, fallback_pred):
    """Evaluate sparsified rules: masks use sparse weights over X (not Z)."""
    real_rules = [r for r in rules if r['conditions']]
    default_rule = next((r for r in rules if not r['conditions']), None)
    covered_correct = np.zeros(len(y), dtype=bool)
    covered_wrong = np.zeros(len(y), dtype=bool)
    for r in real_rules:
        m = _sparse_mask(r, X, b0)
        covered_correct |= (m & (y == r['label']))
        covered_wrong |= (m & (y != r['label']))
    avg_depth = (float(np.mean([len(r['conditions']) for r in real_rules]))
                 if real_rules else 0.0)

    N = len(y)
    pred = np.full(N, -1, dtype=int)
    remaining = np.ones(N, dtype=bool)
    for r in sorted(real_rules, key=lambda r: r.get('coverage', 0), reverse=True):
        if not remaining.any():
            break
        m = _sparse_mask(r, X, b0) & remaining
        pred[m] = r['label']
        remaining[m] = False
    _apply_fallback(pred, fallback_pred, default_rule, X)
    stats = _coverage_stats(covered_correct, covered_wrong, pred, y, y_model)
    stats['avg_conditions_per_rule'] = avg_depth
    return stats


# ─── JSON serialization ──────────────────────────────────────────────────────

def _band(lo_z, hi_z, w_i, b_i, feature_names):
    """Serialize one condition's band. Bands are bias-absorbed into X-space:
    the rule fires when band_lo <= w . x < band_hi (null = unbounded)."""
    lo_x = None if np.isinf(lo_z) and lo_z < 0 else float(lo_z - b_i)
    hi_x = None if np.isinf(hi_z) and hi_z > 0 else float(hi_z - b_i)
    abs_w = np.abs(w_i)
    top_idx = np.argsort(abs_w)[::-1][:5]
    top_features = [
        {'feature': feature_names[j] if j < len(feature_names) else f'f{j}',
         'weight': round(float(w_i[j]), 6)}
        for j in top_idx if abs_w[j] > 1e-9
    ]
    return {'band_lo': lo_x, 'band_hi': hi_x,
            'weight_vector': [round(float(v), 6) for v in w_i],
            'top5_features': top_features}


def serialize_rules(rules_list, W0, b0, feature_names):
    """Serialize selected rules in X-space. Sparsified rules additionally record
    their nonzero sparse_weights."""
    out = []
    for r in rules_list:
        if not r['conditions']:
            continue
        conditions_x = []
        for nidx, (lo_z, hi_z) in r['conditions'].items():
            b_i = float(b0[nidx])
            w_i = r['sparse_weights'][nidx] if 'sparse_weights' in r else W0[nidx]
            cond = {'neuron': int(nidx), **_band(lo_z, hi_z, w_i, b_i, feature_names)}
            if 'sparse_weights' in r:
                nz = np.where(np.abs(w_i) > 1e-9)[0]
                cond['sparse_weights'] = {
                    (feature_names[j] if j < len(feature_names) else f'f{j}'):
                        round(float(w_i[j]), 6) for j in nz}
                cond['k'] = int(nz.shape[0])
                cond['sparsity_pct'] = round(100.0 * (1 - nz.shape[0] / len(w_i)), 1)
            conditions_x.append(cond)
        out.append({
            'label':      r['label'],
            'label_name': '<=50K' if r['label'] == 0 else '>50K',
            'coverage':   r['coverage'],
            'purity':     round(r['purity'], 4),
            'gt_purity':  round(r.get('gt_purity', r['purity']), 4),
            'n_cond':     len(conditions_x),
            'conditions': conditions_x,
        })
    return out


def _fallback_json(default_rule, cfg, fb_ckpt):
    """Serialize the fallback descriptor for the active fallback_mode."""
    if cfg.fallback_mode == 'network':
        return {'type': 'neural_network',
                'description': 'uncovered samples are predicted by the original '
                               'network (the rules wrap the network).',
                'n_uncovered_train': default_rule.get('n_uncovered', 0)}
    if cfg.fallback_mode == 'small_nn':
        return {'type': 'small_bern_nn',
                'description': f'a fresh 14x{cfg.fb_hidden}x2 Bernstein net (deg '
                              f'{cfg.fb_degree}) trained on the uncovered residual.',
                'arch': [None, cfg.fb_hidden, 2], 'degree': cfg.fb_degree,
                'train_set': 'uncovered', 'trained_on': cfg.fb_train_labels,
                'val_acc': round(default_rule.get('fb_val_acc', 0.0), 4),
                'checkpoint': fb_ckpt,
                'n_uncovered_train': default_rule.get('n_uncovered', 0)}
    if cfg.fallback_mode == 'tree':
        return {'type': 'decision_tree',
                'description': f'a shallow CART (max_depth={cfg.fb_tree_depth}) trained '
                              'on the uncovered residual; comparisons only, no multiplies.',
                'max_depth': cfg.fb_tree_depth, 'trained_on': cfg.fb_train_labels,
                'n_nodes': default_rule.get('tree_n_nodes', 0),
                'n_leaves': default_rule.get('tree_n_leaves', 0),
                'depth': default_rule.get('tree_depth', 0),
                'n_uncovered_train': default_rule.get('n_uncovered', 0)}
    return {'type': 'lr_x_space', 'trained_on': 'model_output',
            'w_eff': [round(float(v), 8) for v in default_rule['w_eff']],
            'b_eff': round(float(default_rule['b_eff']), 8),
            'purity': round(default_rule['purity'], 4),
            'n_uncovered_train': default_rule['n_uncovered']}


def build_rules_json(rules_with_default, W0, b0, default_rule, cfg,
                     ckpt_name, arch, feature_names, test_stats, fb_ckpt=None):
    """Assemble the rules_float.json dict (matching the research schema)."""
    real_rules = [r for r in rules_with_default if r['conditions']]
    sparse_note = (f"Fixed-k={cfg.sparsity_k} sparsification (keep-all): each condition "
                   "uses only its top-k weights, recorded in 'sparse_weights'."
                   if cfg.sparsity_k is not None else
                   'No sparsification: each condition uses the full first-layer weight_vector.')
    return {
        'description': 'Rules in input (X) space, purity measured against the network '
                       'output. band_lo/band_hi are bias-absorbed: check '
                       f'band_lo <= w*x <= band_hi. null=unbounded. {sparse_note} '
                       f'Fallback mode: {cfg.fallback_mode}.',
        'feature_names': list(feature_names),
        'config': {
            'ckpt': ckpt_name,
            'arch': arch,
            'n_fixed_grid': cfg.n_fixed_grid,
            'base_purity': cfg.base_purity,
            'purity_stages': [[p, c] for p, c in cfg.purity_stages],
            'max_depth': cfg.max_depth,
            'conflict_alpha': cfg.conflict_alpha,
            'same_cov_alpha': cfg.same_cov_alpha,
            'use_model_purity': True,
            'fallback_mode': cfg.fallback_mode,
            'sparsified': cfg.sparsity_k is not None,
            'sparsity_k': cfg.sparsity_k,
        },
        'fallback': _fallback_json(default_rule, cfg, fb_ckpt),
        'metrics': {
            'n_rules': len(real_rules),
            'test_covered_pct': round(100.0 - test_stats['pct_uncovered'], 2),
            'test_rule_acc': round(test_stats['rule_accuracy'], 2),
            'test_fidelity': (round(test_stats['total_fidelity'], 2)
                              if test_stats['total_fidelity'] is not None else None),
            'n_conflicts': test_stats['n_conflicts'],
            'avg_conditions': round(test_stats.get('avg_conditions_per_rule', 0), 2),
        },
        'rules': serialize_rules(rules_with_default, W0, b0, feature_names),
    }


# ─── Orchestration ───────────────────────────────────────────────────────────

def prepare_context(model, X_train, y_gt_train, X_test, y_gt_test, cfg):
    """Precompute everything shared across (same_cov_alpha, conflict_alpha) combos
    for one checkpoint: first-layer weights, pre-activations, network predictions,
    neuron regimes, and the candidate rule set (generated against the network).
    Returns a context dict consumed by `extract_rules`.
    """
    W0 = model.layers[0].weight.detach().cpu().numpy()
    b0 = model.layers[0].bias.detach().cpu().numpy()
    Z_train = X_train @ W0.T + b0
    Z_test = X_test @ W0.T + b0

    with torch.no_grad():
        y_model_train = model(torch.tensor(X_train, dtype=torch.float32)).argmax(1).numpy()
        y_model_test = model(torch.tensor(X_test, dtype=torch.float32)).argmax(1).numpy()

    neurons = build_neuron_regimes(model, cfg.n_fixed_grid)
    n_active = len(neurons)
    avg_reg = float(np.mean([n['n_regimes'] for n in neurons])) if neurons else 0.0
    print(f"  active neurons: {n_active}/{W0.shape[0]}  avg regimes: {avg_reg:.2f}", flush=True)

    print(f"  generating candidates (network-purity >= {cfg.base_purity}) ...", flush=True)
    all_rules = generate_candidate_rules(
        neurons, Z_train, y_model_train,
        cfg.max_depth, cfg.base_purity, cfg.gen_min_cov)
    # Annotate each candidate with its ground-truth purity (for reporting only).
    for r in all_rules:
        m = r['mask']
        r['gt_purity'] = float((y_gt_train[m] == r['label']).mean()) if m.sum() else 0.0
    print(f"  candidates: {len(all_rules)}", flush=True)

    return {
        'model': model, 'W0': W0, 'b0': b0,
        'X_train': X_train, 'y_gt_train': y_gt_train,
        'X_test': X_test, 'y_gt_test': y_gt_test,
        'Z_train': Z_train, 'Z_test': Z_test,
        'y_model_train': y_model_train, 'y_model_test': y_model_test,
        'neurons': neurons, 'n_active': n_active, 'avg_reg': avg_reg,
        'all_rules': all_rules, 'input_dim': int(W0.shape[1]),
    }


def extract_rules(cfg, ckpt_name, arch, feature_names, *, context=None,
                  model=None, X_train=None, y_gt_train=None,
                  X_test=None, y_gt_test=None):
    """Run selection + fallback + evaluation for one config.

    Either pass a precomputed `context` (from `prepare_context`, reused across
    penalty combos) or the raw (model, X_train, y_gt_train, X_test, y_gt_test)
    and it will be built here.  Returns an `ExtractionResult`.
    """
    if context is None:
        context = prepare_context(model, X_train, y_gt_train, X_test, y_gt_test, cfg)
    c = context
    W0, b0 = c['W0'], c['b0']
    y_model_train, y_model_test = c['y_model_train'], c['y_model_test']

    # 1. Select a covering rule set against the network's predictions.
    selected, uncov_train = cascade_greedy_cover(
        c['all_rules'], y_model_train, cfg.purity_stages,
        cfg.conflict_alpha, cfg.same_cov_alpha)
    train_covered_pct = 100.0 * (1 - uncov_train.mean())
    print(f"  selected {len(selected)} rules, train covered {train_covered_pct:.1f}%", flush=True)

    # 2. Build the LR default rule (always) and the active fallback predictions.
    default_rule = build_default_rule_lr(c['X_train'], y_model_train, uncov_train, W0, b0)
    fb_labels_train = c['y_gt_train'] if cfg.fb_train_labels == 'gt' else y_model_train
    fb_model = fb_tree = fb_train = fb_test = None

    if cfg.fallback_mode == 'network':
        fb_train, fb_test = y_model_train, y_model_test
    elif cfg.fallback_mode == 'small_nn':
        X_fb, y_fb = c['X_train'][uncov_train], fb_labels_train[uncov_train]
        fb_model, fb_val_acc = train_fallback_bern(X_fb, y_fb, c['input_dim'], cfg)
        with torch.no_grad():
            fb_train = fb_model(torch.tensor(c['X_train'], dtype=torch.float32)).argmax(1).numpy()
            fb_test = fb_model(torch.tensor(c['X_test'], dtype=torch.float32)).argmax(1).numpy()
        default_rule['fb_val_acc'] = fb_val_acc
    elif cfg.fallback_mode == 'tree':
        # Shallow CART on the uncovered residual (guard empty / single-class).
        if uncov_train.any() and len(np.unique(fb_labels_train[uncov_train])) > 1:
            X_fb, y_fb = c['X_train'][uncov_train], fb_labels_train[uncov_train]
        else:
            X_fb, y_fb = c['X_train'], fb_labels_train
        clf = DecisionTreeClassifier(max_depth=cfg.fb_tree_depth, random_state=42)
        clf.fit(X_fb, y_fb)
        fb_tree = clf
        fb_train = clf.predict(c['X_train']).astype(int)
        fb_test = clf.predict(c['X_test']).astype(int)
        default_rule['tree_n_nodes'] = int(clf.tree_.node_count)
        default_rule['tree_n_leaves'] = int(clf.get_n_leaves())
        default_rule['tree_depth'] = int(clf.get_depth())
    # else 'lr': the LR default rule handles uncovered points directly.

    # 3. Optional sparsification, then evaluate on test + train.
    if cfg.sparsity_k is not None:
        eval_rules = [sparsify_rule_fixed_k_keepall(r, c['X_train'], y_model_train,
                                                    W0, b0, cfg.sparsity_k)
                      for r in selected if r['conditions']]
        rules_with_default = eval_rules + [default_rule]
        test_stats = eval_sparse_on_val(rules_with_default, c['X_test'], c['y_gt_test'],
                                        b0, y_model_test, fb_test)
    else:
        rules_with_default = selected + [default_rule]
        test_stats = eval_on_val(rules_with_default, c['Z_test'], c['y_gt_test'],
                                 c['X_test'], y_model_test, fb_test)

    print(f"  test: acc(vs GT)={test_stats['rule_accuracy']:.2f}%  "
          f"cov={100.0 - test_stats['pct_uncovered']:.1f}%  "
          f"cov_acc={test_stats['cov_rule_accuracy']:.2f}%  "
          f"fidelity={test_stats['total_fidelity']:.2f}%", flush=True)

    # 4. Serialize JSON + assemble the metrics row.
    rules_json = build_rules_json(rules_with_default, W0, b0, default_rule, cfg,
                                  ckpt_name, arch, feature_names, test_stats)
    metrics = {
        'n_active_neurons':       c['n_active'],
        'avg_regimes_per_neuron': round(c['avg_reg'], 2),
        'total_candidates':       len(c['all_rules']),
        'n_rules':                len(selected),
        'train_covered_pct':      round(train_covered_pct, 2),
        'test_covered_pct':       round(100.0 - test_stats['pct_uncovered'], 2),
        'test_covered_rule_acc':  round(test_stats['cov_rule_accuracy'], 2),
        'test_rule_acc':          round(test_stats['rule_accuracy'], 2),
        'test_fidelity':          round(test_stats['total_fidelity'], 2),
        'test_covered_fidelity':  round(test_stats['cov_fidelity'], 2),
        'n_conflicts':            test_stats['n_conflicts'],
        'avg_conditions':         round(test_stats['avg_conditions_per_rule'], 2),
    }

    return ExtractionResult(rules_json=rules_json, metrics=metrics,
                            selected=selected, default_rule=default_rule,
                            fb_model=fb_model, fb_tree=fb_tree,
                            fb_train=fb_train, fb_test=fb_test)
