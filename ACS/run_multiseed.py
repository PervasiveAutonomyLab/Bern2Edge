"""
run_multiseed.py
================
Multi-seed driver for paper **TABLE XI** (distribution-shift robustness on ACS
Income, training on CA-2018). For each seed it runs the full pipeline
(teacher -> Bernstein + ReLU students -> network-purity rules -> CART fallback ->
evaluation across conditions) and aggregates mean ± std across seeds.

Scope: **CART fallback only** and the single penalty combo used by TABLE XI
(alpha_sc = 0.1, alpha_conf = 0.2). Everything the table needs, nothing more.

Caching (under results/_multiseed_cache/seed<seed>/):
  * models.pt      — trained teacher + Bernstein/ReLU students (the shipped
                     artifact; loaded to reproduce TABLE XI exactly);
  * candidates.pkl — alpha-independent candidate rules, regenerated
                     deterministically from the loaded student on a cache miss
                     (not shipped — ~12 MB/seed);
  * combo_*.json   — the per-combo evaluated rule-system metrics (shipped).

Reuses Bern2Edge shared code via `_compat` (ad / kd_train_models / MLP) +
`models.FCModel`; ACS data lives in `data.acs_income`; the teacher trainer is
`train_teacher.fit_relu_mlp`. No Bern2Edge shared module is modified.

Outputs:
  results/metrics_multiseed_raw.csv  — one row per (seed x system x condition)
  results/metrics_multiseed_agg.csv  — mean/std across seeds
  results/RESULTS_multiseed.md       — mean±std tables (all conditions)
Render TABLE XI ({MS,WY,WV}+{2019,2021,2022}, Δ) with:  python make_table_xi.py
"""

import argparse
import csv
import json
import os
import pickle
import sys
import tempfile

import numpy as np
import torch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = os.path.dirname(SCRIPT_DIR)          # ACS -> Bern2Edge
sys.path.insert(0, REPO_DIR)
sys.path.insert(0, SCRIPT_DIR)

from sklearn.model_selection import train_test_split            # noqa: E402
from sklearn.tree import DecisionTreeClassifier                 # noqa: E402
from _compat import ad, M, make_loader, MLP, kd_train_models    # noqa: E402
from bern2edge.models import FCModel                                      # noqa: E402
from bern2edge.data import acs_income as acs_data                         # noqa: E402
import train_teacher as TT                                      # noqa: E402

CONDITIONS = acs_data.conditions()
GEO_KEYS  = acs_data.geo_keys()
TEMP_KEYS = acs_data.temporal_keys()

# ── Rule-extraction + fallback config (fixed; matches the paper) ──────────────
N_FIXED_GRID, MAX_DEPTH = 5, 3
PURITY_STAGES = [(1.00, 2), (0.95, 3), (0.90, 5)]
BASE_PURITY, GEN_MIN_COV = 0.90, 2
CART_DEPTH = 4


def combo_tag(sc, cf):
    return f"sc{sc}_cf{cf}"


# ── Small helpers (self-contained) ────────────────────────────────────────────
@torch.no_grad()
def net_acc(model, X, y, device):
    model.to(device).eval()
    p = model(torch.tensor(X, dtype=torch.float32, device=device)).argmax(1).cpu().numpy()
    return 100.0 * (p == y).mean()


def slim_rules(selected):
    return [{'conditions': r['conditions'], 'label': int(r['label']),
             'purity': float(r['purity']), 'coverage': int(r.get('coverage', 0))}
            for r in selected]


def train_cart(X_unc, y_unc, X_all, y_all, seed):
    if len(X_unc) and len(np.unique(y_unc)) > 1:
        Xc, yc = X_unc, y_unc
    else:                                   # guard empty / single-class residual
        Xc, yc = X_all, y_all
    clf = DecisionTreeClassifier(max_depth=CART_DEPTH, random_state=seed)
    clf.fit(Xc, yc)
    return clf


def rule_system(selected, W0, b0, X, fb_pred, y):
    """Return (coverage_pct, covered_acc, total_acc) for the rules + CART system."""
    Z = X @ W0.T + b0
    covered = np.zeros(len(X), dtype=bool)
    for r in selected:
        covered |= ad.rule_mask(r['conditions'], Z)
    pred = M._predict_with_rules(selected, Z, X, fallback_pred=fb_pred)
    cov_pct = 100.0 * covered.mean()
    cov_acc = 100.0 * (pred[covered] == y[covered]).mean() if covered.any() else float('nan')
    tot_acc = 100.0 * (pred == y).mean()
    return cov_pct, cov_acc, tot_acc


def cascade_cover(all_rules, y, sc, cf):
    return M.cascade_greedy_cover(all_rules, y, PURITY_STAGES,
                                  conflict_alpha=cf, same_cov_alpha=sc)


def _build_eval_sets(splits, scaler, Xte, yte):
    eval_sets = {}
    for lab, key in CONDITIONS:
        if key == 'ca2018':
            eval_sets[lab] = (Xte, yte)
        else:
            X, y = splits[key]
            eval_sets[lab] = (acs_data.apply_scaler(scaler, X), y)
    return eval_sets


def _train_seed_models(seed, ca_X, ca_y, input_dim, device):
    """Train the per-seed teacher + bern/relu students; return a cacheable blob
    (models + scaler + split indices, NO candidates)."""
    print(f"[seed {seed}] training teacher + students")
    np.random.seed(seed); torch.manual_seed(seed)
    idx = np.arange(len(ca_y))
    idx_tr, idx_te = train_test_split(idx, test_size=0.2, random_state=seed, stratify=ca_y)
    scaler = acs_data.fit_scaler(ca_X[idx_tr])
    Xtr_all = acs_data.apply_scaler(scaler, ca_X[idx_tr]); ytr_all = ca_y[idx_tr]
    Xte = acs_data.apply_scaler(scaler, ca_X[idx_te]);     yte = ca_y[idx_te]
    rel_tr, rel_val = train_test_split(np.arange(len(idx_tr)), test_size=0.1,
                                       random_state=seed, stratify=ytr_all)
    X_tr, X_val = Xtr_all[rel_tr], Xtr_all[rel_val]
    y_tr, y_val = ytr_all[rel_tr], ytr_all[rel_val]

    teacher = TT.fit_relu_mlp(X_tr, y_tr, X_val, y_val, input_dim, device,
                              seed=seed, verbose=False)
    tmp = tempfile.mkdtemp(prefix=f'ms_seed{seed}_')
    df = kd_train_models(
        architectures=[[input_dim, 32, 2]], activations=['bern', 'relu'],
        alphas=[0.5], temps=[2.0], degree=3, last_bern=False,
        learning_rate=3e-3, weight_decay=1e-4,
        train_loader=make_loader(X_tr, y_tr, 256, True),
        val_loader=make_loader(X_val, y_val, 8192, False),
        test_loader=make_loader(Xte, yte, 8192, False),
        class_names=['<=50K', '>50K'], epochs=100, seed=seed,
        teacher=teacher.to(device), file_name=os.path.join(tmp, 'kd.csv'),
        range_penalty_weight=0.0, fold_idx=None, save_dir=tmp)
    paths = {r['activation']: r['model_path'] for _, r in df.iterrows()}
    bern_raw = torch.load(paths['bern'], map_location='cpu', weights_only=False)
    relu_raw = torch.load(paths['relu'], map_location='cpu', weights_only=False)

    return {'bern_raw': bern_raw, 'relu_raw': relu_raw,
            'teacher_state': {k: v.cpu() for k, v in teacher.state_dict().items()},
            'd_layers': TT.D_LAYERS, 'dropout': TT.DROPOUT,
            'scaler': scaler, 'idx_tr': idx_tr, 'idx_te': idx_te}


def load_or_build_seed(seed, splits, ca_X, ca_y, device, cache_root):
    """Return the per-seed artifact dict (models + baselines + lazy candidates)."""
    sdir = os.path.join(cache_root, f'seed{seed}')
    os.makedirs(sdir, exist_ok=True)
    models_p = os.path.join(sdir, 'models.pt')
    cand_p   = os.path.join(sdir, 'candidates.pkl')

    input_dim = ca_X.shape[1]
    if os.path.exists(models_p):
        print(f"[seed {seed}] loading cached models")
        blob = torch.load(models_p, map_location='cpu', weights_only=False)
    else:
        blob = _train_seed_models(seed, ca_X, ca_y, input_dim, device)
        torch.save(blob, models_p)

    scaler, idx_tr, idx_te = blob['scaler'], blob['idx_tr'], blob['idx_te']

    def mk(raw, act):
        m = FCModel(raw['arch'], raw.get('degree') or 3, act=act, last_bern=False, dropout=0.0)
        m.load_state_dict(raw['state_dict']); m.eval(); return m
    bern, relu = mk(blob['bern_raw'], 'bern'), mk(blob['relu_raw'], 'relu')
    teacher = MLP(input_dim, blob['d_layers'], dropout=blob['dropout'], n_classes=2)
    teacher.load_state_dict(blob['teacher_state']); teacher.eval()

    X_train = acs_data.apply_scaler(scaler, ca_X[idx_tr]); y_train = ca_y[idx_tr]
    Xte = acs_data.apply_scaler(scaler, ca_X[idx_te]);     yte = ca_y[idx_te]
    eval_sets = _build_eval_sets(splits, scaler, Xte, yte)
    baselines = {}
    for name, model in [('ReLU FC teacher', teacher),
                        ('Bernstein student', bern),
                        ('ReLU student (same size)', relu)]:
        baselines[name] = {lab: float(net_acc(model, *eval_sets[lab], device))
                           for lab, _ in CONDITIONS}
    bern.cpu()
    W0 = bern.layers[0].weight.detach().cpu().numpy()
    b0 = bern.layers[0].bias.detach().cpu().numpy()
    with torch.no_grad():
        y_model_train = bern(torch.tensor(X_train, dtype=torch.float32)).argmax(1).numpy()

    # candidate rules loaded lazily: only a combo-cache miss needs them.
    _cand = {}

    def get_all_rules():
        if 'r' not in _cand:
            if os.path.exists(cand_p):
                _cand['r'] = pickle.load(open(cand_p, 'rb'))
            else:
                print(f"[seed {seed}] generating candidate rules")
                ad.N_FIXED_GRID, ad.PURITY_THRESHOLD = N_FIXED_GRID, BASE_PURITY
                ad.MIN_COVERAGE, ad.MAX_DEPTH = GEN_MIN_COV, MAX_DEPTH
                rules = ad.generate_candidate_rules(ad.build_neuron_regimes(bern),
                                                    X_train @ W0.T + b0, y_model_train)
                pickle.dump(rules, open(cand_p, 'wb'), protocol=pickle.HIGHEST_PROTOCOL)
                _cand['r'] = rules
        return _cand['r']

    return {'seed': seed, 'W0': W0, 'b0': b0,
            'X_train': X_train, 'y_train': y_train,
            'y_model_train': y_model_train, 'get_all_rules': get_all_rules,
            'eval_sets': eval_sets, 'baselines': baselines, 'input_dim': X_train.shape[1]}


def eval_combo(art, sc, cf, cache_root):
    """Cascade cover + CART fallback + per-condition rule systems for one combo
    (cached per seed+combo as combo_<tag>.json)."""
    sdir = os.path.join(cache_root, f"seed{art['seed']}")
    cpath = os.path.join(sdir, f"combo_{combo_tag(sc, cf)}.json")
    if os.path.exists(cpath):
        rec = json.load(open(cpath))
        if all(lab in rec['conds'] for lab, _ in CONDITIONS):
            return rec
        print(f"    [seed {art['seed']}] combo {combo_tag(sc, cf)} cache stale — recomputing")

    selected, uncov = cascade_cover(art['get_all_rules'](), art['y_model_train'], sc, cf)
    selected = slim_rules(selected)
    train_cov = 100.0 * (1 - uncov.mean())
    X_train, y_train = art['X_train'], art['y_train']
    cart = train_cart(X_train[uncov], y_train[uncov], X_train, y_train, art['seed'])
    conds = {}
    for lab, _ in CONDITIONS:
        Xs, y = art['eval_sets'][lab]
        c = rule_system(selected, art['W0'], art['b0'], Xs, cart.predict(Xs).astype(int), y)
        conds[lab] = {'cart': [float(v) for v in c]}
    rec = {'sc': sc, 'cf': cf, 'n_rules': len(selected),
           'train_cov': float(train_cov), 'conds': conds}
    json.dump(rec, open(cpath, 'w'))
    return rec


def _avg_over(per_cond, keyset):
    vals = [per_cond[lab] for lab, key in CONDITIONS if key in keyset]
    return float(np.mean(vals)) if vals else float('nan')


def geo_avg(per_cond):  return _avg_over(per_cond, GEO_KEYS)
def temp_avg(per_cond): return _avg_over(per_cond, TEMP_KEYS)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--seeds', type=int, nargs='+', default=[42, 1, 2, 3, 4])
    ap.add_argument('--combos', nargs='+', default=['0.1,0.2'],
                    help='"sc,cf" penalty combos (TABLE XI uses 0.1,0.2)')
    ap.add_argument('--data-dir', default=os.path.join(SCRIPT_DIR, 'data'))
    ap.add_argument('--out-dir',  default=os.path.join(SCRIPT_DIR, 'results'))
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    cache_root = os.path.join(args.out_dir, '_multiseed_cache')
    os.makedirs(cache_root, exist_ok=True)

    device = acs_data.get_device()
    combos = [tuple(float(x) for x in c.split(',')) for c in args.combos]
    print(f"Multi-seed: seeds={args.seeds}  combos={combos}  geo={acs_data.GEO_STATES}")
    splits = acs_data.download_acs(args.data_dir)
    ca_X, ca_y = splits['ca2018']

    records = []
    for seed in args.seeds:
        art = load_or_build_seed(seed, splits, ca_X, ca_y, device, cache_root)
        combo_recs = {}
        for sc, cf in combos:
            rec = eval_combo(art, sc, cf, cache_root)
            combo_recs[combo_tag(sc, cf)] = rec
            print(f"  [seed {seed}] sc{sc}_cf{cf}: rules={rec['n_rules']} "
                  f"train_cov={rec['train_cov']:.1f}%", flush=True)
        records.append({'seed': seed, 'baselines': art['baselines'], 'combos': combo_recs})

    aggregate_and_write(records, combos, args.out_dir)


def _mean_std(vals):
    a = np.asarray(vals, dtype=float)
    return float(a.mean()), (float(a.std(ddof=1)) if len(a) > 1 else 0.0)


def aggregate_and_write(records, combos, out_dir):
    conds = [lab for lab, _ in CONDITIONS]
    seeds = [r['seed'] for r in records]

    # ── raw CSV (one row per seed x system x condition) ───────────────────────
    raw_rows = []
    for r in records:
        for name, accs in r['baselines'].items():
            for lab in conds:
                raw_rows.append({'seed': r['seed'], 'alpha_sc': '', 'alpha_conf': '',
                                 'system': name, 'condition': lab, 'coverage_pct': '',
                                 'covered_acc': '', 'total_acc': round(accs[lab], 2)})
        for rec in r['combos'].values():
            for lab in conds:
                cov, cacc, tot = rec['conds'][lab]['cart']
                raw_rows.append({'seed': r['seed'], 'alpha_sc': rec['sc'],
                                 'alpha_conf': rec['cf'], 'system': 'Rules + CART fallback',
                                 'condition': lab, 'coverage_pct': round(cov, 2),
                                 'covered_acc': round(cacc, 2), 'total_acc': round(tot, 2)})
    raw_path = os.path.join(out_dir, 'metrics_multiseed_raw.csv')
    with open(raw_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['seed', 'alpha_sc', 'alpha_conf', 'system',
                                          'condition', 'coverage_pct', 'covered_acc', 'total_acc'])
        w.writeheader(); w.writerows(raw_rows)

    # ── aggregated CSV ────────────────────────────────────────────────────────
    agg = {}
    for name in ['ReLU FC teacher', 'Bernstein student', 'ReLU student (same size)']:
        for lab in conds:
            agg[(name, '', lab)] = _mean_std([r['baselines'][name][lab] for r in records])
        agg[(name, '', 'GEO-AVG')] = _mean_std([geo_avg(r['baselines'][name]) for r in records])
        agg[(name, '', 'TEMP-AVG')] = _mean_std([temp_avg(r['baselines'][name]) for r in records])
    for sc, cf in combos:
        ctag = combo_tag(sc, cf)
        for idx, mname in [(0, 'coverage_pct'), (1, 'covered_acc'), (2, 'total_acc')]:
            for lab in conds:
                agg[(mname, ctag, lab)] = _mean_std(
                    [r['combos'][ctag]['conds'][lab]['cart'][idx] for r in records])
            for avg_lab, avg_fn in [('GEO-AVG', geo_avg), ('TEMP-AVG', temp_avg)]:
                ps = [avg_fn({l: r['combos'][ctag]['conds'][l]['cart'][idx] for l in conds})
                      for r in records]
                agg[(mname, ctag, avg_lab)] = _mean_std(ps)

    agg_path = os.path.join(out_dir, 'metrics_multiseed_agg.csv')
    with open(agg_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['system', 'combo', 'condition', 'metric', 'mean', 'std', 'n_seeds'])
        for (name, ctag, lab), (m, s) in agg.items():
            metric = 'total_acc' if ctag == '' else name
            sysname = name if ctag == '' else 'Rules + CART fallback'
            w.writerow([sysname, ctag, lab, metric, round(m, 2), round(s, 2), len(seeds)])

    write_md(os.path.join(out_dir, 'RESULTS_multiseed.md'), records, combos, agg, conds, seeds)
    print(f"\nWrote {raw_path}\nWrote {agg_path}\n"
          f"Wrote {os.path.join(out_dir, 'RESULTS_multiseed.md')}\n"
          f"Render TABLE XI:  python make_table_xi.py")


def write_md(path, records, combos, agg, conds, seeds):
    geo = ', '.join(acs_data.GEO_STATES)
    temp = ', '.join(str(y) for y in acs_data.TEMPORAL_YEARS)
    disp = conds + ['GEO-AVG', 'TEMP-AVG']
    _avg_keys = {'GEO-AVG': GEO_KEYS, 'TEMP-AVG': TEMP_KEYS}

    def cell(name, ctag, lab):
        if (name, ctag, lab) not in agg:
            return '—'
        m, s = agg[(name, ctag, lab)]
        return f"{m:.2f}±{s:.2f}"

    def stat(ctag, lab, kind):
        idx = {'cov': 0, 'cacc': 1, 'tot': 2}[kind]
        def val(r):
            cd = r['combos'][ctag]['conds']
            if lab in _avg_keys:
                return float(np.mean([cd[l]['cart'][idx]
                                      for l, k in CONDITIONS if k in _avg_keys[lab]]))
            return cd[lab]['cart'][idx]
        m, s = _mean_std([val(r) for r in records])
        return f"{m:.2f}±{s:.2f}"

    L = []
    L.append("# ACS Income — Distribution-Shift Robustness (multi-seed, CART fallback)\n")
    L.append(f"Mean ± std (sample std) over **{len(seeds)} seeds** ({seeds}). Percentages. "
             f"GEO-AVG = mean over the geo-shift states ({geo}); TEMP-AVG = mean over the "
             f"temporal-shift years ({temp}); both computed per seed then aggregated. Rule "
             "systems use the **CART** fallback (depth 4) for uncovered inputs.\n")
    L.append("Preprocessing: raw-code categoricals + standardize-all; `RELSHIPP→RELP` "
             "semantic recode on the temporal splits (the 2020 ACS 1-Year PUMS was not "
             "released — COVID-19). Rule extraction: grid=5, depth=3, p90 cascade, dense.\n")

    L.append("## Network baselines (total accuracy %)\n")
    L.append("| System | " + " | ".join(disp) + " |")
    L.append("|" + "---|" * (len(disp) + 1))
    for name in ['ReLU FC teacher', 'Bernstein student', 'ReLU student (same size)']:
        L.append(f"| {name} | " + " | ".join(cell(name, '', l) for l in disp) + " |")
    L.append("")

    for sc, cf in combos:
        ctag = combo_tag(sc, cf)
        nr = _mean_std([r['combos'][ctag]['n_rules'] for r in records])
        tc = _mean_std([r['combos'][ctag]['train_cov'] for r in records])
        L.append(f"## Rules (α_sc={sc}, α_conf={cf}, CART fb)  "
                 f"(rules {nr[0]:.0f}±{nr[1]:.0f}, train_cov {tc[0]:.1f}±{tc[1]:.1f}%)\n")
        L.append("| Metric | " + " | ".join(disp) + " |")
        L.append("|" + "---|" * (len(disp) + 1))
        L.append("| Coverage % | " + " | ".join(stat(ctag, l, 'cov') for l in disp) + " |")
        L.append("| Covered acc % | " + " | ".join(stat(ctag, l, 'cacc') for l in disp) + " |")
        L.append("| Total acc % | " + " | ".join(stat(ctag, l, 'tot') for l in disp) + " |")
        L.append("")
    L.append("## Reproduce\n\nRun from the repository root:\n\n```bash\npython ACS/run_multiseed.py "
             f"--seeds {' '.join(map(str, seeds))} "
             f"--combos {' '.join(combo_tag(*c).replace('sc','').replace('_cf',',') for c in combos)}\n"
             "python ACS/make_table_xi.py     # render TABLE XI\n```\n")
    with open(path, 'w') as f:
        f.write("\n".join(L) + "\n")


if __name__ == '__main__':
    main()
