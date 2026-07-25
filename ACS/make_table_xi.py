"""
make_table_xi.py
================
Render paper **TABLE XI** ("Distribution-shift robustness on ACS Income, training
on CA-2018") from the multi-seed raw results.

TABLE XI is a *subset + re-aggregation* of results/RESULTS_multiseed.md, NOT a
copy:
  * Geographic shift is restricted to {MS, WY, WV} (the experiment also computes
    SD, PR, kept in RESULTS_multiseed.md); GEO-AVG is recomputed over those three
    per seed, then averaged across seeds.
  * Temporal shift = {2019, 2021, 2022}; TEMP-AVG over those three.
  * Δ = AVG − ID (pp).
  * Systems: ReLU Teacher / ReLU (same-size student) / BNN (Bernstein student)
    accuracy rows, plus the Rules block from α_sc=0.1, α_conf=0.2 with
    Total acc = the CART fallback (the deterministic, paper-headline fallback).

Reads:  results/metrics_multiseed_raw.csv  (one row per seed × system × condition)
Writes: results/table_xi.tex (booktabs) and results/RESULTS_table_xi.md; also
        prints the Markdown table to stdout.

Because everything here is recomputed from the per-seed raw values, the std
columns are exact (sample std, ddof=1).
"""

import argparse
import csv
import os

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# ── TABLE XI selection ────────────────────────────────────────────────────────
ID_COND   = 'ID: CA-2018 (trained on)'
GEO_COLS  = [('MS', 'Geo: MS-2018'), ('WY', 'Geo: WY-2018'), ('WV', 'Geo: WV-2018')]
TEMP_COLS = [('2019', 'Temporal: CA-2019'), ('2021', 'Temporal: CA-2021'),
             ('2022', 'Temporal: CA-2022')]
RULES_ALPHA_SC   = '0.1'     # α_sc for the Rules block (α_conf fixed at 0.2)
RULES_SYSTEM     = 'Rules + CART fallback'

# (display label, source system, alpha_sc, metric column) for each table row.
ROWS = [
    ('ReLU Teacher', 'Acc. (%)',        'ReLU FC teacher',          '', 'total_acc'),
    ('ReLU',         'Acc. (%)',        'ReLU student (same size)', '', 'total_acc'),
    ('BNN',          'Acc. (%)',        'Bernstein student',        '', 'total_acc'),
    ('Rules',        'Coverage (%)',    RULES_SYSTEM, RULES_ALPHA_SC, 'coverage_pct'),
    ('Rules',        'Covered acc. (%)', RULES_SYSTEM, RULES_ALPHA_SC, 'covered_acc'),
    ('Rules',        'Total acc. (%)',  RULES_SYSTEM, RULES_ALPHA_SC, 'total_acc'),
]


def load_raw(path):
    """data[(system, alpha_sc, condition)][seed] = {metric: float}."""
    data = {}
    with open(path, newline='') as f:
        for r in csv.DictReader(f):
            key = (r['system'], r['alpha_sc'], r['condition'])
            rec = data.setdefault(key, {})
            vals = {}
            for m in ('coverage_pct', 'covered_acc', 'total_acc'):
                v = r.get(m, '')
                if v != '':
                    vals[m] = float(v)
            rec[int(r['seed'])] = vals
    return data


def _seed_vals(data, system, alpha_sc, metric, cond):
    """Sorted-by-seed list of a metric's value under one condition."""
    rec = data.get((system, alpha_sc, cond), {})
    out = []
    for seed in sorted(rec):
        v = rec[seed].get(metric)
        if v is not None:
            out.append(v)
    return np.asarray(out, dtype=float)


def _mean_std(a):
    a = np.asarray(a, dtype=float)
    if a.size == 0:
        return float('nan'), 0.0
    return float(a.mean()), (float(a.std(ddof=1)) if a.size > 1 else 0.0)


def compute_row(data, system, alpha_sc, metric):
    """Return the full set of TABLE XI cells for one (system, metric) row."""
    idv = _seed_vals(data, system, alpha_sc, metric, ID_COND)
    id_m, id_s = _mean_std(idv)

    def block(cols):
        # per-seed average over the selected columns, then aggregate across seeds
        per_state = [_seed_vals(data, system, alpha_sc, metric, c) for _, c in cols]
        n_seed = min((len(s) for s in per_state), default=0)
        per_state = [s[:n_seed] for s in per_state]
        avg_series = np.mean(np.vstack(per_state), axis=0) if n_seed else np.array([])
        avg_m, avg_s = _mean_std(avg_series)
        state_means = [float(s.mean()) if len(s) else float('nan') for s in per_state]
        return avg_m, avg_s, state_means

    geo_m, geo_s, geo_states = block(GEO_COLS)
    temp_m, temp_s, temp_states = block(TEMP_COLS)
    return {
        'id': (id_m, id_s),
        'geo_avg': (geo_m, geo_s), 'geo_delta': geo_m - id_m, 'geo_states': geo_states,
        'temp_avg': (temp_m, temp_s), 'temp_delta': temp_m - id_m, 'temp_states': temp_states,
    }


def _ms(mean, std):
    return f"{mean:.2f}±{std:.2f}"


def render_markdown(data, n_seeds):
    geo_names = [n for n, _ in GEO_COLS]
    temp_names = [n for n, _ in TEMP_COLS]
    header = (['System', 'Metric', 'ID (CA-18)', 'GEO-AVG', 'Δ'] + geo_names +
              ['TEMP-AVG', 'Δ'] + temp_names)
    L = []
    L.append("# TABLE XI — Distribution-shift robustness on ACS Income (CA-2018)\n")
    L.append(f"ID = held-out CA-2018 test set; GEO-AVG and TEMP-AVG are means over the "
             f"shifted conditions ({', '.join(geo_names)} / {', '.join(temp_names)}); "
             f"Δ = AVG − ID (pp). Means±std over {n_seeds} seeds. Rules block: "
             f"α_sc={RULES_ALPHA_SC}, α_conf=0.2, Total acc = CART fallback.\n")
    L.append("| " + " | ".join(header) + " |")
    L.append("|" + "---|" * len(header))
    for disp, metric, system, alpha_sc, col in ROWS:
        c = compute_row(data, system, alpha_sc, col)
        cells = [disp, metric, _ms(*c['id']), _ms(*c['geo_avg']), f"{c['geo_delta']:+.1f}"]
        cells += [f"{v:.2f}" for v in c['geo_states']]
        cells += [_ms(*c['temp_avg']), f"{c['temp_delta']:+.1f}"]
        cells += [f"{v:.2f}" for v in c['temp_states']]
        L.append("| " + " | ".join(cells) + " |")
    return "\n".join(L) + "\n"


def render_latex(data, n_seeds):
    geo_names = [n for n, _ in GEO_COLS]
    temp_names = [n for n, _ in TEMP_COLS]
    ncol = 3 + 2 + len(geo_names) + 2 + len(temp_names)   # System,Metric,ID | geo | temp
    L = []
    L.append("% TABLE XI: distribution-shift robustness on ACS Income (CA-2018).")
    L.append("% Generated by make_table_xi.py from results/metrics_multiseed_raw.csv.")
    L.append("\\begin{tabular}{ll" + "c" * (ncol - 2) + "}")
    L.append("\\toprule")
    L.append(" & & & \\multicolumn{" + str(2 + len(geo_names)) +
             "}{c}{Geographic shift} & \\multicolumn{" + str(2 + len(temp_names)) +
             "}{c}{Temporal shift (CA)} \\\\")
    L.append("System & Metric & ID (CA-18) & GEO-AVG & $\\Delta$ & " +
             " & ".join(geo_names) + " & TEMP-AVG & $\\Delta$ & " +
             " & ".join(temp_names) + " \\\\")
    L.append("\\midrule")
    prev = None
    for disp, metric, system, alpha_sc, col in ROWS:
        if prev is not None and prev != disp:
            L.append("\\midrule")
        c = compute_row(data, system, alpha_sc, col)
        def pm(ms):
            return f"{ms[0]:.2f}$\\pm${ms[1]:.2f}"
        metric_tex = metric.replace('%', '\\%')        # % is a LaTeX comment char
        cells = [disp, metric_tex, pm(c['id']), pm(c['geo_avg']), f"{c['geo_delta']:+.1f}"]
        cells += [f"{v:.2f}" for v in c['geo_states']]
        cells += [pm(c['temp_avg']), f"{c['temp_delta']:+.1f}"]
        cells += [f"{v:.2f}" for v in c['temp_states']]
        L.append(" & ".join(cells) + " \\\\")
        prev = disp
    L.append("\\bottomrule")
    L.append("\\end{tabular}")
    return "\n".join(L) + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--raw', default=os.path.join(SCRIPT_DIR, 'results', 'metrics_multiseed_raw.csv'))
    ap.add_argument('--latex', default=os.path.join(SCRIPT_DIR, 'results', 'table_xi.tex'))
    ap.add_argument('--md',    default=os.path.join(SCRIPT_DIR, 'results', 'RESULTS_table_xi.md'))
    args = ap.parse_args()

    data = load_raw(args.raw)
    seeds = sorted({s for rec in data.values() for s in rec})
    n_seeds = len(seeds)

    md = render_markdown(data, n_seeds)
    print(md)
    with open(args.md, 'w') as f:
        f.write(md)
    with open(args.latex, 'w') as f:
        f.write(render_latex(data, n_seeds))
    print(f"Wrote {args.md}\nWrote {args.latex}")


if __name__ == '__main__':
    main()
