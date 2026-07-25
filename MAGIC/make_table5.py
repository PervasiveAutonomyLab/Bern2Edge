"""
make_table5.py  —  paper TABLE V (rule extraction on MAGIC).

Aggregates the 5-seed rule-extraction sweep in ``5_fold_results.csv`` into the four
Bern2Edge rows and prints the full table (console + paste-ready LaTeX). The three
decompositional baselines (DeepRED, REM-D, ECLAIRE) are copied verbatim from
Zarlenga et al. [20] and hard-coded.

Row grouping (verified against the shipped CSV):
  - KD    rows: ckpt contains 'alpha0.5'  (arch 10x64x32x2,   distilled with alpha=0.5)
  - no-KD rows: ckpt contains 'alpha0_T1' (arch 10x64x32x16x2, pure CE, alpha=0)
  - split further by same_cov_alpha (a_sc) in {0.1, 0.5}
Aggregation over the 5 seeds:
  Rules = mean +/- population-std of n_rules ; l-bar = mean avg_conditions ;
  Acc   = mean test_rule_acc_mp_sparse       ; Fid   = mean test_cov_fidelity

Usage:
    python make_table5.py
"""
import os
import csv

HERE = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(HERE, "5_fold_results.csv")

# Decompositional baselines from Zarlenga et al. [20] (mean +/- std, 5 folds) — copied.
BASELINES = [
    # (type, method, rules_str, l_bar, acc, fid)
    ("Decompositional", "DeepRED [19]", "5143 +/- 9799", 5.43, 78.7, 89.3),
    ("Decompositional", "REM-D [35]",   "3617 +/- 6748", 5.41, 78.6, 89.4),
    ("Decompositional", "ECLAIRE [20]", "396 +/- 75",    3.82, 84.6, 89.4),
]


def _mean(xs):
    return sum(xs) / len(xs)


def _pop_std(xs):
    m = _mean(xs)
    return (sum((x - m) ** 2 for x in xs) / len(xs)) ** 0.5


def load_rows():
    rows = []
    with open(CSV_PATH, newline="") as f:
        # The CSV is whitespace-aligned; skipinitialspace lets the quoted arch field
        # (e.g. "[10, 64, 32, 2]") be read as one field despite its internal commas.
        reader = csv.reader(f, skipinitialspace=True)
        header = [h.strip() for h in next(reader)]
        for raw in reader:
            if not raw or not raw[0].strip():
                continue
            rows.append({h: v.strip() for h, v in zip(header, raw)})
    return rows


def aggregate(rows):
    """Return dict keyed by (kd_flag, same_cov_alpha) -> aggregated metrics."""
    groups = {}
    for r in rows:
        ckpt = r["ckpt"]
        if "alpha0.5" in ckpt:
            kd = "KD"
        elif "alpha0_T1" in ckpt:
            kd = "no-KD"
        else:
            continue
        sca = float(r["same_cov_alpha"])
        groups.setdefault((kd, sca), []).append(r)

    out = {}
    for key, grp in groups.items():
        n_rules = [int(r["n_rules"]) for r in grp]
        out[key] = {
            "n_seeds":    len(grp),
            "rules_mean": _mean(n_rules),
            "rules_std":  _pop_std(n_rules),
            "l_bar":      _mean([float(r["avg_conditions"]) for r in grp]),
            "acc":        _mean([float(r["test_rule_acc_mp_sparse"]) for r in grp]),
            "fid":        _mean([float(r["test_cov_fidelity"]) for r in grp]),
        }
    return out


# Order of the four Bern2Edge rows (matches the paper).
_ORDER = [("no-KD", 0.1), ("no-KD", 0.5), ("KD", 0.1), ("KD", 0.5)]


def _label(kd, sca):
    return f"{kd} (a_sc={sca:g})"


def print_console(agg):
    print("=" * 74)
    print("TABLE V — Rule extraction on MAGIC Gamma Telescope")
    print("=" * 74)
    print(f"{'Type':<16}{'Method':<18}{'Rules':>16}{'l':>7}{'Acc(%)':>9}{'Fid(%)':>9}")
    print("-" * 74)
    for t, m, rules, l, acc, fid in BASELINES:
        print(f"{t:<16}{m:<18}{rules:>16}{l:>7.2f}{acc:>9.1f}{fid:>9.1f}")
    print("-" * 74)
    for kd, sca in _ORDER:
        a = agg[(kd, sca)]
        rules = f"{round(a['rules_mean'])} +/- {round(a['rules_std'])}"
        print(f"{'Bern2Edge':<16}{_label(kd, sca):<18}{rules:>16}"
              f"{a['l_bar']:>7.2f}{a['acc']:>9.1f}{a['fid']:>9.1f}")
    print("=" * 74)


def print_latex(agg):
    print("\n--- LaTeX (paste-ready) ---")
    print(r"\begin{tabular}{llcccc}")
    print(r"\toprule")
    print(r"Type & Method & Rules & $\bar{\ell}$ & Acc (\%) & Fid (\%) \\")
    print(r"\midrule")
    for t, m, rules, l, acc, fid in BASELINES:
        rules_tex = rules.replace("+/-", r"$\pm$")
        print(f"{t} & {m} & {rules_tex} & {l:.2f} & {acc:.1f} & {fid:.1f} \\\\")
    print(r"\midrule")
    for i, (kd, sca) in enumerate(_ORDER):
        a = agg[(kd, sca)]
        typ = r"\textbf{Bern2Edge}" if i == 0 else ""
        method = f"{kd} ($\\alpha_{{\\mathrm{{sc}}}}$={sca:g})"
        rules = f"{round(a['rules_mean'])} $\\pm$ {round(a['rules_std'])}"
        print(f"{typ} & {method} & {rules} & {a['l_bar']:.2f} "
              f"& {a['acc']:.1f} & {a['fid']:.1f} \\\\")
    print(r"\bottomrule")
    print(r"\end{tabular}")


def main():
    agg = aggregate(load_rows())
    print_console(agg)
    print_latex(agg)


if __name__ == "__main__":
    main()
