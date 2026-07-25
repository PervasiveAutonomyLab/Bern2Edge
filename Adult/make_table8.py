"""
make_table8.py
--------------
Render the joint penalty-sweep table (paper TABLE VIII) from the sweep CSV.

For each (alpha_conf, alpha_sc) combo it reports:
    Conf    = n_conflicts            (points covered by rules of both labels)
    Cov     = test_covered_pct
    Cov.Acc = test_covered_rule_acc
    Acc_t   = test_rule_acc
    Rules   = n_rules
    l       = avg_conditions

Rows are grouped by alpha_sc, with alpha_conf ascending inside each block (the
paper's layout). Emits Markdown (stdout) and optional LaTeX (--latex).

Run:
    cd Bern2Edge
    python Adult/run_penalty_sweep.py         # produces the CSV first
    python Adult/make_table8.py
    python Adult/make_table8.py --latex Adult/table8.tex
"""

import argparse
import csv
import os

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS_CSV = os.path.join(HERE, "penalty_sweep_14x32x2_results.csv")

# Which combos to show, and in what order (edit to match a different sweep).
SAME_COV_ORDER = ["0.1", "0.3", "0.5", "1.0"]     # block order (alpha_sc)
CONFLICT_ORDER = ["0.1", "1.0"]                    # within-block order (alpha_conf)


def _fmt(x):
    """Trim a stored float string to the paper's 2-decimal style (e.g. 82.1)."""
    try:
        return f"{float(x):g}"
    except (TypeError, ValueError):
        return x


def load_rows(csv_path):
    """Index sweep rows by (conflict_alpha, same_cov_alpha)."""
    table = {}
    with open(csv_path, newline="") as f:
        for raw in csv.DictReader(f, skipinitialspace=True):
            row = {(k.strip() if k else k): (v.strip() if v else v)
                   for k, v in raw.items()}
            table[(row["conflict_alpha"], row["same_cov_alpha"])] = row
    return table


def _cells(row):
    return (row["n_conflicts"], _fmt(row["test_covered_pct"]),
            _fmt(row["test_covered_rule_acc"]), _fmt(row["test_rule_acc"]),
            row["n_rules"], _fmt(row["avg_conditions"]))


def markdown(table):
    lines = ["| a_conf | a_sc | Conf | Cov (%) | Cov.Acc (%) | Acc_t (%) | Rules | l |",
             "|-------:|-----:|-----:|--------:|------------:|----------:|------:|----:|"]
    for sc in SAME_COV_ORDER:
        for conf in CONFLICT_ORDER:
            row = table.get((conf, sc))
            if row is None:
                lines.append(f"| {conf} | {sc} | — | — | — | — | — | — |")
                continue
            n, cov, cacc, acc, rules, l = _cells(row)
            lines.append(f"| {conf} | {sc} | {n} | {cov} | {cacc} | {acc} | {rules} | {l} |")
    return "\n".join(lines)


def latex(table):
    out = [r"\begin{tabular}{llrrrrrr}", r"\toprule",
           r"$\alpha_{\mathrm{conf}}$ & $\alpha_{\mathrm{sc}}$ & Conf. & Cov.\ (\%) & "
           r"Cov.\ Acc.\ (\%) & Acc$_t$ (\%) & Rules & $\bar{\ell}$ \\", r"\midrule"]
    for bi, sc in enumerate(SAME_COV_ORDER):
        for conf in CONFLICT_ORDER:
            row = table.get((conf, sc))
            if row is None:
                out.append(f"{conf} & {sc} & -- & -- & -- & -- & -- & -- \\\\")
            else:
                n, cov, cacc, acc, rules, l = _cells(row)
                out.append(f"{conf} & {sc} & {n} & {cov} & {cacc} & {acc} & {rules} & {l} \\\\")
        if bi != len(SAME_COV_ORDER) - 1:
            out.append(r"\midrule")
    out += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", default=RESULTS_CSV,
                    help="sweep results CSV (default: penalty_sweep_14x32x2_results.csv)")
    ap.add_argument("--latex", default=None, help="also write a LaTeX table to this path")
    args = ap.parse_args()

    if not os.path.exists(args.csv):
        raise SystemExit(f"results CSV not found: {args.csv}\n"
                         "Run Adult/run_penalty_sweep.py first.")
    table = load_rows(args.csv)
    print(markdown(table))
    if args.latex:
        with open(args.latex, "w") as f:
            f.write(latex(table) + "\n")
        print(f"\nLaTeX table written to {args.latex}")


if __name__ == "__main__":
    main()
