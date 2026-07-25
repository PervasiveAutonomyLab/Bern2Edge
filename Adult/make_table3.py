"""
make_table3.py
--------------
Render the rule-extraction table (paper TABLE III) from `rule_results.csv`.

For each architecture and each same-coverage penalty it reports:
    Rules   = n_rules
    l       = avg_conditions          (avg conditions per rule)
    Cov     = test_covered_pct
    Cov.Acc = test_covered_rule_acc
    Acc_t   = test_rule_acc

Emits both a Markdown table (stdout) and a LaTeX table (--latex path). The BNN
accuracy column in the paper is the hardware/int8-deployed accuracy and is
sourced separately, so it is not reproduced here.

Run:
    cd Bern2Edge
    python Adult/make_table3.py
    python Adult/make_table3.py --latex Adult/table3.tex
"""

import argparse
import csv
import os

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS_CSV = os.path.join(HERE, "rule_results.csv")

# Table row order (display label, arch string as stored in the CSV).
ARCH_ORDER = [
    ("[14, 16, 2]",     "{14, 16, 2}"),
    ("[14, 32, 2]",     "{14, 32, 2}"),
    ("[14, 128, 2]",    "{14, 128, 2}"),
    ("[14, 16, 8, 2]",  "{14, 16, 8, 2}"),
    ("[14, 32, 16, 2]", "{14, 32, 16, 2}"),
]
SAME_COV_ORDER = ["0.5", "0.1"]


def load_rows(csv_path, conflict_alpha="0.1"):
    """Index rows by (arch, same_cov_alpha) for the given conflict penalty.

    NOTE: the arch field is quoted and preceded by a space (`"[14, 16, 2]"`), so
    the CSV MUST be read with skipinitialspace=True or its inner commas split it
    into extra columns.
    """
    table = {}
    with open(csv_path, newline="") as f:
        for raw in csv.DictReader(f, skipinitialspace=True):
            row = {(k.strip() if k else k): (v.strip() if v else v)
                   for k, v in raw.items()}
            if row.get("conflict_alpha") != conflict_alpha:
                continue
            table[(row["arch"], row["same_cov_alpha"])] = row
    return table


def _cell(row):
    return (row["n_rules"], row["avg_conditions"], row["test_covered_pct"],
            row["test_covered_rule_acc"], row["test_rule_acc"])


def markdown(table):
    lines = ["| Arch | a_sc | Rules | l | Cov (%) | Cov.Acc (%) | Acc_t (%) |",
             "|------|------|------:|----:|--------:|------------:|----------:|"]
    for arch_key, label in ARCH_ORDER:
        for i, sca in enumerate(SAME_COV_ORDER):
            row = table.get((arch_key, sca))
            arch_col = label if i == 0 else ""
            if row is None:
                lines.append(f"| {arch_col} | {sca} | — | — | — | — | — |")
                continue
            n, l, cov, cacc, acc = _cell(row)
            lines.append(f"| {arch_col} | {sca} | {n} | {l} | {cov} | {cacc} | {acc} |")
    return "\n".join(lines)


def latex(table):
    out = [r"\begin{tabular}{llrrrrr}", r"\toprule",
           r"Arch & $\alpha_{\mathrm{sc}}$ & Rules & $\bar{\ell}$ & "
           r"Cov.\ (\%) & Cov.\ Acc.\ (\%) & Acc$_t$ (\%) \\", r"\midrule"]
    for arch_key, label in ARCH_ORDER:
        for i, sca in enumerate(SAME_COV_ORDER):
            row = table.get((arch_key, sca))
            arch_col = (r"\multirow{2}{*}{$" + label + r"$}") if i == 0 else ""
            if row is None:
                out.append(f"{arch_col} & {sca} & -- & -- & -- & -- & -- \\\\")
            else:
                n, l, cov, cacc, acc = _cell(row)
                out.append(f"{arch_col} & {sca} & {n} & {l} & {cov} & {cacc} & {acc} \\\\")
        if arch_key != ARCH_ORDER[-1][0]:
            out.append(r"\midrule")
    out += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", default=RESULTS_CSV, help="results CSV (default: rule_results.csv)")
    ap.add_argument("--conflict", default="0.1", help="conflict penalty to select (default: 0.1)")
    ap.add_argument("--latex", default=None, help="also write a LaTeX table to this path")
    args = ap.parse_args()

    if not os.path.exists(args.csv):
        raise SystemExit(f"results CSV not found: {args.csv}\n"
                         "Run Adult/run_rule_extraction.py first.")
    table = load_rows(args.csv, args.conflict)
    print(markdown(table))
    if args.latex:
        with open(args.latex, "w") as f:
            f.write(latex(table) + "\n")
        print(f"\nLaTeX table written to {args.latex}")


if __name__ == "__main__":
    main()
