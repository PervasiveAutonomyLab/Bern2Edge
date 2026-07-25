"""
make_table_x.py  —  paper TABLE X (robustness certification on MAGIC).

Reads the shipped certification results (``results/table_x.csv``, produced by
``relu_lirpa_certify.py``) and prints TABLE X exactly: empirical Rule Acc. + Fidelity,
and certified-stable Rules (= decision-flip) / BNN / ReLU. The NN-accuracy and
Boundary-Exit columns are intentionally not part of this table.

Usage:
    python make_table_x.py                     # reads results/table_x.csv
    python make_table_x.py --csv <path>
"""
import os
import csv
import argparse

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CSV = os.path.join(HERE, "results", "table_x.csv")


def load_rows(csv_path):
    with open(csv_path, newline="") as f:
        return list(csv.DictReader(f))


def print_console(rows):
    print("=" * 60)
    print("TABLE X — Robustness certification on MAGIC")
    print("=" * 60)
    print(f"{'':>6}{'Empirical (%)':^18}{'Certified (%)':^30}")
    print(f"{'c':>6}{'Rule Acc.':>10}{'Fid.':>8}{'Rules':>10}{'BNN':>8}{'ReLU':>8}")
    print("-" * 60)
    for r in rows:
        print(f"{float(r['c']):>6.2f}{float(r['rule_acc']):>10.1f}{float(r['fidelity']):>8.1f}"
              f"{float(r['decision_flip']):>10.1f}{float(r['nn_stable_bern']):>8.1f}"
              f"{float(r['relu_ibp']):>8.1f}")
    print("=" * 60)


def print_latex(rows):
    print("\n--- LaTeX (paste-ready) ---")
    print(r"\begin{tabular}{cccccc}")
    print(r"\toprule")
    print(r"& \multicolumn{2}{c}{Empirical (\%)} & \multicolumn{3}{c}{Certified (\%)} \\")
    print(r"\cmidrule(lr){2-3}\cmidrule(lr){4-6}")
    print(r"$c$ & Rule Acc. & Fid. & Rules & BNN & ReLU \\")
    print(r"\midrule")
    for r in rows:
        print(f"{float(r['c']):.2f} & {float(r['rule_acc']):.1f} & {float(r['fidelity']):.1f} & "
              f"{float(r['decision_flip']):.1f} & {float(r['nn_stable_bern']):.1f} & "
              f"{float(r['relu_ibp']):.1f} \\\\")
    print(r"\bottomrule")
    print(r"\end{tabular}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=DEFAULT_CSV)
    args = ap.parse_args()
    if not os.path.exists(args.csv):
        raise SystemExit(
            f"{args.csv} not found. Generate it with:\n"
            "  python relu_lirpa_certify.py --arch 10x64x32x2 "
            "--rule-file rule_jsons/<headline>.json --write-csv results/table_x.csv")
    rows = load_rows(args.csv)
    print_console(rows)
    print_latex(rows)


if __name__ == "__main__":
    main()
