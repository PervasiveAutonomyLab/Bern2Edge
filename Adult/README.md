# Adult Census experiments

This directory contains the Adult components used by Tables I, III, VIII, and
part of the fallback analysis in Table IX. Run commands from the repository
root after following `INSTALL.md`.

## Data and provenance

The loader obtains the Adult dataset from OpenML on first use. The committed
`adult_teacher_ordinal.pt` stores the teacher checkpoint and fixed split/
preprocessing metadata. Network and rule checkpoints are research artifacts;
they contain model parameters and preprocessing state, not personal identifiers.
The underlying dataset remains subject to its source terms.

## Quick check: render committed rule results

```bash
python Adult/make_table3.py
python Adult/make_table8.py
```

These commands read `rule_results.csv` and
`penalty_sweep_14x32x2_results.csv`, print Markdown, and write `table3.tex` and
`table8.tex`. They do not train models or extract new rules.

## Regenerate Table III

```bash
python Adult/run_rule_extraction.py
python Adult/make_table3.py
```

The extraction driver loads the shipped Bernstein checkpoints, regenerates
float and quantized rule JSON files, evaluates coverage and accuracy, and writes
`rule_results.csv`. Use `--arch` to select one architecture and `--fallback
tree` for the paper's CART fallback.

## Regenerate Table VIII

```bash
python Adult/run_penalty_sweep.py
python Adult/make_table8.py
```

The driver evaluates the configured `(conflict_alpha, same_cov_alpha)` grid and
writes `penalty_sweep_14x32x2_results.csv` plus rule/fallback sidecars. The
`penalty_sweep_14x32x2_jsons_original/` directory preserves the original paper
outputs; newly generated outputs go to
`penalty_sweep_14x32x2_jsons/`.

## Outputs

- `kd_compression_results.csv`: Adult student-compression metrics used by the
  Table I aggregation.
- `rule_results.csv`: Table III rule count, complexity, coverage, covered
  accuracy, and total accuracy.
- `penalty_sweep_14x32x2_results.csv`: Table VIII joint penalty results.
- `rule_jsons/` and `penalty_sweep_14x32x2_jsons/`: generated float/int8 rules
  and fallback sidecars.

Regeneration may overwrite derived CSV/JSON/LaTeX outputs. Copy them elsewhere
first if you need to preserve a local run.
