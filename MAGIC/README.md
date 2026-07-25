# MAGIC Gamma Telescope — rule extraction (TABLE V) & certified robustness (TABLE X)

Bernstein-student knowledge distillation → interpretable slab rules, plus a **sound
local-robustness certificate** comparing the rule system, the Bernstein network (BNN),
and a matched ReLU network under per-feature input noise.

- **TABLE V** — rule extraction: 4 Bern2Edge rows (KD / no-KD × same-cov penalty
  `α_sc ∈ {0.1, 0.5}`) over 5 seeds, vs. the copied decompositional baselines.
- **TABLE X** — robustness certification (arch `10×64×32×2`, seed 42): empirical
  Rule Acc. + Fidelity, and certified-stable **Rules** (rule-system decision) / **BNN**
  (Bern-IBP) / **ReLU** (auto_LiRPA IBP).

Everything here is self-contained: it reuses only the shared `bern2edge.data` (MAGIC
loader) and `bern2edge.kdtrain`; the model core, rule extraction, and certification
are local.

## Layout

```
MAGIC/
  magic04.data                 raw UCI dataset (g=gamma->1, h=hadron->0)
  teacher_magic.pt             frozen teacher MLP + fixed split + fitted QuantileTransformer
  student_model_weights/       shipped students: bern {0..3,42} x {10x64x32x2, 10x64x32x16x2} + matched ReLU seed42
  rule_jsons/                  shipped extracted rules (headline g5 seed42 + the g3 TABLE-V set)
  5_fold_results.csv           shipped TABLE V metrics (authoritative)
  results/table_x.csv          shipped TABLE X metrics (authoritative)

  teacher_magic.py             (re)train + save the teacher MLP
  run_kd_experiments.py        train the students (bern + matched ReLU), seeds {0,1,2,3,42}
  extract_rules.py             rule extraction + adaptive-k sparsification  (uses rule_extraction_magic, default_rule_utils)
  bern_net.py                  local model core: BernsteinLayer/FCModel + Bern-IBP certification
  rule_certify.py              sound rule-system decision certificate (interval propagation)
  relu_lirpa_certify.py        TABLE X driver: BNN vs ReLU (auto_LiRPA) + empirical columns  (writes results/table_x.csv)
  make_table5.py               render TABLE V from 5_fold_results.csv
  make_table_x.py              render TABLE X from results/table_x.csv
```

## Reproduce the exact tables (seconds, no training, no auto_LiRPA)

Run from this folder:

```bash
python make_table5.py     # TABLE V  (reads 5_fold_results.csv)
python make_table_x.py    # TABLE X  (reads results/table_x.csv)
```

These read the shipped metrics CSVs and print the published numbers (console + LaTeX).

## Regenerate from the shipped model weights

TABLE X recomputed live from the shipped Bernstein + matched-ReLU students and the
headline rule file (this is the only step that needs **auto_LiRPA** — for the ReLU
column; see the root [INSTALL.md](../INSTALL.md)):

```bash
RF=rule_jsons/magic_rules_kd_fc_10x64x32x2_bern_deg3_alpha0.5_T2_lr0.003_wd0.0001_seed42_g5_p85_mc5_d2_adaptive.json
python relu_lirpa_certify.py --arch 10x64x32x2 --rule-file $RF --write-csv results/table_x.csv
```

It reproduces `results/table_x.csv` and asserts soundness (certified ⊆ empirically-stable,
0 violations) and monotonicity of every certified column.

## Train / extract from scratch (approximate)

A model trained from scratch has different weights, so its learned neuron shapes — and
therefore the *number* of extracted rules — differ; the TABLE V metrics reproduce within
±std, not digit-for-digit (use the shipped artifacts for the exact numbers).

```bash
python teacher_magic.py                       # re-train the teacher -> teacher_magic.pt
python run_kd_experiments.py                  # train all students -> student_model_weights/
#   quick smoke: python run_kd_experiments.py --seeds 42 --epochs 5

# extract rules for one student (TABLE V uses grid=3; the TABLE X headline uses grid=5):
python extract_rules.py \
  --ckpt student_model_weights/kd_fc_10x64x32x2_bern_deg3_alpha0.5_T2_lr0.003_wd0.0001_seed42.pth \
  --grid 3 --purity 0.85 --min_cov 5 --depth 2 --conflict_alpha 0.1 --same_cov_alpha 0.1
#   -> rule_jsons/*.json  and a metrics row in extraction_results.csv
```

`--same_cov_alpha 0.1` vs `0.5` selects the two TABLE V penalty settings. (Candidate
rules are cached under `candidates_cache/`; both are git-ignored regenerable outputs.)

## Notes

- **Certificate semantics.** "Certified" = provably unchanged label over the *entire*
  per-feature noise box `[x−c·σ, x+c·σ]` (worst case), not just under sampled noise.
  BNN uses Bern-IBP (`bern_net.subinterval_bounds`); ReLU uses auto_LiRPA standard IBP —
  the algorithmically matched verifier, isolating the activation, not the verifier.
- **Runtime (CPU):** table renderers — seconds; `relu_lirpa_certify.py` — ~1–2 min;
  one student train — a few minutes; one rule extraction — ~1–2 min (first run builds the
  candidate cache).
