Original rule JSONs from the paper's penalty sweep (TABLE VIII), copied verbatim
from adult_v2_rules/model_purity_distillation/trial_grid5_depth3_p90_dense/jsons_fbtree/.

Per combo (same_cov_alpha=sca, conflict_alpha=ca):
  {stem}_sca{sca}_ca{ca}.json         - the rule set (single-file schema)
  {stem}_sca{sca}_ca{ca}_tree.npz     - CART fallback structure
  {stem}_sca{sca}_ca{ca}_fbtrain.npy  - fallback predictions on train
  {stem}_sca{sca}_ca{ca}_fbtest.npy   - fallback predictions on test

These are the reference artifacts. The tool-regenerated equivalents live in
../penalty_sweep_14x32x2_jsons/ (rules_float.json + rules_int8.json per combo).
