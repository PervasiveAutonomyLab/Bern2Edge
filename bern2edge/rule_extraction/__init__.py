"""
bern2edge.rule_extraction
-------------------------
Symbolic rule extraction from trained Bernstein student networks.

Modules:
  * bern_regimes  - Bernstein activation-geometry (motifs, regime breakpoints);
                    owns the shared N_FIXED_GRID constant.
  * extraction    - candidate generation, greedy cover, fallbacks, metrics, JSON.
  * quantize      - int8 / fix<16,8> quantization of the extracted rules.
  * visualize_neurons - one-figure neuron/regime illustration.

Typical use (see Adult/run_rule_extraction.py for the full driver):

    from bern2edge.rule_extraction import ExtractionConfig, extract_rules
    cfg = ExtractionConfig(fallback_mode="tree", same_cov_alpha=0.5)
    result = extract_rules(cfg, ckpt_name, arch, feature_names,
                           model=model, X_train=Xtr, y_gt_train=ytr,
                           X_test=Xte, y_gt_test=yte)
"""

from .bern_regimes import (N_FIXED_GRID, build_neuron_regimes,
                           classify_motif, classify_motif_extended,
                           compute_regime_breakpoints)
from .extraction import (ExtractionConfig, ExtractionResult,
                         extract_rules, prepare_context)
from .quantize import quantize_rule_json, quantize_cart_thresholds

__all__ = [
    "N_FIXED_GRID", "build_neuron_regimes", "classify_motif",
    "classify_motif_extended", "compute_regime_breakpoints",
    "ExtractionConfig", "ExtractionResult", "extract_rules", "prepare_context",
    "quantize_rule_json", "quantize_cart_thresholds",
]
