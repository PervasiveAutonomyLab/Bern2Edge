# Table VII — XC7S15 deployment

This experiment reproduces the Adult deployment comparison on the low-power
Spartan-7 XC7S15. It includes six degree-3 Bernstein BNNs with architecture
`{14,h,2}` and two CART-fallback symbolic classifiers, R50 and R29.

Artifacts are referenced from their canonical repository locations rather than
duplicated here:

- BNN checkpoints: `Adult/student_model_weights/` (one selected checkpoint per
  width; the selected seed may differ by width);
- R50: the 50-rule JSON and CART fallback under
  `Adult/table9_fallback_ablation/artifacts/tree/`; its full weight vectors are
  emitted as the dense rule ROM used by the XC7S15 implementation;
- R29: the dense `14x16x8x2`, seed-6 rule/CART pair under
  `Adult/rule_jsons/..._sca0.5_ca0.1/`.

## Software and table rendering

```bash
python Adult/table7_xc7s15_deployment/reproduce_table7.py
```

This evaluates the six checkpoints and two rule artifacts, records their direct
paths in `table7_artifact_metrics.csv`, joins the committed post-route metrics
from `hardware_results.csv`, and writes `table7_values.csv` and `table7.tex`.

## HLS generation and synthesis

```bash
python Adult/table7_xc7s15_deployment/generate_and_synthesize_table_vii.py \
  --generate-only
```

The BNNs use `fix<18,8>`, a time-multiplexed narrow-product datapath, and
per-neuron activation LUTs. R50 moves its rule ROM to LUTRAM; R29 retains its
smaller rule ROM in BRAM. All designs target `xc7s15ftgb196-1` at 15 ns.

Load Vitis HLS and Vivado 2024.1 with
`source <Vitis>/2024.1/settings64.sh`, then remove `--generate-only` to run
csynth. The table's LUT, FF,
power, and final resource values are post-route Vivado measurements; csynth
alone does not reproduce them. Fresh post-route verification requires Vivado
and the paper tool setup.
