# Published results and reproduction coverage

This file maps the Bern2Edge paper results to the code and released artifacts in
this repository.

Terminology used below:

- **Render** — format committed result files without rerunning the experiment.
- **Recompute** — evaluate shipped checkpoints or rule artifacts again.
- **Regenerate** — rebuild rules, plots, or HLS source projects.
- **Fresh synthesis** — rerun FPGA synthesis to produce new hardware reports;
  this requires the external Vitis/Vivado environment documented in
  `REQUIREMENTS.md`.

| Paper result | Reproduction support | Main command or evidence |
|---|---|---|
| Table I — compression and KV260 synthesis | Accuracy and CE are recomputed from shipped checkpoints. All 18 HLS source projects can be regenerated. Fresh synthesis requires Vitis. | `python bnn_compression_synth/reproduce_table_i.py`; `python bnn_compression_synth/generate_and_synthesize_table_i.py --generate-only` |
| Table II — matched Covertype budgets | Accuracy is recomputed from all 50 shipped checkpoints. All ten HLS source projects can be regenerated. Fresh synthesis requires Vitis. | `python cover_type/reproduce_table_ii.py`; `python cover_type/reproduce_table_ii_hardware.py --generate-only` |
| Table III — Adult rule extraction | Rule sets and metrics can be regenerated; committed results can also be rendered directly. | `python Adult/run_rule_extraction.py`; `python Adult/make_table3.py` |
| Table IV — LUT BNN vs. rule-network hardware | Network and rule accuracies are evaluated from the shipped artifacts. Five rule-network HLS projects can be regenerated. Fresh synthesis requires Vitis. | `python Adult/table4_rule_network_hardware/reproduce_table4.py`; `python Adult/table4_rule_network_hardware/generate_and_synthesize_table_iv.py --generate-only` |
| Table V — MAGIC rule extraction | The committed five-fold metrics render exactly. Retraining and extraction from scratch are approximate because newly trained weights can differ. | `python MAGIC/make_table5.py` |
| Table VI — end-to-end hardware | Teacher, student, and rule software accuracies are recomputed. Hardware values are joined from the committed post-synthesis measurements. | `python end_to_end_results/reproduce_table_vi.py` |
| Table VII — XC7S15 deployment | Six BNN checkpoints and two rule artifacts are evaluated. All eight XC7S15 HLS projects can be regenerated; fresh synthesis/post-route implementation is optional and requires Vitis/Vivado. | `python Adult/table7_xc7s15_deployment/reproduce_table7.py`; `python Adult/table7_xc7s15_deployment/generate_and_synthesize_table_vii.py --generate-only` |
| Table VIII — penalty sweep | The configured penalty sweep can be regenerated and the committed results rendered. | `python Adult/run_penalty_sweep.py`; `python Adult/make_table8.py` |
| Table IX — fallback ablation | The four shipped rule/fallback variants are re-evaluated. Four full and four fallback-only HLS projects can be regenerated. Fresh synthesis requires Vitis. The Small BNN row uses the int8 `small_nn` implementation. | `python Adult/table9_fallback_ablation/reproduce_table9.py`; `python Adult/table9_fallback_ablation/generate_and_synthesize_table_ix.py --generate-only` |
| Table X — MAGIC certification | Committed metrics render exactly. Live certification can be recomputed from the shipped weights; the ReLU certification column requires optional `auto_LiRPA`. | `python MAGIC/make_table_x.py`; see `MAGIC/README.md` |
| Table XI — ACS distribution shift | Network and rule results are recomputed from the shipped per-seed checkpoints and deterministic rule/fallback workflow. | `python ACS/run_multiseed.py`; `python ACS/make_table_xi.py` |
| Table XII — Transformer FFN | SST-2 accuracy is recomputed from the shipped weights. Five encoder-layer HLS projects can be regenerated. Fresh synthesis requires Vitis. Training from scratch reproduces the result approximately rather than digit-for-digit. | `python Transformer/eval_release.py`; `python Transformer/make_table_xii.py`; `python Transformer/generate_and_synthesize_table_xii.py --generate-only` |
| Figure 9 — penalty curves | All 105 shipped rule/CART pairs are re-evaluated; the five-architecture means and plots are regenerated and all 44 plotted coordinates are verified. | `python Adult/figure9_penalty_sweep/reproduce_figure9.py` |
| Figure 10 — sparsity/BRAM curve | All 13 shipped rule artifacts are re-evaluated; committed BRAM measurements are joined and all 26 plotted coordinates are verified. | `python Adult/figure10_sparsity_sweep/reproduce_figure10.py` |

For installation and optional dependencies, see `INSTALL.md`. For exact
experiment provenance, output files, tolerances, and from-scratch procedures,
see the README inside the corresponding experiment directory.
