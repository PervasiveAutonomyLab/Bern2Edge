# Paper-results coverage

This file maps Section IV of `Bern2Edge.pdf` to the artifact. “Render” means
formatting committed results; “recompute” means evaluating shipped checkpoints
or rerunning the analysis.

| Paper result | Artifact status | Command or evidence |
|---|---|---|
| Table I — compression and KV260 synthesis | **Partial:** accuracy/CE recomputed; all 18 HLS source projects regenerated; synthesis requires external Vitis | `python bnn_compression_synth/reproduce_table_i.py`; `python bnn_compression_synth/generate_and_synthesize_table_i.py --generate-only` |
| Table II — matched Covertype budgets | **Partial:** accuracy recomputed; all ten HLS source projects regenerated; synthesis requires external Vitis | `python cover_type/reproduce_table_ii.py`; `python cover_type/reproduce_table_ii_hardware.py --generate-only` |
| Table III — Adult rule extraction | **Included:** rules and metrics can be regenerated; committed CSV can be rendered | `python Adult/run_rule_extraction.py`; `python Adult/make_table3.py` |
| Table IV — LUT BNN vs rule-network hardware | **Partial:** all network/rule accuracies evaluated and five rule HLS projects regenerated; synthesis requires external Vitis | `python Adult/table4_rule_network_hardware/reproduce_table4.py`; `python Adult/table4_rule_network_hardware/generate_and_synthesize_table_iv.py --generate-only` |
| Table V — MAGIC rule extraction | **Partial:** committed five-fold metrics render exactly; retraining/extraction is approximate | `python MAGIC/make_table5.py` |
| Table VI — end-to-end hardware | **Partial:** all software accuracies evaluated | `python end_to_end_results/reproduce_table_vi.py` |
| Table VII — XC7S15 deployment | **Partial:** six BNN and two rule artifacts evaluated; all eight XC7S15 HLS projects regenerated; fresh Vitis/Vivado post-route results optional | `python Adult/table7_xc7s15_deployment/reproduce_table7.py`; `python Adult/table7_xc7s15_deployment/generate_and_synthesize_table_vii.py --generate-only` |
| Table VIII — penalty sweep | **Included:** committed and regenerable | `python Adult/run_penalty_sweep.py`; `python Adult/make_table8.py` |
| Table IX — fallback ablation | **Partial:** four exact rule/fallback artifacts are evaluated; four full and four fallback-only HLS projects are regenerated; fresh synthesis requires external Vitis. Small BNN is int8 `small_nn` | `python Adult/table9_fallback_ablation/reproduce_table9.py`; `python Adult/table9_fallback_ablation/generate_and_synthesize_table_ix.py --generate-only` |
| Table X — MAGIC certification | **Included:** committed metrics render exactly; live recomputation uses shipped weights and optional auto_LiRPA | `python MAGIC/make_table_x.py`; see `MAGIC/README.md` |
| Table XI — ACS distribution shift | **Included:** recomputed from shipped per-seed checkpoints | `python ACS/run_multiseed.py`; `python ACS/make_table_xi.py` |
| Table XII — transformer FFN | **Partial:** accuracy recomputed; five encoder-layer HLS projects regenerated from the shipped weights; synthesis requires external Vitis | `python Transformer/eval_release.py`; `python Transformer/make_table_xii.py`; `python Transformer/generate_and_synthesize_table_xii.py --generate-only` |
| Figure 9 — penalty curves | **Included:** 105 exact rule/CART pairs are re-evaluated; both five-architecture means and plots are regenerated and all 44 coordinates verified | `python Adult/figure9_penalty_sweep/reproduce_figure9.py` |
| Figure 10 — sparsity/BRAM curve | **Included:** 13 exact rule artifacts are re-evaluated; committed BRAM measurements are joined and all 26 coordinates verified | `python Adult/figure10_sparsity_sweep/reproduce_figure10.py` |
