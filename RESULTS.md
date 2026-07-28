# Paper-results coverage

This file maps Section IV of `Bern2Edge.pdf` to the artifact. “Render” means
formatting committed results; “recompute” means evaluating shipped checkpoints
or rerunning the analysis.

| Paper result | Artifact status | Command or evidence |
|---|---|---|
| Table I — compression and KV260 synthesis | **Partial:** accuracy and CE recomputed from 90 checkpoints | `python table_i_compression/reproduce_table_i.py` |
| Table II — matched Covertype budgets | **Partial:** accuracy recomputed from 50 checkpoints | `python cover_type/reproduce_table_ii.py` |
| Table III — Adult rule extraction | **Included:** rules and metrics can be regenerated; committed CSV can be rendered | `python Adult/run_rule_extraction.py`; `python Adult/make_table3.py` |
| Table IV — LUT BNN vs rule-network hardware | **Partial:** all network/rule accuracies evaluated | `python Adult/table4_rule_network_hardware/reproduce_table4.py` |
| Table V — MAGIC rule extraction | **Partial:** committed five-fold metrics render exactly; retraining/extraction is approximate | `python MAGIC/make_table5.py` |
| Table VI — end-to-end hardware | **Partial:** all software accuracies evaluated | `python end_to_end_results/reproduce_table_vi.py` |
| Table VIII — penalty sweep | **Included:** committed and regenerable | `python Adult/run_penalty_sweep.py`; `python Adult/make_table8.py` |
| Table IX — fallback ablation | **Included:** four exact rule/fallback artifacts are evaluated on uncovered held-out samples; full HLS accuracy and fallback-only resources are joined from the committed synthesis summary | `python Adult/table9_fallback_ablation/reproduce_table9.py` |
| Table X — MAGIC certification | **Included:** committed metrics render exactly; live recomputation uses shipped weights and optional auto_LiRPA | `python MAGIC/make_table_x.py`; see `MAGIC/README.md` |
| Table XI — ACS distribution shift | **Included:** recomputed from shipped per-seed checkpoints | `python ACS/run_multiseed.py`; `python ACS/make_table_xi.py` |
| Table XII — transformer FFN | **Partial:** accuracy recomputed from shipped weights | `python Transformer/eval_release.py`; `python Transformer/make_table_xii.py` |
| Figure 9 — penalty curves | **Included:** 105 exact rule/CART pairs are re-evaluated; both five-architecture means and plots are regenerated and all 44 coordinates verified | `python Adult/figure9_penalty_sweep/reproduce_figure9.py` |
| Figure 10 — sparsity/BRAM curve | **Included:** 13 exact rule artifacts are re-evaluated; committed BRAM measurements are joined and all 26 coordinates verified | `python Adult/figure10_sparsity_sweep/reproduce_figure10.py` |
