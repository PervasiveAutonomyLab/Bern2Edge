# Paper-results coverage

This file maps Section IV of `Bern2Edge.pdf` to the artifact. “Render” means
formatting committed results; “recompute” means evaluating shipped checkpoints
or rerunning the analysis. FPGA results are reproducible only when the artifact
contains the synthesis source, tool configuration, and raw reports—not merely a
number copied from the paper.

| Paper result | Artifact status | Command or evidence |
|---|---|---|
| Table I — compression and KV260 synthesis | **Partial:** accuracy and CE recomputed from 90 checkpoints; HLS values transcribed | `python table_i_compression/reproduce_table_i.py` |
| Table II — matched Covertype budgets | **Partial:** accuracy recomputed from 50 checkpoints; HLS values joined from a shipped raw CSV, but synthesis is not rerun | `python cover_type/reproduce_table_ii.py` |
| Table III — Adult rule extraction | **Included:** rules and metrics can be regenerated; committed CSV can be rendered | `python Adult/run_rule_extraction.py`; `python Adult/make_table3.py` |
| Table IV — LUT BNN vs rule-network hardware | **Partial:** all network/rule accuracies evaluated; hardware values transcribed | `python Adult/table4_rule_network_hardware/reproduce_table4.py`; HLS synthesis is not rerun |
| Table V — MAGIC rule extraction | **Partial:** committed five-fold metrics render exactly; retraining/extraction is approximate | `python MAGIC/make_table5.py` |
| Table VI — end-to-end hardware | **Partial:** all software accuracies evaluated; hardware values transcribed | `python end_to_end_results/reproduce_table_vi.py`; HLS synthesis is not rerun |
| Table VII — Spartan-7 deployment | **Missing** | No XC7S15 HLS projects/reports, quantized release inputs, power-measurement records, or renderer |
| Table VIII — penalty sweep | **Included:** committed and regenerable | `python Adult/run_penalty_sweep.py`; `python Adult/make_table8.py` |
| Table IX — fallback ablation | **Included:** four exact rule/fallback artifacts are evaluated on uncovered held-out samples; full HLS accuracy and fallback-only resources are joined from the committed synthesis summary | `python Adult/table9_fallback_ablation/reproduce_table9.py` |
| Table X — MAGIC certification | **Included:** committed metrics render exactly; live recomputation uses shipped weights and optional auto_LiRPA | `python MAGIC/make_table_x.py`; see `MAGIC/README.md` |
| Table XI — ACS distribution shift | **Included:** recomputed from shipped per-seed checkpoints | `python ACS/run_multiseed.py`; `python ACS/make_table_xi.py` |
| Table XII — transformer FFN | **Partial:** accuracy recomputed from shipped weights; HLS rows transcribed | `python Transformer/eval_release.py`; `python Transformer/make_table_xii.py` |
| Figure 7 — KD-weight sweep | **Missing** | No complete α-sweep result data or plotting script |
| Figure 8 — degree/hardware comparison | **Missing** | No per-degree HLS projects/reports, result data, or plotting script |
| Figure 9 — penalty curves | **Included:** 105 exact rule/CART pairs are re-evaluated; both five-architecture means and plots are regenerated and all 44 coordinates verified | `python Adult/figure9_penalty_sweep/reproduce_figure9.py` |
| Figure 10 — sparsity/BRAM curve | **Included:** 13 exact rule artifacts are re-evaluated; committed BRAM measurements are joined and all 26 coordinates verified | `python Adult/figure10_sparsity_sweep/reproduce_figure10.py` |

## Highest-priority additions

For a strong reproducibility claim, add the following in this order:

1. A single `results/` hierarchy with machine-readable inputs and renderers for
   Tables IV, VI, and VII.
2. Raw HLS reports plus tool/project settings for every hardware table. If full
   HLS projects are too large, archive them separately under the same versioned
   DOI and link them here.
3. CSV data and deterministic plot scripts for Figures 7 and 8.
4. A top-level driver such as `reproduce.py --quick` and
   `reproduce.py --table XI` that runs the documented commands and records
   environment/version information.

Until these are added, badge language should claim reproducibility only for the
specific results marked “Included” or “Partial,” not for every result in the
paper.
