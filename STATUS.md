# Project status

Bern2Edge is an open-source research codebase accompanying the paper
**“Bern2Edge: A Neurosymbolic Compiler for Edge Deployment via Bernstein
Polynomial Networks.”**

## Release

The archived software release is:

- **Version:** v1.0.0
- **GitHub release:** https://github.com/PervasiveAutonomyLab/Bern2Edge/releases/tag/v1.0.0
- **Zenodo DOI:** https://doi.org/10.5281/zenodo.21726441
- **Release date:** July 31, 2026
- **License:** MIT

The v1.0.0 release was prepared as the CODES 2026 artifact release for
Bern2Edge. Its release bundle includes the accepted paper, source code, trained
checkpoints and rule artifacts, reproduction scripts, HLS source-generation
workflows, and the documentation used for artifact evaluation.

The repository `main` branch may continue to evolve after v1.0.0. Use the tagged
GitHub release or the Zenodo archive when an immutable copy of the original
release is required.

## Reproduction coverage

The repository supports the following published results:

| Result | Reproduction support |
|---|---|
| Table I | Accuracy and loss recomputed from shipped checkpoints; HLS projects regenerated |
| Table II | Accuracy recomputed from shipped checkpoints; HLS projects regenerated |
| Tables III and VIII | Rules and metrics can be regenerated |
| Table IV | Five LUT checkpoints and matching rule/CART artifacts evaluated; HLS projects regenerated |
| Table V | Committed five-fold results render exactly; retraining/extraction is approximate |
| Table VI | Teacher, student, and rule software accuracies recomputed |
| Table VII | Six BNN and two rule artifacts evaluated; XC7S15 HLS projects regenerated |
| Table IX | Four fallback variants re-evaluated; full and fallback-only HLS projects regenerated |
| Table X | Committed metrics render exactly; live certification is available from shipped weights |
| Table XI | Results recomputed from shipped per-seed checkpoints |
| Table XII | SST-2 accuracy recomputed from shipped weights; encoder-layer HLS projects regenerated |
| Figure 9 | Re-evaluated from 105 shipped rule/CART pairs; plotted coordinates verified |
| Figure 10 | Re-evaluated from 13 shipped rule artifacts; plotted coordinates verified |

Fresh FPGA synthesis requires the external Vitis/Vivado environment documented
in `REQUIREMENTS.md` and `INSTALL.md`.

For result-specific commands and limitations, see `RESULTS.md`.

## Historical artifact-evaluation material

`ARTIFACT_EVALUATION.md` is retained as the evaluation guide for the archived
v1.0.0 CODES 2026 artifact. It is useful for reproducing the original review
workflow, but regular users should start with `README.md`, `INSTALL.md`, and
`RESULTS.md`.
