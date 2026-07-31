# Artifact status

The authors are applying for the **Available**, **Reviewed**, and
**Reproducible** CODES 2026 / ESWEEK artifact badges. The badges are
independent. Claims below are limited to what this archive contains.

## Available

The artifact includes source code, documentation, model checkpoints, result
files, the paper PDF, and an MIT license.

`Bern2Edge.pdf` is the final accepted 14-page paper and is tracked in this
revision. This revision must be archived in a persistent public repository, and
the resulting version-specific DOI must be included in the artifact submission.

## Reviewed

The artifact provides:

- installation and smoke-test instructions in `INSTALL.md`;
- hardware, software, storage, and network requirements in `REQUIREMENTS.md`;
- runnable commands and expected outputs in the root and experiment READMEs;
- committed inputs for quick, network-free table-rendering tests.

Model metrics are recomputed from shipped checkpoints where available.

## Reproducible

The artifact supports the following paper results:

| Result | Reproduction level |
|---|---|
| Table I | Accuracy and loss recomputed from shipped checkpoints |
| Table II | Accuracy recomputed from shipped checkpoints |
| Tables III and VIII | Rules and metrics can be regenerated |
| Table IV | Five LUT checkpoints and rule/CART artifacts evaluated |
| Table VI | Teacher/student/rule accuracies recomputed |
| Table VII | Six BNN and two rule artifacts evaluated; HLS projects regenerated |
| Table IX | Four fallback artifacts re-evaluated |
| Figure 9 | Re-evaluated from 105 shipped rule/CART pairs; all plotted coordinates verified |
| Figure 10 | Re-evaluated from 13 shipped rule artifacts; all coordinates verified |
| Table V | Exact rendering from shipped five-fold metrics; retraining is approximate |
| Table X | Exact rendering; live certification from shipped weights |
| Table XI | Recomputed from shipped per-seed checkpoints |
| Table XII | Accuracy recomputed from shipped weights |

Commands and limitations are summarized in `RESULTS.md`.

## Submission blockers and final checks

- Archive this exact revision and provide its version-specific DOI (not an
  “always latest” DOI) in the artifact submission.
- Vet the archived revision in a fresh environment and record the OS, Python
  version, resolved packages, commands, runtimes, and outcomes.
- Confirm that the archive contains every large checkpoint; Git hosting and
  Zenodo transfers can omit files managed outside ordinary Git.
