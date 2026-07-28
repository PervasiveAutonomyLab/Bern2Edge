# Artifact status

The authors intend to apply for the **Available**, **Reviewed**, and
**Reproducible** CODES+ISSS / ESWEEK artifact badges.

## Available

The repository includes source code, documentation, model checkpoints, result
files, the accepted-paper PDF, and an MIT license. Before submission, archive
the exact artifact revision in a persistent public repository and add its
version-specific DOI to `CITATION.cff`.

## Reviewed

The artifact provides:

- installation and smoke-test instructions in `INSTALL.md`;
- hardware, software, storage, and network requirements in `REQUIREMENTS.md`;
- runnable commands and expected outputs in the root and experiment READMEs;
- committed inputs for quick, network-free table-rendering tests.

Model metrics are recomputed from shipped checkpoints where available.
Transcribed or previously collected FPGA measurements are identified as such.

## Reproducible

The artifact supports the following paper results:

| Result | Reproduction level |
|---|---|
| Table I | Accuracy and loss recomputed; HLS measurements transcribed |
| Table II | Accuracy recomputed; HLS measurements read from the shipped synthesis CSV |
| Tables III and VIII | Rules and metrics can be regenerated |
| Table IV | Five LUT checkpoints and rule/CART artifacts evaluated; KV260 measurements transcribed |
| Table VI | Teacher/student/rule accuracies recomputed; post-synthesis measurements transcribed |
| Table IX | Four fallback artifacts re-evaluated; HLS accuracy/resources joined from the committed synthesis summary |
| Figure 9 | Re-evaluated from 105 shipped rule/CART pairs; all plotted coordinates verified |
| Figure 10 | Re-evaluated from 13 shipped rule artifacts; BRAM measurements joined and all coordinates verified |
| Table V | Exact rendering from shipped five-fold metrics; retraining is approximate |
| Table X | Exact rendering; live certification from shipped weights |
| Table XI | Recomputed from shipped per-seed checkpoints |
| Table XII | Accuracy recomputed; HLS measurements transcribed |

Commands and limitations are summarized in `RESULTS.md`.

## Before submission

- Replace “The Bern2Edge Authors” in `LICENSE`, `CITATION.cff`, and
  `pyproject.toml` with the actual names.
- Add the version-specific archive DOI.
- Confirm that `Bern2Edge.pdf` is the accepted paper. Its current pages 14–17
  contain reviewer-response material.
- Confirm that all large checkpoints are included in the archived release.
