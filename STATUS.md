# Artifact status

The authors are applying for the **Available**, **Reviewed**, and
**Reproducible** CODES 2026 / ESWEEK artifact badges. The badges are
independent. Claims below are limited to what this archive contains.

## Available

The artifact includes source code, documentation, model checkpoints, result
files, the paper PDF, and an MIT license.

**Not ready until completed:** archive the exact submitted revision in a
persistent public repository, add its version-specific DOI to `CITATION.cff`
and this file, and verify that `Bern2Edge.pdf` is tracked in that revision.

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
| Table IX | Four fallback artifacts re-evaluated |
| Figure 9 | Re-evaluated from 105 shipped rule/CART pairs; all plotted coordinates verified |
| Figure 10 | Re-evaluated from 13 shipped rule artifacts; all coordinates verified |
| Table V | Exact rendering from shipped five-fold metrics; retraining is approximate |
| Table X | Exact rendering; live certification from shipped weights |
| Table XI | Recomputed from shipped per-seed checkpoints |
| Table XII | Accuracy recomputed from shipped weights |

Commands and limitations are summarized in `RESULTS.md`.

## Submission blockers and final checks

- Add the version-specific archive DOI (not an “always latest” DOI).
- Add the archived DOI here: **TODO**.
- Track `Bern2Edge.pdf`; it was previously excluded by `.gitignore`.
- Confirm that `Bern2Edge.pdf` is the final accepted 14-page paper.
- Vet the archived revision on a clean machine and record the OS, Python
  version, resolved packages, commands, runtimes, and outcomes.
- Confirm that the archive contains every large checkpoint; Git hosting and
  Zenodo transfers can omit files managed outside ordinary Git.
