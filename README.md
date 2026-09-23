# Bern2Edge: A Neurosymbolic Compiler for Edge Deployment via Bernstein Polynomial Networks

<p align="center">
  <a href="https://arxiv.org/abs/2608.20497">
    <img src="https://img.shields.io/badge/Paper-arXiv-b31b1b.svg" alt="Paper">
  </a>
  <a href="https://doi.org/10.5281/zenodo.21726441">
    <img src="https://zenodo.org/badge/DOI/10.5281/zenodo.21726441.svg" alt="Artifact DOI">
  </a>
  <a href="LICENSE">
    <img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="MIT License">
  </a>
  <img src="https://img.shields.io/badge/Python-3.9%2B-blue.svg" alt="Python 3.9+">
</p>

**Bern2Edge** is an end-to-end neurosymbolic framework for deploying neural networks on resource-constrained edge hardware. It distills a pretrained teacher into a compact **Bernstein Polynomial Network (BNN)** and supports two deployment paths:

1. **High-accuracy LUT deployment** — learned Bernstein activations are realized as compact lookup tables for FPGA inference.
2. **Interpretable rule deployment** — Bernstein activation geometry is converted into symbolic rules over the input space, with optional fallback models.

This repository accompanies the **2026 IEEE Transactions on Computer-Aided Design of Integrated Circuits and Systems (TCAD)** journal-track paper, scheduled for presentation at **CODES 2026, ESWEEK 2026**.

**Authors:** Malak Gamal El-Din, Yifan Zhang, Yasser Shoukry, Sitao Huang, and Salma Elmalaki

* **Paper:** [arXiv:2608.20497](https://arxiv.org/abs/2608.20497)
* **Archived artifact:** [Zenodo 10.5281/zenodo.21726441](https://doi.org/10.5281/zenodo.21726441)

## Artifact badges

Bern2Edge was awarded all three ACM artifact badges:

<p align="center">
  <img src="figures/badges/code_available.png" alt="Code Available" width="30%">
  &nbsp;&nbsp;
  <img src="figures/badges/code_reviewed.png" alt="Code Reviewed" width="30%">
  &nbsp;&nbsp;
  <img src="figures/badges/code_reproducible.png" alt="Code Reproducible" width="30%">
</p>

- **Code Available** — the research artifact is publicly available.
- **Code Reviewed** — the artifact was independently reviewed.
- **Code Reproducible** — the reported results supported by the artifact were independently reproduced.

| Compression                                            | Hardware deployment                                                                                               | Interpretable deployment                                                             |
| ------------------------------------------------------ | ----------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------ |
| Up to **+2.12 pp** accuracy over matched ReLU networks | Up to **99.8% lower latency** and **95.2% lower BRAM** than the W8A8 teacher while staying within 0.5 pp accuracy | Up to **89.0% lower DSP usage** for the rule path, with a 1.5 pp total-accuracy cost |

## How Bern2Edge works

<p align="center">
  <a href="figures/fig1.pdf">
    <img src="figures/fig1.png" alt="Bern2Edge pipeline" width="92%">
  </a>
</p>

A pretrained teacher is compressed through knowledge distillation into a smaller BNN. The trained model can then follow either a **LUT-based hardware path** for high-fidelity FPGA deployment or a **symbolic rule path** for interpretable inference.

The repository also includes experiments for robustness certification, distribution shift, low-power FPGA deployment, and Bernstein FFN substitution in TinyBERT4.

## Start here

| I want to...                                                                 | Go to                                              |
| ---------------------------------------------------------------------------- | -------------------------------------------------- |
| Reuse the Bernstein layers, models, KD code, data loaders, or rule extractor | [`bern2edge/`](bern2edge/)                         |
| Compile a compatible BNN or rule set to Vitis HLS                            | [`hls/`](hls/)                                     |
| Train compressed Bernstein/ReLU students                                     | `Adult/`, `cover_type/`, `higgs_small/`            |
| Extract and evaluate symbolic rules                                          | [`Adult/`](Adult/)                                 |
| Reproduce compression + FPGA results                                         | [`bnn_compression_synth/`](bnn_compression_synth/) |
| Reproduce matched-hardware Covertype results                                 | [`cover_type/`](cover_type/)                       |
| Reproduce end-to-end deployment results                                      | [`end_to_end_results/`](end_to_end_results/)       |
| Run robustness certification                                                 | [`MAGIC/`](MAGIC/)                                 |
| Evaluate geographic/temporal distribution shift                              | [`ACS/`](ACS/)                                     |
| Run the TinyBERT4 FFN experiment                                             | [`Transformer/`](Transformer/)                     |
| Find the command for a paper table or figure                                 | [`RESULTS.md`](RESULTS.md)                         |

## Installation

Run all commands from the repository root.

```bash
git clone https://github.com/PervasiveAutonomyLab/Bern2Edge.git
cd Bern2Edge

python -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip
python -m pip install -e .
```

Python **3.10 is recommended**; Python **3.9+** is supported.

See [`INSTALL.md`](INSTALL.md) for installation details and optional FPGA setup, and [`REQUIREMENTS.md`](REQUIREMENTS.md) for software, hardware, storage, network, and runtime requirements.

> **Vitis is optional.** You do not need Vitis to evaluate checkpoints, extract rules, render paper results, or generate HLS source. AMD Vitis HLS/Vivado 2024.1 is only required for fresh FPGA synthesis and hardware metrics.

## Quick start

### Check the installation

These commands use committed result files, require no dataset download, and finish in seconds:

```bash
python Adult/make_table3.py
python MAGIC/make_table5.py
python Transformer/make_table_xii.py
```

### Re-evaluate shipped checkpoints

For a live checkpoint evaluation:

```bash
python cover_type/reproduce_table_ii.py
```

The first run downloads and caches the Covertype dataset. The script evaluates all 50 shipped Table II checkpoints and verifies the reported five-fold results.

### Generate FPGA source without Vitis

```bash
python bnn_compression_synth/generate_and_synthesize_table_i.py --generate-only
```

This generates synthesizable C++, ROMs, testbenches, test vectors, golden outputs, and Vitis TCL scripts without running synthesis.

## Repository structure

```text
Bern2Edge/
├── bern2edge/              Reusable Python package
│   ├── bernstein.py        Bernstein activation layer
│   ├── models.py           Shared fully connected models
│   ├── kdtrain.py          Knowledge-distillation utilities
│   ├── data.py             Dataset loaders
│   ├── train_utils.py      Shared training utilities
│   └── rule_extraction/    Regimes, rule generation, quantization
│
├── hls/                    Reusable BNN + rule-network HLS compilers
├── bnn_compression_synth/  Table I compression + synthesis aggregation
├── end_to_end_results/     Table VI end-to-end deployment results
│
├── Adult/                  Training, rules, ablations, KV260/XC7S15 deployment
├── cover_type/             Covertype training + matched-hardware experiment
├── higgs_small/            HIGGS-Small training + checkpoints
├── MAGIC/                  Rule extraction + certified robustness
├── ACS/                    Geographic/temporal distribution shift
├── Transformer/            TinyBERT4 FFN substitution + HLS generation
│
└── figures/                Figures used in this README
```

The key distinction is:

* **`bern2edge/`** contains reusable learning and rule-extraction code.
* **`hls/`** contains reusable model-to-HLS compilation code.
* The remaining directories contain experiment drivers, checkpoints, and published-result reproduction.

## Core workflows

### 1. Train a compressed Bernstein student

The tabular student-training drivers are:

```text
Adult/run_kd_experiments.py
cover_type/run_kd_experiments.py
higgs_small/run_kd_experiments.py
```

Shared models, Bernstein activations, dataset loaders, and KD utilities live under [`bern2edge/`](bern2edge/).

### 2. Compile a BNN checkpoint to HLS

The reusable compiler accepts compatible `.pth` checkpoints produced by the tabular Bern2Edge workflows.

```bash
python -m hls.bern2hls.cli compile \
  --dataset adult \
  --pth Adult/student_model_weights/kd_fc_14x16x2_bern_deg3_alpha0.85_T2_lr0.006_wd0.0001_seed6.pth \
  --out build/my_bnn
```

The generated project includes the kernel, fixed-point configuration, model ROMs, Bernstein LUTs, testbench, test data, and synthesis scripts.

For fresh synthesis:

```bash
source <Vitis>/2024.1/settings64.sh

python -m hls.bern2hls.cli synth \
  --root build/my_bnn \
  --csim \
  --vitis-hls vitis-run

python -m hls.bern2hls.cli collect \
  --root build/my_bnn \
  --csv build/my_bnn_metrics.csv
```

See [`hls/README.md`](hls/README.md) for the checkpoint contract, rule-network compiler, Transformer front-end, quantization, and synthesis workflow.

### 3. Extract symbolic rules

The main rule-extraction workflow is under `Adult/`:

```bash
python Adult/run_rule_extraction.py
python Adult/make_table3.py
```

For one architecture with a CART fallback:

```bash
python Adult/run_rule_extraction.py \
  --arch 14x16x2 \
  --fallback tree
```

Each run produces float and HLS-ready quantized rule artifacts plus evaluation metrics.

Existing rule artifacts can also be evaluated directly:

```bash
python Adult/evaluate_rule_artifacts.py path/to/rules.json
```

### 4. Run the Transformer extension

`Transformer/` replaces the FFN sublayers in all four TinyBERT4 encoder layers with narrower Bernstein or matched-width GeLU FFNs.

```bash
# Recompute SST-2 accuracy from the released weights
python Transformer/eval_release.py
python Transformer/make_table_xii.py

# Run one sentence through a released model
python Transformer/load_and_run.py bern_h312 "this movie was a delight"
```

See [`Transformer/README.md`](Transformer/README.md) for the three-stage training flow and HLS generation.

## Reproduce the published results

All commands below are run from the repository root. For exact artifact provenance, outputs, tolerances, and what is recomputed versus read from committed synthesis measurements, see [`RESULTS.md`](RESULTS.md) and the README inside each experiment directory.

| Result         | Experiment                               | Main command                                                                  |
| -------------- | ---------------------------------------- | ----------------------------------------------------------------------------- |
| **Table I**    | BNN compression + KV260 synthesis        | `python bnn_compression_synth/reproduce_table_i.py`                           |
| **Table II**   | Covertype under matched hardware budgets | `python cover_type/reproduce_table_ii.py`                                     |
| **Table III**  | Adult symbolic rule extraction           | `python Adult/run_rule_extraction.py` → `python Adult/make_table3.py`         |
| **Table IV**   | LUT BNN vs. rule-network hardware        | `python Adult/table4_rule_network_hardware/reproduce_table4.py`               |
| **Table V**    | MAGIC comparison with rule extractors    | `python MAGIC/make_table5.py`                                                 |
| **Table VI**   | End-to-end cross-dataset deployment      | `python end_to_end_results/reproduce_table_vi.py`                             |
| **Table VII**  | Spartan-7 XC7S15 deployment              | `python Adult/table7_xc7s15_deployment/reproduce_table7.py`                   |
| **Table VIII** | Rule-extraction penalty sweep            | `python Adult/run_penalty_sweep.py` → `python Adult/make_table8.py`           |
| **Table IX**   | Rule fallback ablation                   | `python Adult/table9_fallback_ablation/reproduce_table9.py`                   |
| **Table X**    | MAGIC certified robustness               | `python MAGIC/make_table_x.py`                                                |
| **Table XI**   | ACS geographic + temporal shift          | `python ACS/run_multiseed.py` → `python ACS/make_table_xi.py`                 |
| **Table XII**  | TinyBERT4 FFN substitution               | `python Transformer/eval_release.py` → `python Transformer/make_table_xii.py` |
| **Figure 9**   | Rule penalty curves                      | `python Adult/figure9_penalty_sweep/reproduce_figure9.py`                     |
| **Figure 10**  | Sparsity / BRAM trade-off                | `python Adult/figure10_sparsity_sweep/reproduce_figure10.py`                  |

### Fresh FPGA synthesis

HLS source generation does not require FPGA tools. To regenerate fresh latency/resource reports, load **AMD Vitis HLS and Vivado 2024.1** and run the corresponding synthesis driver without `--generate-only`.

For example:

```bash
source <Vitis>/2024.1/settings64.sh
python bnn_compression_synth/generate_and_synthesize_table_i.py --jobs 4
```

The full synthesis commands for Tables I, II, IV, VII, IX, and XII are listed in [`INSTALL.md`](INSTALL.md).

## Documentation

| File                                               | Purpose                                                          |
| -------------------------------------------------- | ---------------------------------------------------------------- |
| [`INSTALL.md`](INSTALL.md)                         | Installation, smoke tests, optional dependencies, FPGA synthesis |
| [`REQUIREMENTS.md`](REQUIREMENTS.md)               | Software, hardware, storage, network, and runtimes               |
| [`RESULTS.md`](RESULTS.md)                         | Result-by-result reproduction coverage and commands              |
| [`hls/README.md`](hls/README.md)                   | Reusable BNN/rule/Transformer hardware compiler                  |
| [`CITATION.cff`](CITATION.cff)                     | Repository citation metadata                                     |
| [`ARTIFACT_EVALUATION.md`](ARTIFACT_EVALUATION.md) | Original artifact-evaluation guide                               |

## Citation

If you use Bern2Edge, please cite the paper and repository.

```bibtex
@article{gamaleldin2026bern2edge,
  title   = {Bern2Edge: A Neurosymbolic Compiler for Edge Deployment via Bernstein Polynomial Networks},
  author  = {Gamal El-Din, Malak and Zhang, Yifan and Shoukry, Yasser and Huang, Sitao and Elmalaki, Salma},
  journal = {IEEE Transactions on Computer-Aided Design of Integrated Circuits and Systems},
  year    = {2026},
  note    = {Presented at CODES 2026, ESWEEK 2026},
  url     = {https://arxiv.org/abs/2608.20497}
}
```

The archived software artifact is available at [Zenodo DOI 10.5281/zenodo.21726441](https://doi.org/10.5281/zenodo.21726441).

## License

This project is released under the [MIT License](LICENSE).
