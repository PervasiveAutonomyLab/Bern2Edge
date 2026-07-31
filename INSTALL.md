# Installation

Run all commands from the repository root.

## 1. Create the environment

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -e .
```

Python 3.10 is recommended. See `REQUIREMENTS.md` for supported versions,
hardware, storage, and network requirements.

The installation needs internet access to download Python packages. Dataset
downloads are separate and are listed in `REQUIREMENTS.md`.

## 2. Smoke test

These commands render committed paper results without downloading data or
training models:

```bash
python Adult/make_table3.py
python Adult/make_table8.py
python MAGIC/make_table5.py
python MAGIC/make_table_x.py
python ACS/make_table_xi.py
python Transformer/make_table_xii.py
```

Each command should print a Markdown table and write or update its corresponding
LaTeX/Markdown result file. For example, the first Transformer row should be:

```text
| SST-2 | Acc. (%) | 90.37 | 90.02 | 90.48 | 89.11 | 90.02 |
```

If the tables render without an exception, the installation is working.

The smoke test renders committed results. Use the result-specific commands in
`RESULTS.md` for full experiments.

## 3. Basic functional test

Recompute Covertype Table II from the shipped checkpoints:

```bash
python cover_type/reproduce_table_ii.py
```

The dataset is downloaded and cached on first use. A successful run ends with:

```text
Verified 50 checkpoints and all 10 five-fold means.
Wrote cover_type/table_ii_checkpoint_results.csv
Wrote cover_type/table_ii_results.csv
```

Additional reproduction commands are organized by paper result in `README.md`.
The recommended reviewer sequence and pass criteria are in
`ARTIFACT_EVALUATION.md`.

## 4. Optional FPGA synthesis

Generating HLS source from a `.pth` checkpoint or an HLS-ready rule artifact
needs no additional Python packages beyond NumPy and PyTorch:

```bash
python bnn_compression_synth/generate_and_synthesize_table_i.py --generate-only
python cover_type/reproduce_table_ii_hardware.py --generate-only
python Adult/table4_rule_network_hardware/generate_and_synthesize_table_iv.py --generate-only
python Adult/table9_fallback_ablation/generate_and_synthesize_table_ix.py --generate-only
python Adult/table7_xc7s15_deployment/generate_and_synthesize_table_vii.py --generate-only
python Transformer/generate_and_synthesize_table_xii.py --generate-only
```

Fresh paper-metric reproduction additionally requires Vitis:

```bash
# Vitis HLS and Vivado 2024.1
source <Vitis>/2024.1/settings64.sh
python bnn_compression_synth/generate_and_synthesize_table_i.py --jobs 4
python cover_type/reproduce_table_ii_hardware.py --jobs 4
python Adult/table4_rule_network_hardware/generate_and_synthesize_table_iv.py --jobs 4
python Adult/table9_fallback_ablation/generate_and_synthesize_table_ix.py --jobs 4
python Adult/table7_xc7s15_deployment/generate_and_synthesize_table_vii.py --jobs 4
python Transformer/generate_and_synthesize_table_xii.py --jobs 4
```

See `hls/README.md` for generic checkpoint, rule quantization, and rule/fallback
compilation commands and the distinction between source generation and
synthesis.

## Optional MAGIC certification dependency

`auto_LiRPA` is needed only to recompute the ReLU certification column in
MAGIC Table X. It is not needed to render the committed table.

```bash
pip install --no-deps git+https://github.com/Verified-Intelligence/auto_LiRPA.git
pip install graphviz
```

The GitHub installation is used because the older PyPI package pins an
incompatible PyTorch version.
