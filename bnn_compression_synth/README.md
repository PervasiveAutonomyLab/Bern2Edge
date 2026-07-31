# BNN compression and LUT-based synthesis

This directory reproduces the model metrics in paper Table I for Adult,
Covertype, and HIGGS-Small:

```bash
python bnn_compression_synth/reproduce_table_i.py
```

The software script evaluates the shipped checkpoints stored in the dataset
directories and writes the aggregate CSV files here. By default, FPGA
measurements are read from `table_i_hls_results.csv`.

To generate HLS source from one canonical seed for every Bernstein and ReLU
row without Vitis, run:

```bash
python bnn_compression_synth/generate_and_synthesize_table_i.py --generate-only
```

This generates all 18 C++ HLS projects, including ROMs, testbenches, test data,
golden outputs, and TCL scripts. It does not calculate hardware metrics.

After loading Vitis, omit `--generate-only` to run csim and csynth, collect
metrics from the fresh reports, and compare latency, DSP, BRAM, and LUT counts
with the paper:

```bash
source <Vitis>/2024.1/settings64.sh
python bnn_compression_synth/generate_and_synthesize_table_i.py --jobs 4
```

The reusable compiler and instructions for compiling any compatible `.pth`
checkpoint are in [`hls/`](../hls/README.md).

Student training remains in each dataset directory:

- `Adult/run_kd_experiments.py`
- `cover_type/run_kd_experiments.py`
- `higgs_small/run_kd_experiments.py`
