# BNN and rule-network hardware compilers

This directory contains the reusable fully connected Bernstein/ReLU compiler
used for Tables I and II and the rule-network compiler used for Tables IV and
IX. It accepts artifacts written by Bern2Edge, generates complete Vitis HLS
projects, runs simulation or synthesis, and extracts metrics from Vitis
reports. The Transformer compiler is not included.

## LUT implementation

Each trained Bernstein activation is converted to a 50-entry per-neuron ROM.
At inference time the hardware normalizes the linear-layer output with the
checkpoint's calibrated `input_bounds`, clamps it to `[0,1]`, reads adjacent
table entries, and linearly interpolates between them. Polynomial degree changes
the table values but not the lookup datapath. ReLU models use the same linear
layer templates without the activation ROM.

Weights, biases, activation tables, and predictions differ across training
seeds. Architecture constants, interfaces, HLS directives, and the compute
datapath are invariant for checkpoints with the same architecture and
activation.

## Checkpoint contract

Checkpoints produced by `Adult/run_kd_experiments.py`,
`higgs_small/run_kd_experiments.py`, and
`cover_type/run_kd_experiments.py` work directly. The compiler reads:

- linear parameters under `layers.{i}.weight` and `layers.{i}.bias`;
- Bernstein `bern_coeffs` and calibrated `input_bounds`;
- `arch`, `activation`, and `degree` checkpoint metadata.

The existing `kd_fc_<architecture>_<activation>_...pth` filename convention is
also supported as a metadata fallback. The selected dataset must match the
checkpoint dimensions: Adult `14→2`, HIGGS-Small `28→2`, and Covertype `54→7`.

## Generate hardware for any checkpoint

Run commands from the repository root:

```bash
python -m hls.bern2hls.cli compile \
  --dataset adult \
  --pth Adult/student_model_weights/kd_fc_14x16x2_bern_deg3_alpha0.85_T2_lr0.006_wd0.0001_seed6.pth \
  --out build/my_bnn
```

The output contains `include/`, `src/`, `tb/`, `data/`, and `script/`. Source
generation requires Python, NumPy, and PyTorch but does not require FPGA tools.

## What this repository does without Vitis

The `compile` command performs the complete model-to-HLS-source conversion. It
generates:

- synthesizable C++ kernels;
- architecture and fixed-point configuration headers;
- weight and bias ROMs or streamed weight files;
- per-neuron Bernstein activation LUTs;
- deterministic test inputs and golden outputs;
- a C++ testbench;
- Vitis TCL scripts for csim and csynth.

This stage is executable with Python, NumPy, and PyTorch alone. It does **not**
produce latency, LUT, DSP, FF, or BRAM measurements.

## Vitis synthesis required for hardware metrics

Hardware metrics are emitted by Vitis after it processes the generated C++ and
TCL files. The paper hardware environment is:

```bash
Vitis HLS and Vivado: 2024.1
Environment/setup command: source <Vitis>/2024.1/settings64.sh
```

After loading a compatible Vitis environment, run:

```bash
python -m hls.bern2hls.cli synth --root build/my_bnn --csim --vitis-hls vitis-run
python -m hls.bern2hls.cli synth --root build/my_bnn --vitis-hls vitis-run
python -m hls.bern2hls.cli collect --root build/my_bnn --csv build/my_bnn_metrics.csv
```

`collect` parses `csynth.xml`; latency, DSP, BRAM, FF, and LUT values are
measured outputs from Vitis, not constants in the compiler. Pass either the
modern `vitis-run` command or an older `vitis_hls` executable as appropriate
for the confirmed tool version.

The packaged test vectors are small deterministic compiler/csim checks. Dataset
accuracy in Tables I and II is evaluated separately by the experiment scripts
on all five held-out folds.

## Rule-network implementation

A rule is a conjunction of bands on learned linear projections:

```text
band_lo <= weight_vector · input < band_hi
```

The generated classifier evaluates these projections, selects the firing rule
with the highest purity, and invokes a fallback when no rule fires. Dense rule
sets store all projection weights. Sparse rule sets store the retained feature
indices and int8 weights. The public compiler supports these fallbacks:

- `none`: no fallback;
- `lr`: int8 logistic-regression weights embedded in the rule JSON;
- `tree`: a CART sidecar with `fix<16,8>` split thresholds;
- `network`: the full Bernstein BNN checkpoint;
- `small_nn`: the `14x4x2` Bernstein fallback with int8 linear weights.

The full-precision small-network control is not part of this project.

### Quantize an extracted rule set

The shared implementation in `bern2edge/rule_extraction/quantize.py` mirrors
the original Adult conversion scripts: each projection uses symmetric int8
quantization with `scale=max(abs(w))/127`, and bands, LR bias, and CART split
thresholds use `fix<16,8>`.

```bash
python -m hls.bern2hls.cli quantize-rules \
  --rules path/to/rules_float.json \
  --output path/to/rules_int8.json
```

For a CART fallback, quantize the JSON and sidecar together:

```bash
python -m hls.bern2hls.cli quantize-rules \
  --rules path/to/rules_float.json \
  --output path/to/rules_int8.json \
  --cart path/to/fallback_tree_float.npz \
  --cart-output path/to/fallback_tree_int.npz
```

Full and small Bernstein fallback checkpoints are not modified by this step.
The `small_nn` HLS generator quantizes its linear weights to int8 while emitting
the hardware ROMs.

### Generate rule-network hardware

LR parameters are read from the rule JSON:

```bash
python -m hls.bern2hls.cli compile-rules \
  --rules path/to/rules_int8.json \
  --fallback lr \
  --out build/my_rules_lr
```

CART and BNN fallbacks take a sidecar:

```bash
python -m hls.bern2hls.cli compile-rules \
  --rules path/to/rules_int8.json \
  --fallback tree \
  --fallback-model path/to/fallback_tree_int.npz \
  --out build/my_rules_tree
```

Replace `tree` and its model with `network` plus a `.pth`, or `small_nn` plus a
`.pt`/`.pth`, as appropriate. Use `--fallback-only` to emit the isolated
fallback kernel used for Table IX resource measurements. Optional
`--test-data data.npz` adds csim vectors; the NPZ must contain `X_test` and
`y_test` (or `X` and `y`). Without it, the generated project is ready for
csynth but not csim.

## Paper experiments

- `bnn_compression_synth/generate_and_synthesize_table_i.py` regenerates all
  18 Bernstein and ReLU hardware rows in Table I from one canonical checkpoint
  per row.
- `cover_type/reproduce_table_ii_hardware.py` regenerates the ten hardware
  designs used to select Table II's five matched-budget pairs.
- `Adult/table4_rule_network_hardware/generate_and_synthesize_table_iv.py`
  regenerates the five rule-classifier rows in Table IV.
- `Adult/table9_fallback_ablation/generate_and_synthesize_table_ix.py`
  regenerates the four full classifiers and four fallback-only designs in
  Table IX.
- `Transformer/generate_and_synthesize_table_xii.py` losslessly slices layer 0
  from the five shipped Transformer checkpoints and regenerates the encoder
  layer designs used for Table XII.
- `Adult/table7_xc7s15_deployment/generate_and_synthesize_table_vii.py`
  regenerates the six BNN and two rule designs targeting XC7S15 at 15 ns and
  optionally runs Vivado post-route implementation for LUT/FF/power metrics.

The drivers compare freshly collected reports with the committed paper metrics
and fail on any mismatch. Source generation is currently verified; synthesis
comparison remains optional until the paper's Vitis environment is available.

## Transformer compiler

The Transformer front-end accepts the clean Bern2Edge releases under
`Transformer/models/`:

- Bernstein or GeLU students with hidden width 312 or 600;
- the original GeLU teacher with hidden width 1200.

It extracts one encoder layer into a torch-free NPZ, preserving the FFN,
Bernstein coefficients/bounds, LayerNorm, and attention tensors exactly. The
generated kernel implements a streamed TinyBERT encoder layer with attention,
softmax, two LayerNorms, and either a degree-15 per-channel Bernstein LUT or the
matched GeLU polynomial.

Generate one paper configuration directly from a release:

```bash
python -m hls.bern2hls.cli compile-transformer \
  --model Transformer/models/release_bern_h312.pt \
  --scope layer \
  --out build/my_transformer
```

Use `--scope ffn` for the standalone FFN kernel. Add `--with-data` to emit the
large streamed weight text files needed by csim/cosim; csynth does not need
them. The Table XII driver synthesizes the encoder-layer projects because both
the full and FFN-only paper rows are derived from the layer hierarchy reports.
