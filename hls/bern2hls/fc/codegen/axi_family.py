"""AXI-family HLS project emitter (covertype).

Linear weights are streamed over m_axi from data/linear*_weights.txt; hidden
layers use a col-vec outer-product datapath split across three source files.
Lifted verbatim from covertype-HW/sweep_test_all/generate_all_models.py,
parameterized only by DatasetSpec fields.
"""

import os

import numpy as np

from ..bern_math import build_full_lut, compute_fused, ref_inference
from ..datasets import NEURON_LUT_SIZE
from ..model_extract import load_model, extract_params
from ...core.emit import arch_str, ceildiv, write_file, write_flat
from ...core.profiles import get_profile


def choose_col_vec_size(max_hidden):
    if max_hidden <= 8:
        return 8
    elif max_hidden <= 16:
        return 16
    else:
        return 32


def gen_config_hpp(cfg, spec, profile=None):
    profile = get_profile(profile)
    """Generate config.hpp for arbitrary layer count."""
    layer_sizes = cfg['layer_sizes']
    num_linear = len(layer_sizes) - 1  # number of linear layers
    hidden_dims = layer_sizes[1:-1]
    max_hidden = max(hidden_dims)
    col_vec_size = choose_col_vec_size(max_hidden)

    lines = []
    lines.append('#ifndef CONFIG_HPP')
    lines.append('#define CONFIG_HPP')
    lines.append('')
    lines.append('constexpr unsigned int ceildiv(unsigned int a, unsigned int b) {')
    lines.append('    return (a + b - 1) / b;')
    lines.append('}')
    lines.append('')
    lines.append(f'// Configuration: {cfg["name"]}  ({arch_str(layer_sizes)}, {cfg["act"]})')
    lines.append(f'// Model: {cfg["file"]}')
    lines.append('')
    lines.append(f'constexpr unsigned int NUM_HIDDEN_LAYERS = {len(hidden_dims)};')
    lines.append(f'constexpr unsigned int INPUT_DIM  = {spec.input_dim};')
    lines.append(f'constexpr unsigned int OUTPUT_DIM = {spec.output_dim};')
    for i, h in enumerate(hidden_dims):
        lines.append(f'constexpr unsigned int HIDDEN{i+1}_DIM = {h};')
    lines.append('')
    lines.append(f'constexpr unsigned int NEURON_LUT_SIZE = {NEURON_LUT_SIZE};')
    lines.append('')
    lines.append(f'constexpr unsigned int AXI_WIDTH = {profile.axi_width};')
    lines.append(f'constexpr unsigned int FIXED_TOTAL_BITS = {profile.data_w};')
    lines.append(f'constexpr unsigned int FIXED_INT_BITS = {profile.data_i};')
    lines.append('constexpr unsigned int AXI_BLOCK_SIZE = AXI_WIDTH / FIXED_TOTAL_BITS;')
    lines.append(f'constexpr unsigned int COL_VEC_SIZE = {col_vec_size};')
    lines.append('')
    lines.append('constexpr unsigned int INPUT_BLOCKS = ceildiv(INPUT_DIM, AXI_BLOCK_SIZE);')

    # For single hidden layer, define HIDDEN_DIM alias for compatibility
    if len(hidden_dims) == 1:
        lines.append(f'constexpr unsigned int HIDDEN_DIM = HIDDEN1_DIM;')

    # Per-linear-layer weight sizes
    # Layers: L1(input->h1), L2(h1->h2), ..., LN(hN->output)
    in_dims = [spec.input_dim] + hidden_dims
    out_dims = hidden_dims + [spec.output_dim]
    for i in range(num_linear):
        li = i + 1
        lines.append('')
        lines.append(f'// Linear{li}: {in_dims[i]} -> {out_dims[i]}')
        lines.append(f'constexpr unsigned int LINEAR{li}_IN_DIM  = {in_dims[i]};')
        lines.append(f'constexpr unsigned int LINEAR{li}_OUT_DIM = {out_dims[i]};')
        lines.append(f'constexpr unsigned int LINEAR{li}_WEIGHT_SIZE = LINEAR{li}_IN_DIM * LINEAR{li}_OUT_DIM;')
        if i < num_linear - 1:
            # Hidden layers use outer-product (col_vec_t)
            lines.append(f'constexpr unsigned int LINEAR{li}_COL_VECS = ceildiv(LINEAR{li}_OUT_DIM, COL_VEC_SIZE);')
            lines.append(f'constexpr unsigned int LINEAR{li}_TOTAL_COL_VECS = LINEAR{li}_IN_DIM * LINEAR{li}_COL_VECS;')
            lines.append(f'constexpr unsigned int LINEAR{li}_AXI_PER_COL_VEC = COL_VEC_SIZE / AXI_BLOCK_SIZE;')
            lines.append(f'constexpr unsigned int LINEAR{li}_WEIGHT_AXI_BLOCKS = LINEAR{li}_TOTAL_COL_VECS * LINEAR{li}_AXI_PER_COL_VEC;')
        else:
            # Last layer uses inner-product (flat row-major)
            lines.append(f'constexpr unsigned int LINEAR{li}_WEIGHT_BLOCKS = ceildiv(LINEAR{li}_WEIGHT_SIZE, AXI_BLOCK_SIZE);')

    lines.append('')
    lines.append('#endif // CONFIG_HPP')
    return '\n'.join(lines) + '\n'


def gen_types_hpp(profile=None):
    """Generate types.hpp (same for all models)."""
    profile = get_profile(profile)
    lines = ['#ifndef TYPES_HPP', '#define TYPES_HPP', '',
             '#include <ap_fixed.h>', '#include <hls_vector.h>',
             '#include <hls_stream.h>', '#include "config.hpp"', '']
    macro = profile.macro_block()
    if macro:
        lines += macro + ['']
    lines.append(f'typedef ap_fixed<{profile.data_t_decl()}> data_t;')
    lines.append('typedef data_t weight_t;')
    lines.append('typedef data_t bias_t;')
    lines.append(f'typedef ap_fixed<{profile.acc_t_decl()}> acc_t;')
    if profile.prod_type:
        lines.append(f'typedef ap_fixed<{profile.prod_t_decl()}> {profile.prod_type};'
                     f'{profile.prod_comment}')
    lines += ['',
              'typedef hls::vector<data_t, AXI_BLOCK_SIZE> axi_block_t;',
              'typedef hls::vector<data_t, COL_VEC_SIZE> col_vec_t;',
              '', '#endif // TYPES_HPP']
    return '\n'.join(lines) + '\n'


def gen_linear_simple_hpp(cfg):
    """Generate linear_simple.hpp — COL_VEC_SIZE comes from config.hpp."""
    layer_sizes = cfg['layer_sizes']
    num_linear = len(layer_sizes) - 1

    lines = []
    lines.append('#ifndef LINEAR_SIMPLE_HPP')
    lines.append('#define LINEAR_SIMPLE_HPP')
    lines.append('')
    lines.append('#include "types.hpp"')
    lines.append('#include "config.hpp"')
    lines.append('')
    lines.append('constexpr unsigned int AXI_PER_COL_VEC = COL_VEC_SIZE / AXI_BLOCK_SIZE;')
    lines.append('')
    lines.append('void load_input_simple(')
    lines.append('    data_t input_dst[],')
    lines.append('    const axi_block_t input_src[],')
    lines.append('    unsigned int num_blocks,')
    lines.append('    unsigned int dim')
    lines.append(');')
    lines.append('')
    lines.append('void load_linear_weights_colvec(')
    lines.append('    col_vec_t weights_dst[],')
    lines.append('    const axi_block_t weights_src[],')
    lines.append('    unsigned int in_dim,')
    lines.append('    unsigned int out_dim')
    lines.append(');')
    lines.append('')
    lines.append('void compute_linear_outer(')
    lines.append('    data_t output[],')
    lines.append('    const data_t input[],')
    lines.append('    const col_vec_t weights[],')
    lines.append('    const data_t bias[],')
    lines.append('    unsigned int out_dim,')
    lines.append('    unsigned int in_dim')
    lines.append(');')
    lines.append('')
    lines.append('void load_linear_weights_flat(')
    lines.append('    data_t weights_dst[],')
    lines.append('    const axi_block_t weights_src[],')
    lines.append('    unsigned int num_blocks,')
    lines.append('    unsigned int total_elements')
    lines.append(');')
    lines.append('')
    lines.append('void compute_linear_inner(')
    lines.append('    data_t output[],')
    lines.append('    const data_t input[],')
    lines.append('    const data_t weights[],')
    lines.append('    const data_t bias[],')
    lines.append('    unsigned int out_dim,')
    lines.append('    unsigned int in_dim')
    lines.append(');')
    lines.append('')
    lines.append('#endif // LINEAR_SIMPLE_HPP')
    return '\n'.join(lines) + '\n'


LINEAR_SIMPLE_CPP = """\
#include "../include/linear_simple.hpp"

void load_input_simple(
    data_t input_dst[],
    const axi_block_t input_src[],
    unsigned int num_blocks,
    unsigned int dim
)
{
    #pragma HLS inline off
    LOAD_INPUT_LOOP:
    for (unsigned int b = 0; b < num_blocks; b++) {
        #pragma HLS pipeline II=1
        axi_block_t block = input_src[b];
        unsigned int base_idx = b * AXI_BLOCK_SIZE;
        for (unsigned int i = 0; i < AXI_BLOCK_SIZE; i++) {
            #pragma HLS unroll
            unsigned int idx = base_idx + i;
            if (idx < dim) {
                input_dst[idx] = block[i];
            }
        }
    }
}

void load_linear_weights_colvec(
    col_vec_t weights_dst[],
    const axi_block_t weights_src[],
    unsigned int in_dim,
    unsigned int out_dim
)
{
    #pragma HLS inline off
    unsigned int col_vecs_per_col = ceildiv(out_dim, COL_VEC_SIZE);
    unsigned int total_col_vecs = in_dim * col_vecs_per_col;

    LOAD_WEIGHTS_CV_LOOP:
    for (unsigned int cv = 0; cv < total_col_vecs; cv++) {
        #pragma HLS pipeline II=AXI_PER_COL_VEC
        col_vec_t vec;
        for (unsigned int a = 0; a < AXI_PER_COL_VEC; a++) {
            unsigned int axi_idx = cv * AXI_PER_COL_VEC + a;
            axi_block_t axi_block = weights_src[axi_idx];
            for (unsigned int i = 0; i < AXI_BLOCK_SIZE; i++) {
                #pragma HLS unroll
                vec[a * AXI_BLOCK_SIZE + i] = axi_block[i];
            }
        }
        unsigned int cv_in_col = cv % col_vecs_per_col;
        unsigned int base_out = cv_in_col * COL_VEC_SIZE;
        for (unsigned int i = 0; i < COL_VEC_SIZE; i++) {
            #pragma HLS unroll
            if (base_out + i >= out_dim) {
                vec[i] = data_t(0);
            }
        }
        weights_dst[cv] = vec;
    }
}

void compute_linear_outer(
    data_t output[],
    const data_t input[],
    const col_vec_t weights[],
    const data_t bias[],
    unsigned int out_dim,
    unsigned int in_dim
)
{
    #pragma HLS inline off
    unsigned int col_vecs_per_col = ceildiv(out_dim, COL_VEC_SIZE);

    COMPUTE_CV_LOOP:
    for (unsigned int cv = 0; cv < col_vecs_per_col; cv++) {
        acc_t acc[COL_VEC_SIZE];
        #pragma HLS array_partition variable=acc complete
        for (unsigned int i = 0; i < COL_VEC_SIZE; i++) {
            #pragma HLS unroll
            unsigned int idx = cv * COL_VEC_SIZE + i;
            acc[i] = (idx < out_dim) ? acc_t(bias[idx]) : acc_t(0);
        }

        COMPUTE_J_LOOP:
        for (unsigned int j = 0; j < in_dim; j++) {
            #pragma HLS pipeline II=1
            data_t x_val = input[j];
            col_vec_t w_vec = weights[j * col_vecs_per_col + cv];
            for (unsigned int i = 0; i < COL_VEC_SIZE; i++) {
                #pragma HLS unroll
                acc[i] += acc_t(x_val) * acc_t(w_vec[i]);
            }
        }

        for (unsigned int i = 0; i < COL_VEC_SIZE; i++) {
            #pragma HLS unroll
            unsigned int idx = cv * COL_VEC_SIZE + i;
            if (idx < out_dim) {
                output[idx] = data_t(acc[i]);
            }
        }
    }
}

void load_linear_weights_flat(
    data_t weights_dst[],
    const axi_block_t weights_src[],
    unsigned int num_blocks,
    unsigned int total_elements
)
{
    #pragma HLS inline off
    LOAD_FLAT_LOOP:
    for (unsigned int b = 0; b < num_blocks; b++) {
        #pragma HLS pipeline II=1
        axi_block_t block = weights_src[b];
        unsigned int base_idx = b * AXI_BLOCK_SIZE;
        for (unsigned int i = 0; i < AXI_BLOCK_SIZE; i++) {
            #pragma HLS unroll
            unsigned int idx = base_idx + i;
            if (idx < total_elements) {
                weights_dst[idx] = block[i];
            }
        }
    }
}

void compute_linear_inner(
    data_t output[],
    const data_t input[],
    const data_t weights[],
    const data_t bias[],
    unsigned int out_dim,
    unsigned int in_dim
)
{
    #pragma HLS inline off
    LINEAR_OUT_LOOP:
    for (unsigned int o = 0; o < out_dim; o++) {
        acc_t acc = acc_t(bias[o]);
        LINEAR_IN_LOOP:
        for (unsigned int i = 0; i < in_dim; i++) {
            #pragma HLS unroll factor=8
            acc += acc_t(input[i]) * acc_t(weights[o * in_dim + i]);
        }
        output[o] = data_t(acc);
    }
}
"""


def gen_activation_hpp(cfg):
    """Generate activation.hpp with per-layer activation declarations."""
    hidden_dims = cfg['layer_sizes'][1:-1]
    num_act = len(hidden_dims)
    lines = []
    lines.append('#ifndef ACTIVATION_HPP')
    lines.append('#define ACTIVATION_HPP')
    lines.append('')
    lines.append('#include "types.hpp"')
    lines.append('#include "config.hpp"')
    lines.append('')
    if num_act == 1:
        lines.append('void compute_activation_layer(')
        lines.append('    data_t output[],')
        lines.append('    const data_t input[],')
        lines.append('    unsigned int num_neurons')
        lines.append(');')
    else:
        for i in range(num_act):
            lines.append(f'void compute_activation_L{i+1}(')
            lines.append('    data_t output[],')
            lines.append('    const data_t input[],')
            lines.append('    unsigned int num_neurons')
            lines.append(');')
            if i < num_act - 1:
                lines.append('')
    lines.append('')
    lines.append('#endif // ACTIVATION_HPP')
    return '\n'.join(lines) + '\n'


def gen_activation_cpp_bern(cfg):
    """Bernstein per-neuron full LUT + lerp activation for multi-layer."""
    hidden_dims = cfg['layer_sizes'][1:-1]
    num_act = len(hidden_dims)

    lines = []
    lines.append('#include "../include/activation.hpp"')
    lines.append('#include "../include/activation_lut_rom.hpp"')
    lines.append('')
    lines.append('static data_t activate_lerp(data_t x_norm, const data_t lut[NEURON_LUT_SIZE]) {')
    lines.append('    #pragma HLS inline')
    lines.append('    if (x_norm < data_t(0)) x_norm = data_t(0);')
    lines.append('    if (x_norm > data_t(1)) x_norm = data_t(1);')
    lines.append('    const data_t scale = data_t(NEURON_LUT_SIZE - 1);')
    lines.append('    data_t pos = x_norm * scale;')
    lines.append('    unsigned int idx_lo = (unsigned int)(pos);')
    lines.append('    if (idx_lo >= NEURON_LUT_SIZE - 1) idx_lo = NEURON_LUT_SIZE - 2;')
    lines.append('    unsigned int idx_hi = idx_lo + 1;')
    lines.append('    data_t frac = pos - data_t(idx_lo);')
    lines.append('    data_t val_lo = lut[idx_lo];')
    lines.append('    data_t val_hi = lut[idx_hi];')
    lines.append('    return val_lo + frac * (val_hi - val_lo);')
    lines.append('}')

    if num_act == 1:
        lines.append('')
        lines.append('void compute_activation_layer(data_t output[], const data_t input[], unsigned int num_neurons) {')
        lines.append('    #pragma HLS inline off')
        lines.append('    ACTIVATION_LOOP:')
        lines.append('    for (unsigned int n = 0; n < num_neurons; n++) {')
        lines.append('        #pragma HLS pipeline II=1')
        lines.append('        output[n] = activate_lerp(input[n], NEURON_ACT_LUT[n]);')
        lines.append('    }')
        lines.append('}')
    else:
        for i in range(num_act):
            lines.append('')
            lines.append(f'void compute_activation_L{i+1}(data_t output[], const data_t input[], unsigned int num_neurons) {{')
            lines.append('    #pragma HLS inline off')
            lines.append(f'    ACTIVATION_L{i+1}_LOOP:')
            lines.append('    for (unsigned int n = 0; n < num_neurons; n++) {')
            lines.append('        #pragma HLS pipeline II=1')
            lines.append(f'        output[n] = activate_lerp(input[n], NEURON_ACT_LUT_L{i+1}[n]);')
            lines.append('    }')
            lines.append('}')

    return '\n'.join(lines) + '\n'


def gen_activation_cpp_relu(cfg):
    """ReLU activation for multi-layer."""
    hidden_dims = cfg['layer_sizes'][1:-1]
    num_act = len(hidden_dims)

    lines = []
    lines.append('#include "../include/activation.hpp"')
    lines.append('')

    if num_act == 1:
        lines.append('void compute_activation_layer(data_t output[], const data_t input[], unsigned int num_neurons) {')
        lines.append('    #pragma HLS inline off')
        lines.append('    ACTIVATION_LOOP:')
        lines.append('    for (unsigned int n = 0; n < num_neurons; n++) {')
        lines.append('        #pragma HLS pipeline II=1')
        lines.append('        data_t x = input[n];')
        lines.append('        output[n] = (x > data_t(0)) ? x : data_t(0);')
        lines.append('    }')
        lines.append('}')
    else:
        for i in range(num_act):
            if i > 0:
                lines.append('')
            lines.append(f'void compute_activation_L{i+1}(data_t output[], const data_t input[], unsigned int num_neurons) {{')
            lines.append('    #pragma HLS inline off')
            lines.append(f'    ACTIVATION_L{i+1}_LOOP:')
            lines.append('    for (unsigned int n = 0; n < num_neurons; n++) {')
            lines.append('        #pragma HLS pipeline II=1')
            lines.append('        data_t x = input[n];')
            lines.append('        output[n] = (x > data_t(0)) ? x : data_t(0);')
            lines.append('    }')
            lines.append('}')

    return '\n'.join(lines) + '\n'


def gen_model_hpp(cfg, spec):
    """Generate top-level header with function signature."""
    ds = spec.name
    guard = f'{ds.upper()}_MODEL_HPP'
    layer_sizes = cfg['layer_sizes']
    num_linear = len(layer_sizes) - 1

    lines = []
    lines.append(f'#ifndef {guard}')
    lines.append(f'#define {guard}')
    lines.append('')
    lines.append('#include "linear_simple.hpp"')
    lines.append('#include "activation.hpp"')
    lines.append('')
    lines.append(f'void {ds}_top(')
    lines.append('    data_t output[OUTPUT_DIM],')
    lines.append('    const axi_block_t input[INPUT_BLOCKS],')
    for i in range(num_linear):
        li = i + 1
        if i < num_linear - 1:
            lines.append(f'    const axi_block_t linear{li}_weights[LINEAR{li}_WEIGHT_AXI_BLOCKS],')
        else:
            lines.append(f'    const axi_block_t linear{li}_weights[LINEAR{li}_WEIGHT_BLOCKS]')
    lines.append(');')
    lines.append('')
    lines.append(f'#endif // {guard}')
    return '\n'.join(lines) + '\n'


def gen_top_cpp(cfg, spec):
    """Generate top-level C++ with multi-layer pipeline."""
    ds = spec.name
    layer_sizes = cfg['layer_sizes']
    num_linear = len(layer_sizes) - 1
    hidden_dims = layer_sizes[1:-1]
    num_act = len(hidden_dims)

    lines = []
    lines.append(f'#include "../include/{ds}_model.hpp"')
    lines.append('#include "../include/bias_rom.hpp"')
    lines.append('')

    for i in range(num_linear):
        li = i + 1
        if i < num_linear - 1:
            lines.append(f'static col_vec_t linear{li}_weights_buf[LINEAR{li}_TOTAL_COL_VECS];')
        else:
            lines.append(f'static data_t linear{li}_weights_buf[LINEAR{li}_WEIGHT_SIZE];')

    lines.append('')
    lines.append('static data_t input_buf[INPUT_DIM];')
    for i, h in enumerate(hidden_dims):
        lines.append(f'static data_t hidden{i+1}_buf[HIDDEN{i+1}_DIM];')
        lines.append(f'static data_t act{i+1}_buf[HIDDEN{i+1}_DIM];')
    lines.append('static data_t output_buf[OUTPUT_DIM];')
    lines.append('')

    # Function signature
    lines.append(f'void {ds}_top(')
    lines.append('    data_t output[OUTPUT_DIM],')
    lines.append('    const axi_block_t input[INPUT_BLOCKS],')
    for i in range(num_linear):
        li = i + 1
        if i < num_linear - 1:
            lines.append(f'    const axi_block_t linear{li}_weights[LINEAR{li}_WEIGHT_AXI_BLOCKS],')
        else:
            lines.append(f'    const axi_block_t linear{li}_weights[LINEAR{li}_WEIGHT_BLOCKS]')
    lines.append(')')
    lines.append('{')

    # HLS interface pragmas
    lines.append('    #pragma HLS INTERFACE m_axi port=output offset=slave bundle=gmem0 depth=OUTPUT_DIM')
    lines.append('    #pragma HLS INTERFACE m_axi port=input offset=slave bundle=gmem1 depth=INPUT_BLOCKS')
    for i in range(num_linear):
        li = i + 1
        if i < num_linear - 1:
            lines.append(f'    #pragma HLS INTERFACE m_axi port=linear{li}_weights offset=slave bundle=gmem1 depth=LINEAR{li}_WEIGHT_AXI_BLOCKS')
        else:
            lines.append(f'    #pragma HLS INTERFACE m_axi port=linear{li}_weights offset=slave bundle=gmem1 depth=LINEAR{li}_WEIGHT_BLOCKS')
    lines.append('    #pragma HLS INTERFACE s_axilite port=return bundle=control')
    lines.append('')

    # Array partitions
    lines.append('    #pragma HLS array_partition variable=input_buf cyclic factor=AXI_BLOCK_SIZE')
    for i in range(num_act):
        lines.append(f'    #pragma HLS array_partition variable=hidden{i+1}_buf cyclic factor=COL_VEC_SIZE')
        lines.append(f'    #pragma HLS array_partition variable=act{i+1}_buf cyclic factor=COL_VEC_SIZE')
    last_li = num_linear
    lines.append(f'    #pragma HLS array_partition variable=linear{last_li}_weights_buf cyclic factor=AXI_BLOCK_SIZE')
    lines.append('')

    # Load input
    lines.append('    load_input_simple(input_buf, input, INPUT_BLOCKS, INPUT_DIM);')
    lines.append('')

    # Load all weights
    for i in range(num_linear):
        li = i + 1
        if i < num_linear - 1:
            lines.append(f'    load_linear_weights_colvec(linear{li}_weights_buf, linear{li}_weights, LINEAR{li}_IN_DIM, LINEAR{li}_OUT_DIM);')
        else:
            lines.append(f'    load_linear_weights_flat(linear{li}_weights_buf, linear{li}_weights, LINEAR{li}_WEIGHT_BLOCKS, LINEAR{li}_WEIGHT_SIZE);')
    lines.append('')

    # Compute pipeline: linear -> activation -> ... -> linear_last
    for i in range(num_linear):
        li = i + 1
        if i < num_linear - 1:
            in_buf = 'input_buf' if i == 0 else f'act{i}_buf'
            lines.append(f'    compute_linear_outer(hidden{i+1}_buf, {in_buf}, linear{li}_weights_buf, LINEAR{li}_BIAS_ROM, LINEAR{li}_OUT_DIM, LINEAR{li}_IN_DIM);')
            if num_act == 1:
                lines.append(f'    compute_activation_layer(act{i+1}_buf, hidden{i+1}_buf, HIDDEN{i+1}_DIM);')
            else:
                lines.append(f'    compute_activation_L{i+1}(act{i+1}_buf, hidden{i+1}_buf, HIDDEN{i+1}_DIM);')
        else:
            in_buf = f'act{num_act}_buf'
            lines.append(f'    compute_linear_inner(output_buf, {in_buf}, linear{li}_weights_buf, LINEAR{li}_BIAS_ROM, LINEAR{li}_OUT_DIM, LINEAR{li}_IN_DIM);')
    lines.append('')

    # Store output
    lines.append('    STORE_OUTPUT_LOOP:')
    lines.append('    for (unsigned int i = 0; i < OUTPUT_DIM; i++) {')
    lines.append('        #pragma HLS pipeline II=1')
    lines.append('        output[i] = output_buf[i];')
    lines.append('    }')
    lines.append('}')
    return '\n'.join(lines) + '\n'


def gen_tb_cpp(cfg, spec, profile=None):
    profile = get_profile(profile)
    """Generate testbench for arbitrary layer count."""
    ds = spec.name
    layer_sizes = cfg['layer_sizes']
    num_linear = len(layer_sizes) - 1

    lines = []
    lines.append('#include <iostream>')
    lines.append('#include <fstream>')
    lines.append('#include <cmath>')
    lines.append('#include <iomanip>')
    lines.append('#include <string>')
    lines.append(f'#include "../include/{ds}_model.hpp"')
    lines.append('')
    lines.append('#ifndef NUM_TEST_SAMPLES')
    lines.append('#define NUM_TEST_SAMPLES 10')
    lines.append('#endif')
    lines.append(f'#define TOLERANCE {profile.tb_tolerance}')
    lines.append('')

    # Static float arrays for weight loading (padded to AXI block boundaries)
    for i in range(num_linear):
        li = i + 1
        if i < num_linear - 1:
            lines.append(f'static float linear{li}_w_f[LINEAR{li}_WEIGHT_AXI_BLOCKS * AXI_BLOCK_SIZE];')
        else:
            lines.append(f'static float linear{li}_w_f[LINEAR{li}_WEIGHT_BLOCKS * AXI_BLOCK_SIZE];')
    lines.append('static float test_inputs_f[NUM_TEST_SAMPLES * INPUT_DIM];')
    lines.append('static float test_ref_f[NUM_TEST_SAMPLES * OUTPUT_DIM];')
    lines.append('')

    # AXI arrays
    for i in range(num_linear):
        li = i + 1
        if i < num_linear - 1:
            lines.append(f'static axi_block_t linear{li}_axi[LINEAR{li}_WEIGHT_AXI_BLOCKS];')
        else:
            lines.append(f'static axi_block_t linear{li}_axi[LINEAR{li}_WEIGHT_BLOCKS];')
    lines.append('')

    lines.append('bool load_values(const std::string& filename, float* data, int count) {')
    lines.append('    std::ifstream f(filename);')
    lines.append('    if (!f.is_open()) { std::cerr << "ERROR: Cannot open " << filename << std::endl; return false; }')
    lines.append('    for (int i = 0; i < count; i++) {')
    lines.append('        if (!(f >> data[i])) { std::cerr << "ERROR: Insufficient data in " << filename << std::endl; return false; }')
    lines.append('    }')
    lines.append('    return true;')
    lines.append('}')
    lines.append('')

    lines.append('std::string find_data_dir() {')
    lines.append('    const std::string candidates[] = {"../../data","../../../data","../../../../data","../../../../../data","data","../data"};')
    lines.append('    for (const auto& dir : candidates) {')
    lines.append('        std::ifstream f(dir + "/linear1_weights.txt");')
    lines.append('        if (f.is_open()) { f.close(); return dir; }')
    lines.append('    }')
    lines.append('    return "../../data";')
    lines.append('}')
    lines.append('')

    lines.append('int main() {')
    lines.append('    std::string data_dir = find_data_dir();')
    lines.append('    std::cout << "==============================" << std::endl;')
    lines.append(f'    std::cout << "  {cfg["name"]} Testbench" << std::endl;')
    lines.append('    std::cout << "==============================" << std::endl;')
    lines.append('    std::cout << "  Data dir: " << data_dir << std::endl << std::endl;')
    lines.append('')

    # Load all weight files (padded sizes match data files)
    for i in range(num_linear):
        li = i + 1
        if i < num_linear - 1:
            sz = f'LINEAR{li}_WEIGHT_AXI_BLOCKS * AXI_BLOCK_SIZE'
        else:
            sz = f'LINEAR{li}_WEIGHT_BLOCKS * AXI_BLOCK_SIZE'
        lines.append(f'    if (!load_values(data_dir+"/linear{li}_weights.txt", linear{li}_w_f, {sz})) return 1;')
    lines.append('    if (!load_values(data_dir+"/test_input.txt", test_inputs_f, NUM_TEST_SAMPLES*INPUT_DIM)) return 1;')
    lines.append('    if (!load_values(data_dir+"/test_output_ref.txt", test_ref_f, NUM_TEST_SAMPLES*OUTPUT_DIM)) return 1;')
    lines.append('')

    # Pack weights to AXI (data files already padded, single loop)
    for i in range(num_linear):
        li = i + 1
        if i < num_linear - 1:
            total_axi = f'LINEAR{li}_WEIGHT_AXI_BLOCKS'
        else:
            total_axi = f'LINEAR{li}_WEIGHT_BLOCKS'
        lines.append(f'    for (unsigned int i = 0; i < {total_axi}*AXI_BLOCK_SIZE; i++)')
        lines.append(f'        linear{li}_axi[i/AXI_BLOCK_SIZE][i%AXI_BLOCK_SIZE] = data_t(linear{li}_w_f[i]);')
        lines.append('')

    lines.append('    int total_pass=0, total_fail=0, pred_agree=0;')
    lines.append('    float global_max_err=0, global_sum_err=0;')
    lines.append('')
    lines.append('    for (int s = 0; s < NUM_TEST_SAMPLES; s++) {')
    lines.append('        axi_block_t input_axi[INPUT_BLOCKS];')
    lines.append('        for (unsigned int i = 0; i < INPUT_BLOCKS*AXI_BLOCK_SIZE; i++) {')
    lines.append('            float v = (i < INPUT_DIM) ? test_inputs_f[s*INPUT_DIM+i] : 0.0f;')
    lines.append('            input_axi[i/AXI_BLOCK_SIZE][i%AXI_BLOCK_SIZE] = data_t(v);')
    lines.append('        }')
    lines.append('        data_t output_hls[OUTPUT_DIM];')

    # Call top function with correct number of weight arrays
    args = ['output_hls', 'input_axi']
    for i in range(num_linear):
        args.append(f'linear{i+1}_axi')
    lines.append(f'        {ds}_top({", ".join(args)});')
    lines.append('')

    lines.append('        float max_err=0, sum_err=0;')
    lines.append('        int hls_pred=0, ref_pred=0;')
    lines.append('        float hls_max=-1e30f, ref_max=-1e30f;')
    lines.append('        std::cout << std::fixed << std::setprecision(6);')
    lines.append('        std::cout << "--- Sample " << s << " ---" << std::endl;')
    lines.append('        for (int o = 0; o < (int)OUTPUT_DIM; o++) {')
    lines.append('            float hv=(float)output_hls[o], rv=test_ref_f[s*OUTPUT_DIM+o];')
    lines.append('            float err=std::abs(hv-rv);')
    lines.append('            if (hv>hls_max){hls_max=hv;hls_pred=o;}')
    lines.append('            if (rv>ref_max){ref_max=rv;ref_pred=o;}')
    lines.append('            max_err=std::max(max_err,err); sum_err+=err;')
    lines.append('            std::cout << "  logit[" << o << "]: HLS=" << hv << "  ref=" << rv << "  err=" << err << std::endl;')
    lines.append('        }')
    lines.append('        float mean_err=sum_err/OUTPUT_DIM;')
    lines.append('        global_max_err=std::max(global_max_err,max_err);')
    lines.append('        global_sum_err+=mean_err;')
    lines.append('        if (hls_pred==ref_pred) pred_agree++;')
    lines.append('        if (max_err<TOLERANCE) total_pass++; else total_fail++;')
    lines.append('        std::cout << "  max_err=" << max_err << "  pred: HLS=" << hls_pred << " ref=" << ref_pred')
    lines.append('                  << (hls_pred==ref_pred?" AGREE":" **DISAGREE**") << std::endl;')
    lines.append('    }')
    lines.append('')
    lines.append('    std::cout << "\\n==============================" << std::endl;')
    lines.append('    std::cout << "  Summary: " << total_pass << " pass, " << total_fail << " fail" << std::endl;')
    lines.append('    std::cout << "  Global max err: " << global_max_err << std::endl;')
    lines.append('    std::cout << "  Avg mean err:   " << global_sum_err/NUM_TEST_SAMPLES << std::endl;')
    lines.append('    std::cout << "  Pred agreement: " << pred_agree << "/" << NUM_TEST_SAMPLES << std::endl;')
    lines.append('    std::cout << "==============================" << std::endl;')
    lines.append('    std::cout << (total_fail>0 ? "RESULT: SOME TESTS EXCEEDED TOLERANCE" : "RESULT: ALL TESTS PASSED") << std::endl;')
    lines.append('    return (total_fail > 0) ? 1 : 0;')
    lines.append('}')
    return '\n'.join(lines) + '\n'


def gen_tcl(cfg, spec, mode, profile=None):
    """Generate TCL script for csim or csynth."""
    profile = get_profile(profile)
    part, clock_ns = profile.part, profile.clock_ns
    ds = spec.name
    name = cfg['name']
    cmd = 'csim_design' if mode == 'csim' else 'csynth_design'
    proj = profile.proj_name_tmpl.format(name=name)
    return f"""\
open_project {proj}
set_top {ds}_top
add_files ../src/{ds}_top.cpp
add_files ../src/linear_simple.cpp
add_files ../src/activation.cpp
add_files -tb ../tb/{ds}_tb.cpp
open_solution "solution1"
set_part {{{part}}}
create_clock -period {clock_ns} -name default
{cmd}
exit
"""


# =====================================================================
# ROM header writers
# =====================================================================

def write_bias_rom(path, bias_list):
    """Write bias ROM header."""
    with open(path, 'w') as f:
        f.write('#ifndef BIAS_ROM_HPP\n#define BIAS_ROM_HPP\n\n')
        f.write('#include "types.hpp"\n#include "config.hpp"\n\n')
        for rom_name, bias in bias_list:
            dim = len(bias)
            f.write(f'static const data_t {rom_name}[{dim}] = {{\n')
            for i in range(dim):
                c = ',' if i < dim - 1 else ''
                f.write(f'    data_t({bias[i]:.10f}){c}\n')
            f.write('};\n\n')
        f.write('#endif // BIAS_ROM_HPP\n')


def write_activation_lut_rom(path, lut_list):
    """Write activation LUT ROM header."""
    with open(path, 'w') as f:
        f.write('#ifndef ACTIVATION_LUT_ROM_HPP\n#define ACTIVATION_LUT_ROM_HPP\n\n')
        f.write('#include "types.hpp"\n#include "config.hpp"\n\n')
        for arr_name, dim_name, lut in lut_list:
            h, lut_size = lut.shape
            f.write(f'static const data_t {arr_name}[{dim_name}][NEURON_LUT_SIZE] = {{\n')
            for n in range(h):
                f.write(f'    /* n{n:3d} */ {{')
                for i in range(lut_size):
                    if i > 0 and i % 10 == 0:
                        f.write('\n              ')
                    c = ', ' if i < lut_size - 1 else ''
                    f.write(f'{lut[n, i]:.10f}{c}')
                c = ',' if n < h - 1 else ''
                f.write(f'}}{c}\n')
            f.write('};\n\n')
        f.write('#endif // ACTIVATION_LUT_ROM_HPP\n')


# =====================================================================
# Main generation logic
# =====================================================================

def generate_one_model(cfg, spec, model_dir, output_dir, X_test, y_test, profile=None):
    """Generate a complete HLS project for one model configuration."""
    profile = get_profile(profile)
    ds = spec.name
    name = cfg['name']
    layer_sizes = cfg['layer_sizes']
    num_linear = len(layer_sizes) - 1
    hidden_dims = layer_sizes[1:-1]
    is_bern = cfg['act'] == 'bern'

    config_dir = os.path.join(output_dir, name)
    inc = os.path.join(config_dir, 'include')
    src = os.path.join(config_dir, 'src')
    tb = os.path.join(config_dir, 'tb')
    scr = os.path.join(config_dir, 'script')
    data_dir = os.path.join(config_dir, 'data')

    print(f"  [1] Loading model and extracting parameters...")
    model = load_model(cfg, model_dir, spec.input_dim)
    linears, bern_layers = extract_params(model, cfg)

    # Process parameters: fuse normalization for Bernstein, prepare LUTs
    linears_data = []  # (W, b) to write — may be fused
    luts = []          # per-hidden-layer LUT arrays (or None)
    bias_rom_list = [] # (rom_name, bias_array)

    for li in range(num_linear):
        W, b = linears[li]
        if li < num_linear - 1 and is_bern:
            # Hidden layer with Bernstein: fuse normalization
            bl = bern_layers[li]
            bounds = bl.input_bounds.data.numpy()
            W_fused, b_fused, _, _ = compute_fused(W, b, bounds)
            linears_data.append((W_fused, b_fused))
            bias_rom_list.append((f'LINEAR{li+1}_BIAS_ROM', b_fused))
            lut = build_full_lut(bl, NEURON_LUT_SIZE, spec.lut_grid)
            luts.append(lut)
        else:
            linears_data.append((W, b))
            bias_rom_list.append((f'LINEAR{li+1}_BIAS_ROM', b))
            luts.append(None)

    # Generate HLS source files
    print(f"  [2] Generating HLS source files...")
    write_file(os.path.join(inc, 'config.hpp'), gen_config_hpp(cfg, spec, profile))
    write_file(os.path.join(inc, 'types.hpp'), gen_types_hpp(profile))
    write_file(os.path.join(inc, 'linear_simple.hpp'), gen_linear_simple_hpp(cfg))
    write_file(os.path.join(inc, 'activation.hpp'), gen_activation_hpp(cfg))
    write_file(os.path.join(inc, f'{ds}_model.hpp'), gen_model_hpp(cfg, spec))
    write_file(os.path.join(src, 'linear_simple.cpp'), LINEAR_SIMPLE_CPP)
    write_file(os.path.join(src, f'{ds}_top.cpp'), gen_top_cpp(cfg, spec))

    if is_bern:
        write_file(os.path.join(src, 'activation.cpp'), gen_activation_cpp_bern(cfg))
    else:
        write_file(os.path.join(src, 'activation.cpp'), gen_activation_cpp_relu(cfg))

    write_file(os.path.join(tb, f'{ds}_tb.cpp'), gen_tb_cpp(cfg, spec, profile))
    for stage in profile.tcl_stages:
        write_file(os.path.join(scr, f'run_{stage}.tcl'), gen_tcl(cfg, spec, stage, profile))

    # Generate ROM headers
    print(f"  [3] Generating ROM headers...")
    write_bias_rom(os.path.join(inc, 'bias_rom.hpp'), bias_rom_list)

    if is_bern:
        if len(hidden_dims) == 1:
            lut_entries = [('NEURON_ACT_LUT', 'HIDDEN_DIM', luts[0])]
        else:
            lut_entries = []
            for i in range(len(hidden_dims)):
                lut_entries.append((f'NEURON_ACT_LUT_L{i+1}', f'HIDDEN{i+1}_DIM', luts[i]))
        write_activation_lut_rom(os.path.join(inc, 'activation_lut_rom.hpp'), lut_entries)

    # Generate weight data files
    print(f"  [4] Generating weight data files...")
    col_vec_size = choose_col_vec_size(max(hidden_dims))
    axi_block_size = 8  # AXI_WIDTH(256) / FIXED_TOTAL_BITS(32)
    for li in range(num_linear):
        W, b = linears_data[li]
        if li < num_linear - 1:
            # Outer-product: pad each input row to col_vec_size boundary
            out_dim = W.shape[0]  # W is (out, in) per PyTorch convention
            in_dim = W.shape[1]
            cvpc = ceildiv(out_dim, col_vec_size)  # col_vecs_per_col
            padded_cols = cvpc * col_vec_size
            W_padded = np.zeros((in_dim, padded_cols))
            W_padded[:, :out_dim] = W.T  # W.T is (in_dim, out_dim)
            write_flat(os.path.join(data_dir, f'linear{li+1}_weights.txt'), W_padded.flatten())
        else:
            # Inner-product: row-major, padded to AXI block boundary
            W_flat = W.flatten()
            padded_total = ceildiv(len(W_flat), axi_block_size) * axi_block_size
            W_padded = np.zeros(padded_total)
            W_padded[:len(W_flat)] = W_flat
            write_flat(os.path.join(data_dir, f'linear{li+1}_weights.txt'), W_padded)

    # Compute reference outputs
    print(f"  [5] Computing reference outputs...")
    ref = ref_inference(X_test, linears_data, luts, is_bern, NEURON_LUT_SIZE)

    write_flat(os.path.join(data_dir, 'test_input.txt'), X_test)
    write_flat(os.path.join(data_dir, 'test_output_ref.txt'), ref)
    with open(os.path.join(data_dir, 'test_labels.txt'), 'w') as f:
        for label in y_test:
            f.write(f'{label}\n')

    pred = ref.argmax(axis=1)
    print(f"  Done: {arch_str(layer_sizes)} {cfg['act']}, "
          f"{len(X_test)} test samples, preds: {pred[:5].tolist()}")
