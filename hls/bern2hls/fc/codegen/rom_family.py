"""ROM-family HLS project emitter (adult / higgs).

All weights live in on-chip ROM headers; s_axilite-only interface; single
monolithic <ds>_model.cpp with tiered unroll. Lifted verbatim from
adult-HW/sweep_test_all/generate_all_models.py (byte-identical to the higgs
copy modulo dataset name and dims), parameterized only by DatasetSpec fields.
"""

import os

from ..bern_math import build_full_lut, compute_fused, ref_inference
from ..datasets import NEURON_LUT_SIZE
from ..model_extract import load_model, extract_params
from ...core.emit import arch_str, write_file, write_flat
from ...core.profiles import get_profile
from . import shared_basis as sb


def choose_unroll_factor(in_dim, out_dim, is_output_layer, strategy='tier1',
                         max_output_f=8, max_hidden_f=16):
    """Choose inner-loop unroll factor based on layer role and strategy.

    strategy: 'full' = always full unroll (baseline),
              'tier1' = partial unroll only on output layer,
              'tier2' = partial unroll on output + hidden->hidden layers.
    Returns an unroll factor (int), or None for full unroll.
    """
    if strategy == 'full':
        return None  # full unroll

    MAX_OUTPUT_F = max_output_f   # cap for output layer
    MAX_HIDDEN_F = max_hidden_f   # cap for hidden->hidden layers

    if is_output_layer:
        if in_dim <= MAX_OUTPUT_F:
            return None  # small enough, full unroll is fine
        return MAX_OUTPUT_F

    if strategy == 'tier2':
        if in_dim <= MAX_HIDDEN_F:
            return None
        return MAX_HIDDEN_F

    return None  # tier1: full unroll for non-output layers


def gen_config_hpp(cfg, spec, profile=None):
    """Generate config.hpp — simplified, no AXI weight constants."""
    profile = get_profile(profile)
    layer_sizes = cfg['layer_sizes']
    num_linear = len(layer_sizes) - 1
    hidden_dims = layer_sizes[1:-1]
    in_dims = [spec.input_dim] + hidden_dims
    out_dims = hidden_dims + [spec.output_dim]

    lines = []
    lines.append('#ifndef CONFIG_HPP')
    lines.append('#define CONFIG_HPP')
    lines.append('')
    lines.append(f'// Configuration: {cfg["name"]}  ({arch_str(layer_sizes)}, {cfg["act"]})')
    lines.append(f'// Model: {cfg["file"]}')
    lines.append('')
    lines.append(f'constexpr unsigned int NUM_HIDDEN_LAYERS = {len(hidden_dims)};')
    lines.append(f'constexpr unsigned int INPUT_DIM  = {spec.input_dim};')
    lines.append(f'constexpr unsigned int OUTPUT_DIM = {spec.output_dim};')
    for i, h in enumerate(hidden_dims):
        lines.append(f'constexpr unsigned int HIDDEN{i+1}_DIM = {h};')
    if len(hidden_dims) == 1:
        lines.append(f'constexpr unsigned int HIDDEN_DIM = HIDDEN1_DIM;')
    lines.append('')
    lines.append(f'constexpr unsigned int NEURON_LUT_SIZE = {NEURON_LUT_SIZE};')
    lines.append('')
    lines.append(f'constexpr unsigned int FIXED_TOTAL_BITS = {profile.data_w};')
    lines.append(f'constexpr unsigned int FIXED_INT_BITS = {profile.data_i};')
    lines.append('')
    for i in range(num_linear):
        li = i + 1
        lines.append(f'// Linear{li}: {in_dims[i]} -> {out_dims[i]}')
        lines.append(f'constexpr unsigned int LINEAR{li}_IN_DIM  = {in_dims[i]};')
        lines.append(f'constexpr unsigned int LINEAR{li}_OUT_DIM = {out_dims[i]};')
        lines.append('')
    lines.append('#endif // CONFIG_HPP')
    return '\n'.join(lines) + '\n'


def gen_types_hpp(profile=None):
    """Generate types.hpp — simplified, no AXI/col_vec types."""
    profile = get_profile(profile)
    lines = ['#ifndef TYPES_HPP', '#define TYPES_HPP', '',
             '#include <ap_fixed.h>', '#include "config.hpp"', '']
    macro = profile.macro_block()
    if macro:
        lines += macro + ['']
    lines.append(f'typedef ap_fixed<{profile.data_t_decl()}> data_t;')
    lines.append(f'typedef ap_fixed<{profile.acc_t_decl()}> acc_t;')
    if profile.prod_type:
        lines.append(f'typedef ap_fixed<{profile.prod_t_decl()}> {profile.prod_type};'
                     f'{profile.prod_comment}')
    lines += ['', '#endif // TYPES_HPP']
    return '\n'.join(lines) + '\n'


def gen_model_hpp(cfg, spec):
    """Generate <ds>_model.hpp — model function signature."""
    ds = spec.name
    guard = f'{ds.upper()}_MODEL_HPP'
    lines = []
    lines.append(f'#ifndef {guard}')
    lines.append(f'#define {guard}')
    lines.append('')
    lines.append('#include "types.hpp"')
    lines.append('#include "config.hpp"')
    lines.append('')
    lines.append(f'void {ds}_top(')
    lines.append('    const data_t input[INPUT_DIM],')
    lines.append('    data_t output[OUTPUT_DIM]')
    lines.append(');')
    lines.append('')
    lines.append(f'#endif // {guard}')
    return '\n'.join(lines) + '\n'


def gen_model_cpp(cfg, spec, unroll_strategy='tier1', profile=None):
    """Generate <ds>_model.cpp — linear + activation compute in one file."""
    profile = get_profile(profile)
    ds = spec.name
    layer_sizes = cfg['layer_sizes']
    num_linear = len(layer_sizes) - 1
    hidden_dims = layer_sizes[1:-1]
    num_act = len(hidden_dims)
    is_bern = cfg['act'] == 'bern'

    lines = []
    lines.append(f'#include "../include/{ds}_model.hpp"')
    lines.append('#include "../include/weight_rom.hpp"')
    lines.append('#include "../include/bias_rom.hpp"')
    shared = is_bern and profile.activation_impl == 'shared_basis'
    if is_bern:
        lines.append('#include "../include/shared_basis_rom.hpp"' if shared
                     else '#include "../include/activation_lut_rom.hpp"')
    lines.append('')

    # ---- activation helper (Bernstein only) ----
    if shared:
        lines += sb.gen_activate_shared(profile)
        lines.append('')
    elif is_bern:
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
        lines.append('')

    # ---- Top function ----
    lines.append(f'void {ds}_top(')
    lines.append('    const data_t input[INPUT_DIM],')
    lines.append('    data_t output[OUTPUT_DIM]')
    lines.append(')')
    lines.append('{')
    lines.append('    #pragma HLS INTERFACE s_axilite port=input bundle=control')
    lines.append('    #pragma HLS INTERFACE s_axilite port=output bundle=control')
    lines.append('    #pragma HLS INTERFACE s_axilite port=return bundle=control')
    lines.append('')

    # Declare intermediate buffers
    for i, h in enumerate(hidden_dims):
        lines.append(f'    data_t hidden{i+1}[HIDDEN{i+1}_DIM];')
        lines.append(f'    data_t act{i+1}[HIDDEN{i+1}_DIM];')
    lines.append(f'    data_t out_buf[OUTPUT_DIM];')
    lines.append('')

    # ==== Per-layer compute ====
    in_dims = [spec.input_dim] + hidden_dims
    out_dims = hidden_dims + [spec.output_dim]
    for li_idx in range(num_linear):
        li = li_idx + 1
        in_d = in_dims[li_idx]
        out_d = out_dims[li_idx]
        # Determine input/output buffer names
        if li_idx == 0:
            in_buf = 'input'
        else:
            in_buf = f'act{li_idx}'

        if li_idx < num_linear - 1:
            out_buf = f'hidden{li_idx+1}'
        else:
            out_buf = 'out_buf'

        is_output = (li_idx == num_linear - 1)
        uf = choose_unroll_factor(in_d, out_d, is_output, unroll_strategy,
                                  profile.max_output_unroll, profile.max_hidden_unroll)

        lines.append(f'    // --- Linear{li}: {in_d} -> {out_d} ---')
        lines.append(f'    LINEAR{li}_LOOP:')
        lines.append(f'    for (unsigned int o = 0; o < LINEAR{li}_OUT_DIM; o++) {{')
        if uf is None:
            # Full unroll: pipeline outer loop, fully unroll inner loop
            lines.append(f'        #pragma HLS pipeline II=1')
        lines.append(f'        acc_t acc = acc_t(LINEAR{li}_BIAS_ROM[o]);')
        lines.append(f'        for (unsigned int i = 0; i < LINEAR{li}_IN_DIM; i++) {{')
        if uf is None:
            lines.append(f'            #pragma HLS unroll')
        else:
            # Partial unroll: pipeline + partial unroll on inner loop
            lines.append(f'            #pragma HLS pipeline II=1')
            lines.append(f'            #pragma HLS unroll factor={uf}')
        if profile.narrow_mult:
            # Exact narrow product instead of widening both operands to acc_t:
            # data_t*data_t fits one DSP48E1 at 16-bit-class widths.
            pv = f'p{li}'
            lines.append(f'            {profile.prod_type} {pv} = {in_buf}[i] * '
                         f'LINEAR{li}_WEIGHT_ROM[o][i];{profile.narrow_mult_comment}')
            if profile.bind_op_mult:
                lines.append(f'            #pragma HLS BIND_OP variable={pv} op=mul '
                             f'impl={profile.bind_op_mult}')
            lines.append(f'            acc += {pv};')
        else:
            lines.append(f'            acc += acc_t({in_buf}[i]) * acc_t(LINEAR{li}_WEIGHT_ROM[o][i]);')
        lines.append(f'        }}')
        lines.append(f'        {out_buf}[o] = data_t(acc);')
        lines.append(f'    }}')
        lines.append('')

        # Apply activation after hidden layers
        if li_idx < num_linear - 1:
            act_idx = li_idx + 1
            if is_bern:
                if num_act == 1:
                    lut_name = 'NEURON_ACT_LUT'
                else:
                    lut_name = f'NEURON_ACT_LUT_L{act_idx}'
                lines.append(f'    // --- Activation{act_idx}: Bernstein LUT + lerp ---')
                if shared:
                    lines.append('    #pragma HLS ARRAY_PARTITION variable=BASIS_LUT complete dim=2')
                    lines.append('    #pragma HLS ARRAY_PARTITION variable=NEURON_COEFF complete dim=2')
                lines.append(f'    ACTIVATION{act_idx}_LOOP:')
                lines.append(f'    for (unsigned int n = 0; n < HIDDEN{act_idx}_DIM; n++) {{')
                lines.append(f'        #pragma HLS pipeline II=1')
                if shared:
                    lines.append(f'        act{act_idx}[n] = activate_shared('
                                 f'hidden{act_idx}[n], NEURON_COEFF[n]);')
                else:
                    lines.append(f'        act{act_idx}[n] = activate_lerp('
                                 f'hidden{act_idx}[n], {lut_name}[n]);')
                lines.append(f'    }}')
            else:
                lines.append(f'    // --- Activation{act_idx}: ReLU ---')
                lines.append(f'    ACTIVATION{act_idx}_LOOP:')
                lines.append(f'    for (unsigned int n = 0; n < HIDDEN{act_idx}_DIM; n++) {{')
                lines.append(f'        #pragma HLS pipeline II=1')
                lines.append(f'        data_t x = hidden{act_idx}[n];')
                lines.append(f'        act{act_idx}[n] = (x > data_t(0)) ? x : data_t(0);')
                lines.append(f'    }}')
            lines.append('')

    # Store output
    lines.append('    STORE_OUTPUT:')
    lines.append('    for (unsigned int i = 0; i < OUTPUT_DIM; i++) {')
    lines.append('        #pragma HLS unroll')
    lines.append('        output[i] = out_buf[i];')
    lines.append('    }')
    lines.append('}')
    return '\n'.join(lines) + '\n'


def gen_tb_cpp(cfg, spec, profile=None):
    profile = get_profile(profile)
    """Generate testbench for ROM-based design."""
    ds = spec.name
    layer_sizes = cfg['layer_sizes']

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
    lines.append('static float test_inputs_f[NUM_TEST_SAMPLES * INPUT_DIM];')
    lines.append('static float test_ref_f[NUM_TEST_SAMPLES * OUTPUT_DIM];')
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
    lines.append('        std::ifstream f(dir + "/test_input.txt");')
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
    lines.append('    if (!load_values(data_dir+"/test_input.txt", test_inputs_f, NUM_TEST_SAMPLES*INPUT_DIM)) return 1;')
    lines.append('    if (!load_values(data_dir+"/test_output_ref.txt", test_ref_f, NUM_TEST_SAMPLES*OUTPUT_DIM)) return 1;')
    lines.append('')
    lines.append('    int total_pass=0, total_fail=0, pred_agree=0;')
    lines.append('    float global_max_err=0, global_sum_err=0;')
    lines.append('')
    lines.append('    for (int s = 0; s < NUM_TEST_SAMPLES; s++) {')
    lines.append('        data_t input_hls[INPUT_DIM];')
    lines.append('        for (unsigned int i = 0; i < INPUT_DIM; i++)')
    lines.append('            input_hls[i] = data_t(test_inputs_f[s*INPUT_DIM+i]);')
    lines.append('')
    lines.append('        data_t output_hls[OUTPUT_DIM];')
    lines.append(f'        {ds}_top(input_hls, output_hls);')
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


ACC_TB_TMPL = """\
// Accuracy testbench: full {n}-sample {ds} test set, argmax vs labels.
// (The shipped {sc}-sample TB only checks logit tolerance; this measures accuracy.)
#include <iostream>
#include <fstream>
#include <string>
#include "../include/{ds}_model.hpp"

#ifndef NUM_TEST_SAMPLES
#define NUM_TEST_SAMPLES {n}
#endif

static float test_inputs_f[NUM_TEST_SAMPLES * INPUT_DIM];
static int   test_labels[NUM_TEST_SAMPLES];

static bool load_floats(const std::string& fn, float* d, int n) {{
    std::ifstream f(fn);
    if (!f.is_open()) {{ std::cerr << "ERROR: cannot open " << fn << std::endl; return false; }}
    for (int i = 0; i < n; i++) if (!(f >> d[i])) {{ std::cerr << "ERROR: short file " << fn << std::endl; return false; }}
    return true;
}}
static bool load_ints(const std::string& fn, int* d, int n) {{
    std::ifstream f(fn);
    if (!f.is_open()) {{ std::cerr << "ERROR: cannot open " << fn << std::endl; return false; }}
    for (int i = 0; i < n; i++) if (!(f >> d[i])) {{ std::cerr << "ERROR: short file " << fn << std::endl; return false; }}
    return true;
}}

static std::string find_data_dir() {{
    const std::string c[] = {{".","../../data","../../../data","../../../../data","../../../../../data","../../../../../../data","data","../data"}};
    for (const auto& dir : c) {{
        std::ifstream f(dir + "/test_labels.txt");
        if (f.is_open()) return dir;
    }}
    return "../../data";
}}

int main() {{
    std::string dd = find_data_dir();
    std::cout << "{Ds} accuracy TB  (data: " << dd << ", samples: " << NUM_TEST_SAMPLES
              << ", data_t=<" << DATA_W << "," << DATA_I << ">)" << std::endl;

    if (!load_floats(dd + "/test_input.txt", test_inputs_f, NUM_TEST_SAMPLES * INPUT_DIM)) return 1;
    if (!load_ints(dd + "/test_labels.txt", test_labels, NUM_TEST_SAMPLES)) return 1;

    int correct = 0;
    for (int s = 0; s < NUM_TEST_SAMPLES; s++) {{
        data_t in[INPUT_DIM], out[OUTPUT_DIM];
        for (unsigned int i = 0; i < INPUT_DIM; i++)
            in[i] = data_t(test_inputs_f[s * INPUT_DIM + i]);
        {ds}_top(in, out);
        int pred = 0;
        for (int o = 1; o < (int)OUTPUT_DIM; o++)
            if (out[o] > out[pred]) pred = o;
        if (pred == test_labels[s]) correct++;
    }}

    double acc = 100.0 * correct / NUM_TEST_SAMPLES;
    std::cout << "ACCURACY: " << correct << "/" << NUM_TEST_SAMPLES << " = " << acc << "%" << std::endl;
    std::cout << "RESULT: OK" << std::endl;
    return 0;
}}
"""


def gen_acc_tb_cpp(cfg, spec, profile, num_samples):
    """Accuracy testbench: argmax over a large sample set, vs test_labels.txt.

    Requires types_style='macro' — it prints DATA_W/DATA_I so a quantization
    sweep run is self-identifying in the csim log.
    """
    return ACC_TB_TMPL.format(ds=spec.name, Ds=spec.name.capitalize(),
                              n=num_samples, sc=spec.num_samples)


def gen_env_tcl(cfg, spec, profile, num_samples):
    """Env-parameterized TCL: one script drives a whole sweep axis.

    HLS_PART / HLS_PERIOD / HLS_TAG / HLS_SKIP_CSIM pick the target, QW / QI
    the fixed-point width via -DDATA_W/-DDATA_I. This is why one low-power
    project directory holds up to nine solutions (see core/collect.py
    --all-solutions).
    """
    ds = spec.name
    tb = f'{ds}_acc_tb.cpp' if profile.tb_mode == 'accuracy' else f'{ds}_tb.cpp'
    data = (profile.data_dir_mode.split(':', 1)[1]
            if profile.data_dir_mode.startswith('shared:') else '../data')
    proj = profile.proj_name_tmpl.format(
        name=cfg['name'], arch=arch_str(cfg['layer_sizes']),
        qw='${qw}', qi='${qi}', tag='${tag}')
    L = [f'# {cfg["name"]} ({profile.name}): env HLS_PART / HLS_PERIOD / HLS_TAG /'
         f' HLS_SKIP_CSIM / QW / QI']
    for var, env, dflt in (('part', 'HLS_PART', profile.part),
                           ('period', 'HLS_PERIOD', profile.clock_ns),
                           ('tag', 'HLS_TAG', profile.tcl_tag),
                           ('skip_csim', 'HLS_SKIP_CSIM', 0),
                           ('qw', 'QW', profile.data_w),
                           ('qi', 'QI', profile.data_i)):
        L.append(f'set {var:9s} [expr {{[info exists ::env({env})] ? '
                 f'$::env({env}) : "{dflt}"}}]')
    L += ['set cflags "-DDATA_W=$qw -DDATA_I=$qi"', '',
          f'open_project -reset {proj}', f'set_top {ds}_top',
          f'add_files ../src/{ds}_model.cpp -cflags $cflags',
          f'add_files -tb ../tb/{tb} -cflags $cflags',
          f'add_files -tb {data}/test_input.txt',
          f'add_files -tb {data}/test_labels.txt',
          'open_solution -reset solution1', 'set_part $part',
          'create_clock -period $period -name default']
    if 'csim' in profile.tcl_stages:
        L += ['if {!$skip_csim} {', '    csim_design', '}']
    if 'csynth' in profile.tcl_stages:
        L.append('csynth_design')
    L += ['close_project', f'puts "{cfg["name"]} ({profile.name}) q${{qw}}_${{qi}}'
          f' ($tag) done."', 'exit']
    return '\n'.join(L) + '\n'


def gen_tcl(cfg, spec, mode, num_samples, profile=None):
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
add_files ../src/{ds}_model.cpp
add_files -tb ../tb/{ds}_tb.cpp -cflags "-DNUM_TEST_SAMPLES={num_samples}"
open_solution "solution1"
set_part {{{part}}}
create_clock -period {clock_ns} -name default
{cmd}
exit
"""


# =====================================================================
# ROM header writers
# =====================================================================

def write_weight_rom(path, weight_list):
    """Write weight ROM header with 2D arrays.

    weight_list: [(name, W_array), ...]; W_array shape (out_dim, in_dim).
    """
    with open(path, 'w') as f:
        f.write('#ifndef WEIGHT_ROM_HPP\n#define WEIGHT_ROM_HPP\n\n')
        f.write('#include "types.hpp"\n#include "config.hpp"\n\n')
        for rom_name, W in weight_list:
            out_dim, in_dim = W.shape
            f.write(f'static const data_t {rom_name}[{out_dim}][{in_dim}] = {{\n')
            for o in range(out_dim):
                f.write(f'    /* o{o:3d} */ {{')
                for i in range(in_dim):
                    if i > 0 and i % 8 == 0:
                        f.write('\n              ')
                    c = ', ' if i < in_dim - 1 else ''
                    f.write(f'{W[o, i]:.10f}{c}')
                c = ',' if o < out_dim - 1 else ''
                f.write(f'}}{c}\n')
            f.write('};\n\n')
        f.write('#endif // WEIGHT_ROM_HPP\n')


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
    unroll_strategy = profile.unroll_strategy or spec.unroll_strategy
    ds = spec.name
    name = cfg['name']
    layer_sizes = cfg['layer_sizes']
    num_linear = len(layer_sizes) - 1
    hidden_dims = layer_sizes[1:-1]
    is_bern = cfg['act'] == 'bern'
    num_samples = len(X_test)

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
    linears_data = []   # (W, b) — may be fused
    luts = []           # per-hidden-layer LUT arrays (or None)
    weight_rom_list = []  # (rom_name, W_array)
    bias_rom_list = []    # (rom_name, bias_array)

    for li in range(num_linear):
        W, b = linears[li]
        if li < num_linear - 1 and is_bern:
            bl = bern_layers[li]
            bounds = bl.input_bounds.data.numpy()
            W_fused, b_fused, _, _ = compute_fused(W, b, bounds)
            linears_data.append((W_fused, b_fused))
            weight_rom_list.append((f'LINEAR{li+1}_WEIGHT_ROM', W_fused))
            bias_rom_list.append((f'LINEAR{li+1}_BIAS_ROM', b_fused))
            lut = build_full_lut(bl, NEURON_LUT_SIZE, spec.lut_grid)
            luts.append(lut)
        else:
            linears_data.append((W, b))
            weight_rom_list.append((f'LINEAR{li+1}_WEIGHT_ROM', W))
            bias_rom_list.append((f'LINEAR{li+1}_BIAS_ROM', b))
            luts.append(None)

    # Generate HLS source files
    print(f"  [2] Generating HLS source files...")
    write_file(os.path.join(inc, 'config.hpp'), gen_config_hpp(cfg, spec, profile))
    write_file(os.path.join(inc, 'types.hpp'), gen_types_hpp(profile))
    write_file(os.path.join(inc, f'{ds}_model.hpp'), gen_model_hpp(cfg, spec))
    write_file(os.path.join(src, f'{ds}_model.cpp'), gen_model_cpp(cfg, spec, unroll_strategy, profile))
    if profile.tb_mode == 'accuracy':
        write_file(os.path.join(tb, f'{ds}_acc_tb.cpp'),
                   gen_acc_tb_cpp(cfg, spec, profile, num_samples))
    else:
        write_file(os.path.join(tb, f'{ds}_tb.cpp'), gen_tb_cpp(cfg, spec, profile))
    if profile.tcl_style == 'env':
        # One script drives the whole sweep axis, so there is a single TCL
        # rather than one per stage.
        write_file(os.path.join(scr, 'run_csynth.tcl'),
                   gen_env_tcl(cfg, spec, profile, num_samples))
    else:
        for stage in profile.tcl_stages:
            write_file(os.path.join(scr, f'run_{stage}.tcl'),
                       gen_tcl(cfg, spec, stage, num_samples, profile))

    # Generate ROM headers
    print(f"  [3] Generating ROM headers...")
    write_weight_rom(os.path.join(inc, 'weight_rom.hpp'), weight_rom_list)
    write_bias_rom(os.path.join(inc, 'bias_rom.hpp'), bias_rom_list)

    if is_bern and profile.activation_impl == 'shared_basis':
        if len(hidden_dims) != 1:
            raise ValueError('shared-basis activation supports a single hidden layer; '
                             f"{cfg['name']} has {len(hidden_dims)}")
        bl = bern_layers[0]
        basis, coeffs = sb.factor_shared_basis(bl, NEURON_LUT_SIZE, profile.basis_grid)
        sb.check_reconstruction(basis, coeffs, luts[0])
        write_file(os.path.join(inc, 'shared_basis_rom.hpp'),
                   sb.gen_shared_basis_rom(basis, coeffs, cfg['degree'], NEURON_LUT_SIZE))
    elif is_bern:
        if len(hidden_dims) == 1:
            lut_entries = [('NEURON_ACT_LUT', 'HIDDEN_DIM', luts[0])]
        else:
            lut_entries = []
            for i in range(len(hidden_dims)):
                lut_entries.append((f'NEURON_ACT_LUT_L{i+1}', f'HIDDEN{i+1}_DIM', luts[i]))
        write_activation_lut_rom(os.path.join(inc, 'activation_lut_rom.hpp'), lut_entries)

    # Generate test data files
    print(f"  [4] Computing reference outputs...")
    ref = ref_inference(X_test, linears_data, luts, is_bern, NEURON_LUT_SIZE)

    if profile.emit_data:
        write_flat(os.path.join(data_dir, 'test_input.txt'), X_test)
        write_flat(os.path.join(data_dir, 'test_output_ref.txt'), ref)
        with open(os.path.join(data_dir, 'test_labels.txt'), 'w') as f:
            for label in y_test:
                f.write(f'{label}\n')

    pred = ref.argmax(axis=1)
    print(f"  Done: {arch_str(layer_sizes)} {cfg['act']}, "
          f"{len(X_test)} test samples, preds: {pred[:5].tolist()}")
