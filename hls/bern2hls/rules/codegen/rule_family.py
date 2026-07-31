"""Emitter for the rule-classifier HLS projects.

Replaces the forked per-study generators under rule_distillation/*/hls/. All
three shipped suites produce the same project shape — the differences are the
Phase-1 storage form (dense vs sparse), the Phase-3 fallback body, the ROM
binding pragmas and per-suite comment prose (see prose.py).
"""

import os

from ...core.emit import write_file
from .. import fallback as fb
from ..rules_io import fp_to_int8
from . import prose


def _fmt(x):
    """Emit floats the way the original generators did: repr(float(x))."""
    return repr(float(x))


def _arr1(name, ctype, vals):
    return f'static const {ctype} {name}[{len(vals)}] = {{ ' + ', '.join(vals) + ' };'


def _arr2(name, ctype, rows):
    L = [f'static const {ctype} {name}[{len(rows)}][{len(rows[0])}] = {{']
    for i, row in enumerate(rows):
        L.append('  { ' + ', '.join(row) + ' }' + (',' if i < len(rows) - 1 else ''))
    L.append('};')
    return '\n'.join(L)


# ------------------------------------------------------------- rule ROM

def _emit_rom_f4v(model, tables, spec):
    """ROM layout used by fallback_4_variance and tree_arch: 2-space indent,
    lowercase rule labels, single-line metadata arrays."""
    nr, mc = model.n_rules, model.max_conds
    st = prose.ROM_STYLE[spec.prose]
    L = [st['header'],
         'static const w8_t COND_W8[N_RULES][MAX_CONDS][N_FEATURES] = {']
    for r in range(nr):
        L.append(f'  {{ // rule {r}')
        for c in range(mc):
            vals = ', '.join(str(v) for v in tables['cond_w8'][r][c])
            L.append(f'    {{ {vals} }}{"," if c < mc - 1 else ""}')
        L.append(f'  }}{"," if r < nr - 1 else ""}')
    L.append('};\n')

    L.append('static const acc_t COND_SCALE[N_RULES][MAX_CONDS] = {')
    for r in range(nr):
        vals = ', '.join(f'acc_t({_fmt(tables["cond_scale"][r][c])})' for c in range(mc))
        L.append(f'  {{ {vals} }}{"," if r < nr - 1 else ""}')
    L.append('};\n')

    for name, key in (('COND_BAND_LO', 'band_lo'), ('COND_BAND_HI', 'band_hi')):
        L.append(f'static const data_t {name}[N_RULES][MAX_CONDS] = {{')
        for r in range(nr):
            vals = ', '.join(
                f'data_t({_fmt(tables[key][r][c] if tables[key][r][c] is not None else 0.0)})'
                for c in range(mc))
            L.append(f'  {{ {vals} }}{"," if r < nr - 1 else ""}')
        L.append('};\n')

    for name, key in (('COND_HAS_LO', 'has_lo'), ('COND_HAS_HI', 'has_hi')):
        L.append(f'static const bool {name}[N_RULES][MAX_CONDS] = {{')
        for r in range(nr):
            vals = ', '.join('true' if tables[key][r][c] else 'false' for c in range(mc))
            L.append(f'  {{ {vals} }}{"," if r < nr - 1 else ""}')
        L.append('};\n')

    L.append('static const int RULE_LABEL[N_RULES] = { '
             + ', '.join(str(v) for v in tables['labels']) + ' };')
    L.append('static const purity_t RULE_PURITY[N_RULES] = { '
             + ', '.join(str(v) for v in tables['purity']) + ' };')
    L.append('static const unsigned int RULE_N_COND[N_RULES] = { '
             + ', '.join(str(v) for v in tables['n_cond']) + ' };\n')
    return '\n'.join(L)


def _emit_rom_ksweep(model, tables, spec):
    """ROM layout used by sparsity_sweep: 4-space indent, a section comment per
    array, sparse COND_IDX, and multi-line metadata arrays. Also emits an absent
    band bound as `data_t(0)` while a padded slot stays `data_t(0.0)`."""
    nr, mc = model.n_rules, model.max_conds
    L = ['// Sparse int8 quantized condition weights [rule][cond][K_SPARSE]',
         'static const w8_t COND_W8[N_RULES][MAX_CONDS][K_SPARSE] = {']

    def cube(key):
        out = []
        for r in range(nr):
            out.append(f'    {{ // Rule {r}')
            for c in range(mc):
                vals = ', '.join(str(v) for v in tables[key][r][c])
                out.append(f'        {{ {vals} }}{"," if c < mc - 1 else ""}')
            out.append(f'    }}{"," if r < nr - 1 else ""}')
        return out

    L += cube('cond_w8')
    L += ['};', '']
    L.append('// Feature indices for sparse weights (0-13)')
    L.append('static const idx_t COND_IDX[N_RULES][MAX_CONDS][K_SPARSE] = {')
    L += cube('cond_idx')
    L += ['};', '']

    L.append('// Per-condition int8 scale factors')
    L.append('static const acc_t COND_SCALE[N_RULES][MAX_CONDS] = {')
    for r in range(nr):
        vals = ', '.join(f'acc_t({_fmt(tables["cond_scale"][r][c])})' for c in range(mc))
        L.append(f'    {{ {vals} }}{"," if r < nr - 1 else ""}')
    L += ['};', '']

    for name, key, cmt in (('COND_BAND_LO', 'band_lo', 'lower'),
                           ('COND_BAND_HI', 'band_hi', 'upper')):
        L.append(f'// Condition band {cmt} bounds')
        L.append(f'static const data_t {name}[N_RULES][MAX_CONDS] = {{')
        for r in range(nr):
            vals = ', '.join(
                f'data_t({tables[key][r][c]})' if tables[key][r][c] is not None
                else 'data_t(0)' for c in range(mc))
            L.append(f'    {{ {vals} }}{"," if r < nr - 1 else ""}')
        L += ['};', '']

    for name, key in (('COND_HAS_LO', 'has_lo'), ('COND_HAS_HI', 'has_hi')):
        L.append(f'static const bool {name}[N_RULES][MAX_CONDS] = {{')
        for r in range(nr):
            vals = ', '.join('true' if tables[key][r][c] else 'false' for c in range(mc))
            L.append(f'    {{ {vals} }}{"," if r < nr - 1 else ""}')
        L += ['};', '']

    for name, ctype, key in (('RULE_LABEL', 'int', 'labels'),
                             ('RULE_PURITY', 'purity_t', 'purity'),
                             ('RULE_N_COND', 'unsigned int', 'n_cond')):
        L.append(f'static const {ctype} {name}[N_RULES] = {{')
        L.append('    ' + ', '.join(str(v) for v in tables[key]))
        L += ['};', '']
    return '\n'.join(L)


def emit_rule_rom(model, tables, spec):
    if spec.prose == 'ksweep':
        return _emit_rom_ksweep(model, tables, spec)
    return _emit_rom_f4v(model, tables, spec)


# --------------------------------------------------------- fallback ROM

def emit_fallback_rom(kind, params, style='f4v'):
    if kind == 'none':
        return ''

    if kind == 'lr':
        w8 = [str(v) for v in fp_to_int8(params['w'], params['scale'])]
        scale = f'static const acc_t LR_SCALE = acc_t({_fmt(params["scale"])});'
        bias = f'static const data_t LR_B_EFF = data_t({_fmt(params["b"])});'
        if style == 'ksweep':
            return '\n'.join([
                '// LR fallback: int8 weights, scale, bias',
                'static const w8_t LR_W8[N_FEATURES] = {',
                '    ' + ', '.join(w8), '};', '', scale, bias, ''])
        return '\n'.join([
            '// ---- LR fallback (int8) ----',
            _arr1('LR_W8', 'w8_t', w8), scale, bias, ''])

    if kind in ('network', 'small_nn'):
        P, ct = params, 'nn_t'
        H = P['W0'].shape[0]
        lo = P['bounds'][:, 0]
        inv = 1.0 / (P['bounds'][:, 1] - P['bounds'][:, 0] + 1e-8)
        L = [f'// ---- {kind} Bernstein-NN fallback (H={H}, deg 3) ----']
        L.append(_arr1('NN_B0', ct, [f'{ct}({_fmt(v)})' for v in P['b0']]))
        L.append(_arr1('NN_LO', ct, [f'{ct}({_fmt(v)})' for v in lo]))
        L.append(_arr1('NN_INVR', ct, [f'{ct}({_fmt(v)})' for v in inv]))
        L.append(_arr2('NN_COEFF', ct,
                       [[f'{ct}({_fmt(v)})' for v in P['coeffs'][j]] for j in range(H)]))
        L.append(_arr1('NN_B2', ct, [f'{ct}({_fmt(v)})' for v in P['b2']]))
        if kind == 'network':
            L.append(_arr2('NN_W0', ct,
                           [[f'{ct}({_fmt(v)})' for v in P['W0'][j]] for j in range(H)]))
            L.append(_arr2('NN_W2', ct,
                           [[f'{ct}({_fmt(v)})' for v in P['W2'][o]] for o in range(2)]))
        else:
            # int8 weights with one scale per tensor
            import numpy as np
            s0 = float(np.abs(P['W0']).max()) / 127.0
            s2 = float(np.abs(P['W2']).max()) / 127.0
            L.append(_arr2('NN_W0Q', 'w8_t',
                           [[str(v) for v in fp_to_int8(P['W0'][j], s0)] for j in range(H)]))
            L.append(_arr2('NN_W2Q', 'w8_t',
                           [[str(v) for v in fp_to_int8(P['W2'][o], s2)] for o in range(2)]))
            L.append(f'static const acc_t NN_S0 = acc_t({_fmt(s0)});')
            L.append(f'static const acc_t NN_S2 = acc_t({_fmt(s2)});')
        L.append('')
        return '\n'.join(L)

    if kind == 'tree':
        T = params
        leaf = T['value'].argmax(1)
        L = ['// ---- CART fallback (compares only, fix<16,8> thresholds) ----']
        L.append(_arr1('TREE_FEAT', 'int', [str(int(v)) for v in T['feat']]))
        L.append(_arr1('TREE_THR', 'data_t', [f'data_t({_fmt(v)})' for v in T['thr']]))
        L.append(_arr1('TREE_LEFT', 'int', [str(int(v)) for v in T['left']]))
        L.append(_arr1('TREE_RIGHT', 'int', [str(int(v)) for v in T['right']]))
        L.append(_arr1('TREE_LEAF', 'int', [str(int(v)) for v in leaf]))
        L.append('')
        return '\n'.join(L)

    raise ValueError(kind)


# ------------------------------------------------------------- source files

def gen_config_hpp(model, spec):
    extra = ''
    if spec.nn_hidden is not None:
        extra = f'constexpr unsigned int NN_HID = {spec.nn_hidden};\n'
    elif spec.tree_max_depth is not None:
        extra = f'constexpr unsigned int TREE_MAX_DEPTH = {spec.tree_max_depth};\n'
    if spec.scope == 'fb_only':
        return prose.CONFIG_HPP_FB_ONLY.format(n_features=model.n_features, extra=extra)
    return prose.CONFIG_HPP[spec.prose].format(
        n_features=model.n_features, n_rules=model.n_rules,
        max_conds=model.max_conds, k_sparse=spec.k_sparse, extra=extra)


def gen_types_hpp(spec, profile=None):
    text = prose.TYPES_HPP[spec.prose]
    # A fallback-only top has no Phase-1 dot product, so no exact-product type.
    if profile is not None and profile.prod_type and spec.scope != 'fb_only':
        line = (f'typedef ap_fixed<{profile.prod_t_decl()}> {profile.prod_type};'
                f'{profile.prod_comment}\n')
        text = text.replace('typedef unsigned int', line + 'typedef unsigned int')
    return text


def gen_hdr_hpp(spec):
    key = 'f4v' if spec.prose == 'tree_arch' else spec.prose
    return prose.HDR_HPP[key].format(top_fn=spec.top_fn)


def gen_rom_hpp(model, tables, spec, fb_params):
    fb_rom = emit_fallback_rom(spec.fallback_kind, fb_params, spec.prose)
    if spec.scope == 'fb_only':
        # No rule tables — the point is the fallback's area in isolation.
        return ('#ifndef FB_ROM_HPP\n#define FB_ROM_HPP\n\n'
                '#include "types.hpp"\n#include "config.hpp"\n\n'
                + fb_rom + '\n#endif\n')
    return ('#ifndef RULE_ROM_HPP\n#define RULE_ROM_HPP\n\n'
            '#include "types.hpp"\n#include "config.hpp"\n\n'
            + emit_rule_rom(model, tables, spec) + '\n' + fb_rom
            + '\n#endif\n')


def _dot_line(profile):
    """The Phase-1 multiply-accumulate, plain or exact-narrow."""
    if profile is None or not profile.narrow_mult:
        return '                z += acc_t(COND_W8[r][c][i]) * acc_t(x_local[i]);'
    return (f'                {profile.prod_type} p = COND_W8[r][c][i] * x_local[i];'
            f'{profile.narrow_mult_comment}\n                z += p;')


def _rom_pragmas(model, profile):
    """BIND_STORAGE/ARRAY_RESHAPE block, when a profile relocates the rule ROM."""
    if profile is None or not profile.rom_pragma:
        return ''
    nr, mc, nf = model.n_rules, model.max_conds, model.n_features
    comment = profile.rom_pragma_comment.format(
        rom_dims=f'{nr}x{mc}x{nf}', rom_kb=f'{nr * mc * nf / 1000:.1f}')
    verb = {'reshape_dim3': 'ARRAY_RESHAPE variable=COND_W8 complete dim=3',
            'partition_dim3': 'ARRAY_PARTITION variable=COND_W8 dim=3 type=complete'}
    return (comment + '\n'
            f'    #pragma HLS {verb[profile.rom_pragma]}\n'
            f'    #pragma HLS BIND_STORAGE  variable=COND_W8 type=rom_1p '
            f'impl={profile.rom_impl}\n\n')


def gen_src_cpp(model, spec, profile=None):
    phase3 = prose.PHASE3[spec.fallback_kind] if spec.fallback_kind != 'none' else None

    if spec.scope == 'fb_only':
        # Just the Phase-3 block, so its area can be measured in isolation.
        body = [f'#include "../include/rule_classifier.hpp"',
                '#include "../include/rule_rom.hpp"', '',
                prose.FB_ONLY_HEADER.format(variant=spec.fallback_kind),
                f'void {spec.top_fn}(int *result, const data_t x[N_FEATURES]) {{',
                '    #pragma HLS INTERFACE s_axilite port=result bundle=control',
                '    #pragma HLS INTERFACE s_axilite port=x      bundle=control',
                '    #pragma HLS INTERFACE s_axilite port=return bundle=control', '',
                '    data_t x_local[N_FEATURES];',
                '    #pragma HLS ARRAY_PARTITION variable=x_local complete',
                '    for (unsigned int i = 0; i < N_FEATURES; i++) {',
                '        #pragma HLS UNROLL', '        x_local[i] = x[i];', '    }', '',
                '    int best_label = -1;', '    {', phase3, '    }',
                '    *result = best_label;', '}']
        return '\n'.join(body) + '\n'

    text = prose.SRC_TMPL[spec.prose].format(
        variant=spec.fallback_kind, top_fn=spec.top_fn, phase3=phase3,
        tag=spec.arch or spec.name, banked=model.n_rules * model.max_conds)
    text = text.replace(
        '                z += acc_t(COND_W8[r][c][i]) * acc_t(x_local[i]);',
        _dot_line(profile))
    pragmas = _rom_pragmas(model, profile)
    if pragmas:
        text = text.replace('    data_t x_local[N_FEATURES];',
                            pragmas + '    data_t x_local[N_FEATURES];', 1)
    return text


TB_CPP = """\
#include <iostream>
#include <fstream>
#include <iomanip>
#include <string>
#include "../include/rule_classifier.hpp"

#define NUM_TEST {num_test}

static bool load_floats(const std::string& fn, float* data, int count) {{
    std::ifstream f(fn);
    if (!f.is_open()) {{ std::cerr << "Cannot open " << fn << std::endl; return false; }}
    for (int i = 0; i < count; i++)
        if (!(f >> data[i])) {{ std::cerr << "Short read " << fn << std::endl; return false; }}
    return true;
}}
static bool load_ints(const std::string& fn, int* data, int count) {{
    std::ifstream f(fn);
    if (!f.is_open()) {{ std::cerr << "Cannot open " << fn << std::endl; return false; }}
    for (int i = 0; i < count; i++)
        if (!(f >> data[i])) {{ std::cerr << "Short read " << fn << std::endl; return false; }}
    return true;
}}
static std::string find_data_dir() {{
    const std::string c[] = {{"../data","../../data","../../../data",
                              "../../../../data","../../../../../data","data"}};
    for (const auto& dir : c) {{ std::ifstream f(dir + "/test_input.txt");
        if (f.is_open()) return dir; }}
    return "../data";
}}

static float test_inputs_f[NUM_TEST * N_FEATURES];
static int   test_labels[NUM_TEST];
static int   test_ref_preds[NUM_TEST];

int main() {{
    std::string dd = find_data_dir();
    std::cout << "N_FEATURES=" << N_FEATURES << " N_RULES=" << N_RULES
              << " MAX_CONDS=" << MAX_CONDS << " samples=" << NUM_TEST << std::endl;
    if (!load_floats(dd + "/test_input.txt", test_inputs_f, NUM_TEST*N_FEATURES)) return 1;
    if (!load_ints(dd + "/test_labels.txt", test_labels, NUM_TEST)) return 1;
    if (!load_ints(dd + "/test_output_ref.txt", test_ref_preds, NUM_TEST)) return 1;

    int hls_correct=0, ref_correct=0, agree=0, mismatch=0;
    for (int s = 0; s < NUM_TEST; s++) {{
        data_t x[N_FEATURES];
        for (unsigned int j = 0; j < N_FEATURES; j++)
            x[j] = data_t(test_inputs_f[s*N_FEATURES + j]);
        int pred; {top_fn}(&pred, x);
        if (pred == test_labels[s])    hls_correct++;
        if (test_ref_preds[s] == test_labels[s]) ref_correct++;
        if (pred == test_ref_preds[s]) agree++; else mismatch++;
    }}
    std::cout << std::fixed << std::setprecision(2);
    std::cout << "HLS  acc : " << hls_correct << "/" << NUM_TEST
              << " (" << (float)hls_correct/NUM_TEST*100 << "%)" << std::endl;
    std::cout << "Ref  acc : " << ref_correct << "/" << NUM_TEST
              << " (" << (float)ref_correct/NUM_TEST*100 << "%)" << std::endl;
    std::cout << "HLS==Ref : " << agree << "/" << NUM_TEST
              << " (" << (float)agree/NUM_TEST*100 << "%)  mismatch=" << mismatch << std::endl;
    if (mismatch == 0) std::cout << "RESULT: ALL MATCH" << std::endl;
    else std::cout << "RESULT: " << mismatch << " mismatches (fixed-point)" << std::endl;
    return (mismatch > NUM_TEST * 10 / 100) ? 1 : 0;
}}
"""


def gen_tb_cpp(spec, num_test, profile=None):
    if spec.prose == 'ksweep':
        return prose.TB_KSWEEP.format(num_test=num_test)
    tb = TB_CPP.format(num_test=num_test, top_fn=spec.top_fn)
    if profile is not None and profile.tb_guard_num_test:
        tb = tb.replace(f'#define NUM_TEST {num_test}',
                        f'#ifndef NUM_TEST\n#define NUM_TEST {num_test}\n#endif')
    if spec.prose == 'tree_arch':
        tb = tb.replace(prose.TB_F4V_BANNER, prose.TB_TREE_ARCH_BANNER)
    return tb


def gen_env_tcl(spec, profile):
    """Env-parameterized TCL for the low-power sweeps: one script per design,
    target chosen at run time via HLS_PART / HLS_PERIOD / HLS_TAG."""
    rows = (('part', 'HLS_PART', profile.part),
            ('period', 'HLS_PERIOD', profile.clock_ns),
            ('tag', 'HLS_TAG', profile.tcl_tag))
    w = max(len(e) for _, e, _ in rows)          # the three lines are column-aligned
    L = ['# Parameterized for lowpower_fpga study: PART / PERIOD / PROJ_TAG /'
         ' SKIP_CSIM via env']
    for var, env, dflt in rows:
        pad = ' ' * (w - len(env) + 1)
        L.append(f'set {var:6s} [expr {{[info exists ::env({env})]{pad}? '
                 f'$::env({env}){pad}: "{dflt}"}}]')
    L.append('set skip_csim [expr {[info exists ::env(HLS_SKIP_CSIM)] ? '
             '$::env(HLS_SKIP_CSIM) : "0"}]')
    sep = [] if spec.prose == 'tree_arch' else ['']
    L += ['', f'open_project -reset {spec.hls_proj}', f'set_top {spec.top_fn}'] + sep + [
          'add_files ../src/rule_classifier.cpp',
          'add_files ../include/config.hpp',
          'add_files ../include/types.hpp',
          'add_files ../include/rule_rom.hpp',
          'add_files ../include/rule_classifier.hpp'] + sep + [
          'add_files -tb ../tb/rule_classifier_tb.cpp',
          'add_files -tb ../data/test_input.txt',
          'add_files -tb ../data/test_labels.txt',
          'add_files -tb ../data/test_output_ref.txt'] + sep + [
          'open_solution -reset solution1', 'set_part $part',
          'create_clock -period $period -name default',
          'config_compile -pragma_strict_mode=true'] + sep + [
          'if {!$skip_csim} {', '    csim_design', '}', 'csynth_design'] + sep + [
          'close_project',
          f'puts "{spec.tcl_label} ($tag, $part, ${{period}}ns) csim+csynth completed."',
          'exit']
    return '\n'.join(L) + '\n'


def gen_tcl(spec, profile):
    if profile is not None and profile.tcl_style == 'env':
        return gen_env_tcl(spec, profile)
    """csim+csynth for a full classifier; csynth only for a fallback-only top
    (it has no testbench — its whole point is isolated area)."""
    sep = [] if (spec.scope == 'fb_only' or spec.prose == 'tree_arch') else ['']
    L = [f'open_project -reset {spec.hls_proj}', f'set_top {spec.top_fn}'] + sep + [
         'add_files ../src/rule_classifier.cpp',
         'add_files ../include/config.hpp',
         'add_files ../include/types.hpp',
         'add_files ../include/rule_rom.hpp',
         'add_files ../include/rule_classifier.hpp'] + sep
    if spec.scope != 'fb_only':
        L += ['add_files -tb ../tb/rule_classifier_tb.cpp',
              'add_files -tb ../data/test_input.txt',
              'add_files -tb ../data/test_labels.txt',
              'add_files -tb ../data/test_output_ref.txt'] + sep
    L += ['open_solution -reset solution1',
          f'set_part {{{profile.part}}}',
          f'create_clock -period {profile.clock_ns} -name default',
          'config_compile -pragma_strict_mode=true'] + sep
    if spec.scope != 'fb_only':
        L += ['csim_design', 'csynth_design']
        if spec.prose == 'ksweep':
            done = f'{spec.name} csim+csynth completed.'
        elif spec.prose == 'tree_arch':
            done = f'{spec.name} tree csim+csynth completed.'
        else:
            done = f'{spec.fallback_kind} csim+csynth completed.'
    else:
        L += ['csynth_design']
        done = f'{spec.fallback_kind} fallback-only csynth completed.'
    L += sep + ['close_project', f'puts "{done}"', 'exit']
    return '\n'.join(L) + '\n'
