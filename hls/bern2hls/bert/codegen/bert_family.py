"""Stamp a transformer HLS project from the kernel library plus a VariantSpec.

Only the parts the study varies are generated: config.hpp, the ROM headers, the
weight streams and the TCL. The kernels — linear, attention, softmax,
LayerNorm, the activations — are copied from bern2hls/bert/kernels/, because
they are hand-tuned and generating them would add risk without serving the
claim, which is about the activation.
"""

import os

from . import prose

KERNELS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       'kernels')

# Which kernel files a variant needs, and what each is called in the project.
# (source-in-library, destination-in-project)
_FFN_COMMON = [
    ('include/linear.hpp', 'include/linear.hpp'),
    ('include/types_ffn.hpp', 'include/types.hpp'),
    ('src/linear.cpp', 'src/linear.cpp'),
    ('tb/ffn_tb.cpp', 'tb/ffn_tb.cpp'),
]
_LAYER_COMMON = [
    ('include/linear.hpp', 'include/linear.hpp'),
    ('include/attention.hpp', 'include/attention.hpp'),
    ('include/softmax.hpp', 'include/softmax.hpp'),
    ('include/layernorm.hpp', 'include/layernorm.hpp'),
    ('include/types_layer.hpp', 'include/types.hpp'),
    ('src/linear.cpp', 'src/linear.cpp'),
    ('src/attention.cpp', 'src/attention.cpp'),
    ('src/softmax.cpp', 'src/softmax.cpp'),
    ('src/layernorm.cpp', 'src/layernorm.cpp'),
    ('tb/layer_tb.cpp', 'tb/layer_tb.cpp'),
]
_ACT_FILES = {
    ('ffn', 'bern'): [('include/bernstein.hpp', 'include/bernstein.hpp'),
                      ('src/bernstein.cpp', 'src/bernstein.cpp'),
                      ('include/ffn_model_bern.hpp', 'include/ffn_model.hpp'),
                      ('src/bern_ffn.cpp', 'src/bern_ffn.cpp')],
    ('ffn', 'gelu_lut'): [('include/gelu_act_lut.hpp', 'include/gelu_act.hpp'),
                          ('src/gelu_act_lut.cpp', 'src/gelu_act.cpp'),
                          ('include/ffn_model_gelu_lut.hpp', 'include/ffn_model.hpp'),
                          ('src/gelu_ffn_lut.cpp', 'src/gelu_ffn.cpp')],
    ('ffn', 'gelu_poly'): [('include/gelu_act_poly.hpp', 'include/gelu_act.hpp'),
                           ('src/gelu_act_poly.cpp', 'src/gelu_act.cpp'),
                           ('include/ffn_model_gelu_poly.hpp', 'include/ffn_model.hpp'),
                           ('src/gelu_ffn_poly.cpp', 'src/gelu_ffn.cpp')],
    ('layer', 'bern'): [('include/bernstein.hpp', 'include/bernstein.hpp'),
                        ('src/bernstein.cpp', 'src/bernstein.cpp'),
                        ('include/layer_model_bern.hpp', 'include/layer_model.hpp'),
                        ('src/bern_layer.cpp', 'src/bern_layer.cpp')],
    ('layer', 'gelu_lut'): [('include/gelu_act_lut.hpp', 'include/gelu_act.hpp'),
                            ('src/gelu_act_lut.cpp', 'src/gelu_act.cpp'),
                            ('include/layer_model_gelu.hpp', 'include/layer_model.hpp'),
                            ('src/gelu_layer.cpp', 'src/gelu_layer.cpp')],
    ('layer', 'gelu_poly'): [('include/gelu_act_poly.hpp', 'include/gelu_act.hpp'),
                             ('src/gelu_act_poly.cpp', 'src/gelu_act.cpp'),
                             ('include/layer_model_gelu.hpp', 'include/layer_model.hpp'),
                             ('src/gelu_layer.cpp', 'src/gelu_layer.cpp')],
}

# The residual widening, expressed as the edits that produced it. The library
# holds the h312 form (widened, with the explanatory comments); these run it
# backwards. `no_comment` drops the commentary the h600 pair lacks; `legacy`
# then undoes the widening itself for the four pre-fix designs. Both are keyed
# per file because the wording differs per file — layernorm.hpp in particular
# carries three distinct comment blocks across the three states.
# layernorm.hpp's three states are not a chain — each carries its own rationale
# for the input width — so the whole comment+declaration block is swapped.
_LN_SIG_WIDE = ('void layernorm(data_t out[D_MODEL], const wide_t x[D_MODEL],\n')
_LN_SIG_NARROW = ('void layernorm(data_t out[D_MODEL], const data_t x[D_MODEL],\n')
_LN_HPP_H312 = (
    "// out[i] = gamma[i]*(x[i]-mean)/sqrt(var+eps) + beta[i]\n"
    "// Input x is wide_t: the LN2 residual (x_ffn + fc2_out) reaches ~+-430 in h312, exceeding\n"
    "// data_t's +-128. wide_t shares data_t's 24 fractional bits (lossless for the data_t residual)\n"
    "// and adds integer headroom; stats are computed in float regardless.\n" + _LN_SIG_WIDE)
_LN_HPP_H600 = (
    "// x is wide_t: the post-residual sum (y + x_ffn) reaches ~\u00b1200 for the h600 GeLU\n"
    "// FFN, exceeding data_t's \u00b1128 \u2014 so the residual/LN input is widened.\n"
    + _LN_SIG_WIDE)
_LN_HPP_LEGACY = ("// out[i] = gamma[i]*(x[i]-mean)/sqrt(var+eps) + beta[i]\n"
                  + _LN_SIG_NARROW)

_ATTN_COMMENT = (
    "        // z1 in wide_t to match layernorm's widened input (LN1 residual is small, ~+-22,\n"
    "        // but the type must agree; wide_t is lossless from data_t).\n")
_BERN_LAYER_COMMENT = (
    "        // LN2 residual: y (fc2 output) reaches ~+-430 in the h312 model, which exceeds\n"
    "        // data_t's +-128 -> keep z2 in wide_t so the LayerNorm sees the true value (a data_t\n"
    "        // z2 would saturate and corrupt the variance for high-output tokens).\n")
_GELU_LAYER_COMMENT = (
    "        // LN2 residual in wide_t: y (fc2 output) exceeds data_t's +-128 in the h312 model;\n"
    "        // a data_t z2 would saturate and corrupt the LayerNorm variance for high-output tokens.\n")

_Z1 = [('        wide_t z1[D_MODEL];', '        data_t z1[D_MODEL];'),
       ('            z1[j] = wide_t(x[t][j]) + wide_t(AO[t][j]);',
        '            z1[j] = data_t(x[t][j] + AO[t][j]);')]
_Z2 = [('        wide_t z2[D_MODEL];', '        data_t z2[D_MODEL];'),
       ('            z2[j] = wide_t(xffn_buf[t][j]) + y[j];',
        '            z2[j] = data_t(wide_t(xffn_buf[t][j]) + y[j]);')]
_LN_SIG = [('void layernorm(data_t out[D_MODEL], const wide_t x[D_MODEL],',
            'void layernorm(data_t out[D_MODEL], const data_t x[D_MODEL],')]

TRANSFORM = {
    'include/layernorm.hpp': {
        'no_comment': [(_LN_HPP_H312, _LN_HPP_H600)],
        'legacy': [(_LN_HPP_H600, _LN_HPP_LEGACY)],
    },
    'src/layernorm.cpp': {'no_comment': [], 'legacy': _LN_SIG},
    'src/attention.cpp': {'no_comment': [(_ATTN_COMMENT, '')], 'legacy': _Z1},
    'src/bern_layer.cpp': {'no_comment': [(_BERN_LAYER_COMMENT, '')], 'legacy': _Z2},
    'src/gelu_layer.cpp': {'no_comment': [(_GELU_LAYER_COMMENT, '')], 'legacy': _Z2},
}


def _apply_residual(rel, text, spec):
    """Walk the canonical h312 kernel back to the state a variant shipped in."""
    rules = TRANSFORM.get(rel)
    if not rules:
        return text
    if not spec.residual_comment:
        for old, new in rules['no_comment']:
            text = text.replace(old, new)
    if not spec.residual_fix:
        for old, new in rules['legacy']:
            text = text.replace(old, new)
    return text


def kernel_files(spec):
    """{project-relative path: contents} for every copied kernel file."""
    pairs = (_FFN_COMMON if spec.scope == 'ffn' else _LAYER_COMMON)
    pairs = pairs + _ACT_FILES[(spec.scope, spec.activation)]
    if spec.synth_only:
        # LUT-sweep stamps exist to measure area, so they ship csynth only.
        pairs = [p for p in pairs if not p[1].startswith('tb/')]
    out = {}
    for src, dst in pairs:
        text = open(os.path.join(KERNELS, src)).read()
        if spec.scope == 'layer':
            text = _apply_residual(dst, text, spec)
        out[dst] = text
    return out


def gen_config_hpp(spec):
    tmpl = (prose.CONFIG_FFN if spec.scope == 'ffn' else prose.CONFIG_LAYER)
    return tmpl[spec.activation].format(
        in_dim=spec.input_dim, hidden=spec.hidden_dim, out_dim=spec.output_dim,
        lut_size=spec.lut_size, gelu_lut_size=spec.gelu_lut_size,
        seq_len=spec.seq_len, hidden_note=spec.hidden_note,
        lut_note=spec.lut_note, ffn_note=spec.ffn_note)


_TITLE_ACT = {'bern': 'Bernstein', 'gelu_lut': 'GeLU-LUT', 'gelu_poly': 'GeLU-poly'}


def rom_headers(spec):
    """ROM headers this variant compiles in. gelu_poly evaluates erf directly
    and so has no activation table."""
    roms = ['bias_rom.hpp']
    if spec.activation != 'gelu_poly':
        roms.append('activation_lut_rom.hpp')
    return roms


def gen_tcl(spec, profile, stage):
    """csim / csynth / cosim.

    NOTE: held to semantic equivalence with the shipped scripts, not
    byte-identity. The originals carry hand-written per-variant titles, some
    inconsistent (gelu_ffn_lut_312x600x312 is captioned 312x1200x312, and two
    layer scripts have no caption at all). Reproducing that would mean encoding
    a wrong dimension into a generator. What must match — and does — is the
    solution name, the file set and order, the part, the clock and the
    interface configuration.
    """
    title = (f'{_TITLE_ACT[spec.activation]} '
             f'{"FFN" if spec.scope == "ffn" else "full layer"} '
             f'({spec.input_dim}x{spec.hidden_dim}x{spec.output_dim})')
    if spec.scope == 'ffn' and spec.name.startswith('bern_lut_sweep/'):
        title += f' LUT={spec.lut_size}'
    caption = {'csynth': 'C synthesis', 'csim': 'C simulation',
               'cosim': f'RTL co-simulation ({spec.cosim_samples} samples)'}[stage]

    files = kernel_files(spec)
    # Shipped order puts the activation implementation right after linear.cpp.
    act_src = [f for f in _ACT_FILES[(spec.scope, spec.activation)]
               if f[1].startswith('src/')]
    srcs = ['src/linear.cpp'] + [act_src[0][1]] + \
           [f for f in files if f.startswith('src/')
            and f not in ('src/linear.cpp', act_src[0][1], act_src[1][1])] + \
           [act_src[1][1]]
    incs = [f for f in files if f.startswith('include/')]
    # config.hpp and types.hpp lead; the rest follow in library order.
    incs = ['include/config.hpp', 'include/types.hpp'] + \
           [f for f in incs if f not in ('include/types.hpp',)]

    L = [f'# {title} \u2014 {caption}']
    L.append(f'open_project{" -reset" if stage == "csynth" else ""} {spec.hls_proj}')
    L.append(f'set_top {spec.top_fn}')
    L += [f'add_files ../{f}' for f in srcs]
    if stage == 'csynth':
        if spec.scope == 'ffn':
            L += [f'add_files ../{f}' for f in incs]
            L += [f'add_files ../include/{r}' for r in rom_headers(spec)]
        L.append('open_solution -reset solution1')
    else:
        tb = f'tb/{"ffn" if spec.scope == "ffn" else "layer"}_tb.cpp'
        cflags = (f' -cflags "-DNUM_TEST_SAMPLES={spec.cosim_samples}"'
                  if stage == 'cosim' else '')
        L.append(f'add_files -tb ../{tb}{cflags}')
        L.append('open_solution "solution1"')
    L += [f'set_part {{{profile.part}}}',
          f'create_clock -period {profile.clock_ns} -name default']
    if stage == 'csynth':
        L.append('config_compile -pragma_strict_mode=true')
        L += list(profile.axi_config)
        L += ['csynth_design', 'close_project']
    else:
        L.append({'csim': 'csim_design', 'cosim': 'cosim_design -rtl verilog'}[stage])
    L.append('exit')
    return '\n'.join(L) + '\n'
