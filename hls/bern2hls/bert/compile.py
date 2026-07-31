"""bert: a TinyBERT encoder block -> a complete Vitis HLS project.

Stamps the fixed kernel library plus the parts the study varies. The weight
streams are written too, since they are the design's actual inputs — they are
large (a layer variant runs to ~10 MB of text) and regenerated rather than
stored.
"""

import os

from ..core.emit import write_file
from ..core.profiles import get_profile
from . import weights as W
from .codegen import bert_family as bf
from .model_extract import extract_layer
from .variants import VARIANTS

PAPER_VARIANTS = {
    ('bern', 312, 'ffn'): ('h312', 'bern_ffn_312x312x312', False),
    ('bern', 312, 'layer'): ('h312', 'bern_layer_lut50', False),
    ('gelu', 312, 'ffn'): ('h312', 'gelu_ffn_poly_312x312x312', False),
    ('gelu', 312, 'layer'): ('h312', 'gelu_poly_layer', False),
    ('bern', 600, 'ffn'): ('h600', 'bern_ffn_312x600x312_lut50', False),
    ('bern', 600, 'layer'): ('h600', 'bern_layer_lut50', True),
    ('gelu', 600, 'ffn'): ('h600', 'gelu_ffn_poly_312x600x312', False),
    ('gelu', 600, 'layer'): ('h600', 'gelu_poly_layer_h600', False),
    ('gelu', 1200, 'ffn'): ('h600', 'gelu_ffn_poly_312x1200x312', True),
    ('gelu', 1200, 'layer'): ('h600', 'gelu_poly_layer', True),
}


def _rom_and_data(spec, weights_root, base, with_data):
    w = W.load_layer(os.path.join(weights_root, spec.weights))
    if spec.activation == 'bern':
        Wf, bf_, inv, _ = W.fuse_bern(w['fc1.weight'], w['fc1.bias'],
                                      w['bern.input_bounds'])
        write_file(os.path.join(base, 'include/bias_rom.hpp'),
                   W.gen_bias_rom(bf_, w['fc2.bias'], fused=True))
        write_file(os.path.join(base, 'include/activation_lut_rom.hpp'),
                   W.gen_bern_lut_rom(
                       W.build_bern_lut(w['bern.bern_coeffs'], spec.lut_size),
                       spec.lut_size))
        if not spec.synth_only:
            write_file(os.path.join(base, 'include/norm_params_rom.hpp'),
                       W.gen_norm_rom(inv))
        fc1 = Wf
    else:
        write_file(os.path.join(base, 'include/bias_rom.hpp'),
                   W.gen_bias_rom(w['fc1.bias'], w['fc2.bias'], fused=False))
        if spec.activation == 'gelu_lut':
            write_file(os.path.join(base, 'include/activation_lut_rom.hpp'),
                       W.gen_gelu_lut_rom(W.build_gelu_lut(spec.gelu_lut_size),
                                          spec.gelu_lut_size))
        fc1 = w['fc1.weight']
    if with_data and not spec.synth_only:
        W.write_flat(os.path.join(base, 'data/fc1_weights.txt'),
                     W.pack_fc1_colvec(fc1, 32))
        W.write_flat(os.path.join(base, 'data/fc2_weights.txt'), w['fc2.weight'])


def compile_variants(bundle=None, scope=None, only=None, out_dir=None,
                     with_data=False, include_legacy=False, weights_root=None):
    from ..fc.datasets import ARTIFACT_DIR
    picked = [v for v in VARIANTS
              if (bundle in (None, v.bundle)) and (scope in (None, v.scope))
              and (only in (None, v.name))
              and (include_legacy or not v.legacy)]
    if not picked:
        raise ValueError('no variants match that selection')
    out_dir = out_dir or os.path.join(ARTIFACT_DIR, 'generated', 'bert')
    weights_root = weights_root or os.path.join(ARTIFACT_DIR, 'bert_weights')

    written = []
    for i, spec in enumerate(picked):
        print(f"\n{'='*60}")
        print(f"  [{i+1}/{len(picked)}] {spec.bundle}/{spec.scope}/{spec.name}  "
              f"({spec.activation}, hidden={spec.hidden_dim})")
        print(f"{'='*60}")
        base = os.path.join(out_dir, spec.bundle, spec.scope, spec.name)
        for rel, text in bf.kernel_files(spec).items():
            write_file(os.path.join(base, rel), text)
        write_file(os.path.join(base, 'include/config.hpp'), bf.gen_config_hpp(spec))
        _rom_and_data(spec, weights_root, base, with_data)
        profile = get_profile(spec.profile)
        stages = ['csynth'] if spec.synth_only else ['csim', 'csynth', 'cosim']
        for stage in stages:
            write_file(os.path.join(base, f'script/run_{stage}.tcl'),
                       bf.gen_tcl(spec, profile, stage))
        written.append(f'{spec.bundle}/{spec.scope}/{spec.name}')

    print(f"\nGenerated {len(written)} project(s) under {out_dir}")
    if not with_data:
        print("  (pass --with-data to also write the m_axi weight streams; "
              "they are ~10 MB per layer variant and csim/cosim need them)")
    return written


def compile_checkpoint(checkpoint, out_dir, scope='layer', layer=0,
                       with_data=False, weights_dir=None):
    """Compile a paper configuration directly from a Bern2Edge release."""
    import torch

    checkpoint = os.path.abspath(checkpoint)
    raw = torch.load(checkpoint, map_location='cpu', weights_only=False)
    if 'state_dict' in raw:
        meta = raw.get('meta', {})
        act = meta.get('act')
        hidden = int(meta.get('hidden'))
    else:
        state = raw['bert']
        key = f'bert.encoder.layer.{layer}.intermediate.dense.weight'
        act, hidden = 'gelu', int(state[key].shape[0])
    key = (act, hidden, scope)
    if key not in PAPER_VARIANTS:
        raise ValueError(
            f'unsupported paper Transformer configuration {key}; '
            'expected Bernstein/GeLU h=312 or h=600, or GeLU teacher h=1200'
        )
    bundle, name, legacy = PAPER_VARIANTS[key]
    kind = 'bern' if act == 'bern' else 'gelu'
    weight_tag = 'h1200' if hidden == 1200 else f'h{hidden}'
    weights_dir = weights_dir or os.path.join(out_dir, '_weights')
    slim_path = os.path.join(weights_dir, weight_tag, f'{kind}_layer{layer}.npz')
    extract_layer(checkpoint, slim_path, layer=layer)
    return compile_variants(
        bundle=bundle,
        scope=scope,
        only=name,
        out_dir=out_dir,
        with_data=with_data,
        include_legacy=legacy,
        weights_root=weights_dir,
    )
