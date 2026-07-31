"""Registry of the transformer designs the artifact reproduces.

Two scopes: the FFN sub-block alone (fc1 -> activation -> fc2), and a whole
encoder layer (attention + softmax + LayerNorm + FFN). Three activations:
per-channel Bernstein via LUT, the shared-LUT GeLU baseline, and the GeLU
polynomial baseline that evaluates erf in hardware.

`residual_fix` is the one real fork. The h312 study widened two residual paths
from data_t to wide_t after finding that fc2 outputs reach about +-430 in that
model, well past data_t's +-128, which saturates and corrupts the LayerNorm
variance. The fix was never back-ported, so four h600 layer designs were
synthesized without it. They are marked legacy: the artifact ships the fixed
kernels and derives the unfixed form by a stated substitution, rather than
shipping two implementations of the same thing.
"""

from dataclasses import dataclass

# Trailing comments that drift between bundles; see codegen/prose.py.
_H312_HID = '   // fc1 expansion (h312 bundle: equal-width)'
_H600_HID = '   // fc1 expansion'
_H312_LUT = '                    // entries per channel (h312 default)'
_H600_LUT = '                   // entries per channel'
# The layer configs annotate the FFN-dims heading only in the equal-width bundle.
_H312_FFN = '; h312 bundle: equal-width'
_H600_FFN = ''


_PROJ = {('h312', 'bern_ffn_312x312x312'): 'bern_ffn_hls', ('h312', 'gelu_ffn_lut_312x312x312'): 'gelu_ffn_lut_hls', ('h312', 'gelu_ffn_poly_312x312x312'): 'gelu_ffn_poly_hls', ('h600', 'bern_ffn_312x600x312'): 'bern_ffn_hls', ('h600', 'bern_ffn_312x600x312_lut50'): 'bern_ffn_lut50_hls', ('h600', 'gelu_ffn_lut_312x600x312'): 'gelu_ffn_lut600_hls', ('h600', 'gelu_ffn_poly_312x600x312'): 'gelu_ffn_poly600_hls', ('h600', 'gelu_ffn_lut_312x1200x312'): 'gelu_ffn_lut_hls', ('h600', 'gelu_ffn_poly_312x1200x312'): 'gelu_ffn_poly_hls'}


def _derive_proj(spec):
    if spec.name.startswith('bern_lut_sweep/'):
        return f'bern_ffn_lut{spec.lut_size}_hls'
    return f'{spec.name}_hls'


@dataclass
class VariantSpec:
    name: str                       # project dir name
    bundle: str                     # 'h312' | 'h600'
    scope: str                      # 'ffn' | 'layer'
    activation: str                 # 'bern' | 'gelu_lut' | 'gelu_poly'
    hidden_dim: int
    input_dim: int = 312
    output_dim: int = 312
    lut_size: int = 64              # Bernstein entries/channel; None for gelu_poly
    gelu_lut_size: int = 512
    bern_degree: int = 15
    seq_len: int = None             # layer scope only
    weights: str = ''               # slim bundle, rel. to bert_weights/
    hidden_note: str = _H312_HID
    lut_note: str = _H312_LUT
    ffn_note: str = _H312_FFN
    # False reproduces the pre-fix residual path (the four legacy h600 layers).
    residual_fix: bool = True
    # The h312 sources carry an explanatory comment above each widened residual
    # that the h600 fixed pair does not.
    residual_comment: bool = True
    legacy: bool = False            # excluded from the default verify set
    synth_only: bool = False        # LUT-sweep stamps: no testbench, csynth only
    num_samples: int = 16
    cosim_samples: int = 2
    profile: str = 'bert_kv260'
    ground_truth: str = None
    hls_proj: str = ''       # Vitis solution dir name; collect scans for it

    def __post_init__(self):
        if not self.hls_proj:
            self.hls_proj = _PROJ.get((self.bundle, self.name)) or _derive_proj(self)

    @property
    def top_fn(self):
        return 'ffn_top' if self.scope == 'ffn' else 'layer_top'


def _gt(bundle, scope, name):
    b = 'bernbert_synthesis_bundle' + ('_h312' if bundle == 'h312' else '')
    d = 'ffn_synthesis_comparison' if scope == 'ffn' else 'layer_synthesis_comparison'
    return f'transformer_bern_synthesis/{b}/{d}/{name}'


def _ffn(bundle, name, act, hidden, lut=64, **kw):
    notes = ({'hidden_note': _H312_HID, 'lut_note': _H312_LUT, 'ffn_note': _H312_FFN}
             if bundle == 'h312'
             else {'hidden_note': _H600_HID, 'lut_note': _H600_LUT, 'ffn_note': _H600_FFN})
    if act != 'bern':
        # the GeLU FFN configs annotate none of the dimension lines
        notes['hidden_note'] = ''
    notes.update(kw)
    return VariantSpec(name=name, bundle=bundle, scope='ffn', activation=act,
                       hidden_dim=hidden, lut_size=lut,
                       ground_truth=_gt(bundle, 'ffn', name),
                       **{'weights': f'{bundle}/{"bern" if act == "bern" else "gelu"}_layer0.npz',
                          **notes})


def _layer(bundle, name, act, hidden, lut=64, **kw):
    notes = ({'hidden_note': _H312_HID, 'lut_note': _H312_LUT, 'ffn_note': _H312_FFN}
             if bundle == 'h312'
             else {'hidden_note': _H600_HID, 'lut_note': _H600_LUT, 'ffn_note': _H600_FFN})
    # Only the h312 gelu-poly layer annotates its HIDDEN_DIM line.
    notes['hidden_note'] = ('   // h312 bundle: equal-width FFN'
                            if (bundle == 'h312' and act == 'gelu_poly') else '')
    notes.update(kw)
    return VariantSpec(name=name, bundle=bundle, scope='layer', activation=act,
                       hidden_dim=hidden, lut_size=lut, seq_len=16, num_samples=2,
                       cosim_samples=1,
                       ground_truth=_gt(bundle, 'layer', name),
                       **{'weights': f'{bundle}/{"bern" if act == "bern" else "gelu"}_layer0.npz',
                          **notes})


VARIANTS = [
    # ---- FFN, h312: the equal-width study that isolates the activation ----
    _ffn('h312', 'bern_ffn_312x312x312', 'bern', 312, 64),
    _ffn('h312', 'gelu_ffn_lut_312x312x312', 'gelu_lut', 312),
    _ffn('h312', 'gelu_ffn_poly_312x312x312', 'gelu_poly', 312),
    *[_ffn('h312', f'bern_lut_sweep/lut{n}', 'bern', 312, n, synth_only=True)
      for n in (32, 50, 64, 128, 256)],

    # ---- FFN, h600 ----
    _ffn('h600', 'bern_ffn_312x600x312', 'bern', 600, 64),
    _ffn('h600', 'bern_ffn_312x600x312_lut50', 'bern', 600, 50),
    *[_ffn('h600', f'bern_lut_sweep/lut{n}', 'bern', 600, n, synth_only=True)
      for n in (64, 128)],
    _ffn('h600', 'gelu_ffn_lut_312x600x312', 'gelu_lut', 600),
    _ffn('h600', 'gelu_ffn_poly_312x600x312', 'gelu_poly', 600),
    # the 1200-wide baselines are the teacher, not the h600 student
    _ffn('h600', 'gelu_ffn_lut_312x1200x312', 'gelu_lut', 1200, weights='h1200/gelu_layer0.npz'),
    _ffn('h600', 'gelu_ffn_poly_312x1200x312', 'gelu_poly', 1200, weights='h1200/gelu_layer0.npz'),

    # ---- full encoder layer, h312 ----
    _layer('h312', 'bern_layer', 'bern', 312, 64),
    _layer('h312', 'bern_layer_lut50', 'bern', 312, 50),
    _layer('h312', 'gelu_lut_layer', 'gelu_lut', 312),
    _layer('h312', 'gelu_poly_layer', 'gelu_poly', 312),

    # ---- full encoder layer, h600: the four pre-fix designs ----
    _layer('h600', 'bern_layer', 'bern', 600, 64,
           residual_fix=False, residual_comment=False, legacy=True),
    _layer('h600', 'bern_layer_lut50', 'bern', 600, 50,
           residual_fix=False, residual_comment=False, legacy=True),
    _layer('h600', 'gelu_lut_layer', 'gelu_lut', 1200, weights='h1200/gelu_layer0.npz',
           residual_fix=False, residual_comment=False, legacy=True),
    _layer('h600', 'gelu_poly_layer', 'gelu_poly', 1200, weights='h1200/gelu_layer0.npz',
           residual_fix=False, residual_comment=False, legacy=True),
    # ...and the two that did get the fix, but without the explanatory comments
    _layer('h600', 'gelu_lut_layer_h600', 'gelu_lut', 600, residual_comment=False),
    _layer('h600', 'gelu_poly_layer_h600', 'gelu_poly', 600, residual_comment=False),
]

BY_NAME = {f'{v.bundle}/{v.scope}/{v.name}': v for v in VARIANTS}
