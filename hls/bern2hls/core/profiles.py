"""Backend profiles: everything about *how* a design is emitted, as opposed
to *what* the design computes.

A front-end (fc, rules, bert) decides the algorithm and the weights; a
BackendProfile decides the fixed-point types, the multiplier datapath, the
ROM binding, the testbench style, the target part and the TCL flavour. The
same profile therefore applies across front-ends — the narrow-multiply +
LUTRAM patches in lowpower_fpga are applied identically to the FC and the
rule kernels.

INVARIANT: ``PROFILES['kv260_default']`` is ``BackendProfile()`` with only a
name — i.e. every default below reproduces the literal that was hard-coded
in the emitters before profiles existed. ``verify --flow fc`` (67 projects,
byte-identical) is the enforcement mechanism; see ``assert_default_profile``.
"""

from dataclasses import dataclass, field, fields, replace

__all__ = ['BackendProfile', 'PROFILES', 'get_profile', 'assert_default_profile']


@dataclass(frozen=True)
class BackendProfile:
    name: str = 'kv260_default'

    # ---- numeric types -------------------------------------------------
    # 'literal': config.hpp carries `constexpr unsigned int FIXED_TOTAL_BITS = 32;`
    # 'macro'  : types.hpp carries an #ifndef DATA_W/DATA_I block so the width
    #            can be overridden per-run with -DDATA_W=... (the lowpower
    #            quantization sweep rides on this).
    types_style: str = 'literal'
    data_w: int = 32
    data_i: int = 16
    # Ints under types_style='literal'; under 'macro' these may be strings that
    # are expressions over DATA_W/DATA_I, e.g. '(DATA_W + 16)'.
    acc_w: object = 48
    acc_i: object = 24
    # Header comment reproduced verbatim above the macro block. Explains the
    # DATA_I floor and the AP_RND/AP_SAT choice; dataset-specific prose, so it
    # is carried as data (see the prose note in the module docstring).
    types_comment: str = ''
    # None -> `ap_fixed<W, I>`; 'AP_RND, AP_SAT' -> rounding/saturating data_t
    data_qmode: str = None
    # Name of the exact-product typedef used by narrow_mult (None = no typedef).
    prod_type: str = None
    # Width of that typedef, as (total, int). Strings are emitted verbatim so
    # macro forms like ('2*DATA_W', '2*DATA_I') work.
    prod_bits: tuple = None
    # Trailing comment on the prod_t typedef line (byte-identity, see R4).
    prod_comment: str = ''

    # ---- multiplier datapath -------------------------------------------
    # False: `acc += acc_t(a) * acc_t(b)`  (widens both operands first)
    # True : `prod_t p = a * b; acc += p;` (exact narrow product, saves DSPs)
    narrow_mult: bool = False
    # Trailing comment reproduced verbatim on the narrow-mult line. The
    # hand-patched lowpower sources use two different wordings for the same
    # patch, so byte-identity requires carrying it as data.
    narrow_mult_comment: str = ''
    bind_op_mult: str = None            # None | 'dsp' | 'fabric'

    # ---- ROM binding (rules front-end) ----------------------------------
    rom_impl: str = None                # None | 'bram' | 'lutram'
    rom_pragma: str = None              # None | 'reshape_dim3' | 'partition_dim3'
    rom_pragma_comment: str = ''        # verbatim prose above the pragmas

    # ---- activation -----------------------------------------------------
    # 'per_neuron_lut': NEURON_ACT_LUT[H][E], one LUT row per neuron
    # 'shared_basis'  : BASIS_LUT[E][deg+1] + NEURON_COEFF[H][deg+1]
    activation_impl: str = 'per_neuron_lut'
    # Sampling grid for the shared Bernstein basis, in bern_math.lut_grid's
    # vocabulary. The shared-basis ROMs were generated later than the adult
    # per-neuron LUTs and use the torch 2.x grid while DatasetSpec.lut_grid
    # stays 'torch1' — one dataset, two grids. The grid is only half of it:
    # the basis must also be evaluated through BernsteinLayer.bern_basis
    # (lgamma binomial); math.comb in float32 does not reproduce the shipped
    # values on the identical grid.
    basis_grid: str = 'torch2'
    act_bind_op: str = None             # None | 'fabric' — on the lerp/coeff mults

    # ---- unrolling ------------------------------------------------------
    max_output_unroll: int = 8          # cap for the output layer
    max_hidden_unroll: int = 16         # cap for hidden->hidden layers
    # None -> inherit DatasetSpec.unroll_strategy; else 'full'|'tier1'|'tier2'
    unroll_strategy: str = None

    # ---- testbench / data ------------------------------------------------
    # 'selfcheck': compare logits against test_output_ref.txt within tolerance
    # 'accuracy' : argmax against test_labels.txt over a large sample set
    tb_mode: str = 'selfcheck'
    tb_tolerance: float = 0.5
    # Wrap the testbench's sample count in #ifndef so a run can override it.
    tb_guard_num_test: bool = False
    test_vector_set: str = 'default'    # | 'adult9045' | 'rules200'
    # 'per_project' -> data/ next to the sources; 'shared:<relpath>' -> the TCL
    # points -tb at a directory outside the project (lowpower quant sweep).
    data_dir_mode: str = 'per_project'
    emit_data: bool = True

    # ---- interface ------------------------------------------------------
    axi_width: int = 256                # axi family only

    # ---- tool -----------------------------------------------------------
    part: str = 'xck26-sfvc784-2LV-c'
    clock_ns: float = 10
    # 'literal': set_part {xc...} / create_clock -period 10
    # 'env'    : read HLS_PART / HLS_PERIOD / HLS_TAG / HLS_SKIP_CSIM / QW / QI
    tcl_style: str = 'literal'
    tcl_stages: tuple = ('csim', 'csynth')
    tcl_tag: str = 'kv260'          # HLS_TAG default; names the solution dir
    proj_name_tmpl: str = '{name}_hls'
    # extra config_interface lines (the transformer streams weights)
    axi_config: tuple = ()

    def derive(self, **overrides):
        """A copy with fields replaced (the profile is frozen)."""
        return replace(self, **overrides)

    # -- emitter helpers: keep the C++ spelling in one place --------------

    def data_t_decl(self):
        """The `ap_fixed<...>` template arguments for data_t."""
        base = ('FIXED_TOTAL_BITS, FIXED_INT_BITS' if self.types_style == 'literal'
                else 'DATA_W, DATA_I')
        return f'{base}, {self.data_qmode}' if self.data_qmode else base

    def acc_t_decl(self):
        """Template arguments for acc_t.

        Under 'macro' the typedef names the macros; `acc_w`/`acc_i` are then
        the *defaults* those macros take in the #ifndef block, not the spelling
        used at the typedef.
        """
        if self.types_style == 'macro':
            return 'ACC_W, ACC_I'
        return f'{self.acc_w}, {self.acc_i}'

    def macro_block(self):
        """The `#ifndef DATA_W/...` block that makes types_style='macro' valid.

        Emitted immediately above the typedefs so the widths can be overridden
        per synthesis run with -DDATA_W=16 -DDATA_I=8 (the quantization sweep).
        """
        if self.types_style != 'macro':
            return []
        out = self.types_comment.splitlines() if self.types_comment else []
        for macro, default in (('DATA_W', 'FIXED_TOTAL_BITS'),
                               ('DATA_I', 'FIXED_INT_BITS'),
                               ('ACC_W', self.acc_w), ('ACC_I', self.acc_i)):
            out += [f'#ifndef {macro}', f'#define {macro} {default}', '#endif']
        return out

    def prod_t_decl(self):
        if not self.prod_bits:
            return None
        return f'{self.prod_bits[0]}, {self.prod_bits[1]}'


def _defaults_only(name):
    """A profile that is BackendProfile() in every field but `name`."""
    return BackendProfile(name=name)


# Prose reproduced verbatim from the hand-patched lowpower sources. Carried as
# data because byte-identity requires it and it cannot be derived (see R4 in
# the plan): the same patch appears with three different trailing comments.
_LP_TYPES_COMMENT = (
    '// Quantization-sweep parameterization: override via -DDATA_W=16 -DDATA_I=8.\n'
    '// DATA_I must be >= 7: adult inputs contain ordinal codes up to 40 (+sign).\n'
    '// AP_RND/AP_SAT so narrow widths round and saturate instead of wrap.')

# Common to every lowpower profile: macro-parameterized widths, rounding/
# saturating data_t, the 9045-sample accuracy testbench, env-driven TCL, and
# the Spartan-7 target at 15 ns.
_LP_BASE = dict(
    types_style='macro', data_qmode='AP_RND, AP_SAT',
    acc_w='(DATA_W + 16)', acc_i='(DATA_I + 8)', types_comment=_LP_TYPES_COMMENT,
    tb_mode='accuracy', test_vector_set='adult9045',
    data_dir_mode='shared:../../data', emit_data=False,
    tcl_style='env', part='xc7s15ftgb196-1', clock_ns=15, tcl_tag='xc7s15',
)
_LP_QPROD = dict(prod_type='qprod_t', prod_bits=('2 * DATA_W', '2 * DATA_I'),
                 prod_comment='  // exact data_t*data_t product')

PROFILES = {
    # The frozen baseline: reproduces the 67 shipped projects byte-identically.
    'kv260_default': _defaults_only('kv260_default'),

    # --- low-power (XC7S15 / XC7A15T) -----------------------------------
    # Widths only: the kernel itself is unchanged from kv260_default.
    'lp_quant': BackendProfile(name='lp_quant', **_LP_BASE),
    # + exact narrow product, explicitly bound to a DSP. This variant carries
    # no comment on either the typedef or the product line, where the
    # per_neuron sweep comments both — same patch, three prose variants.
    'lp_quant_dspbind': BackendProfile(
        name='lp_quant_dspbind', **{**_LP_BASE, **_LP_QPROD, 'prod_comment': ''},
        narrow_mult=True, bind_op_mult='dsp'),
    # The architecture sweep: narrow product, output-layer unroll capped at 4.
    'lp_per_neuron': BackendProfile(
        name='lp_per_neuron', **_LP_BASE, **_LP_QPROD,
        narrow_mult=True, narrow_mult_comment='  // exact narrow product, 1 DSP',
        max_output_unroll=4),
    'lp_pn_dspbind': BackendProfile(
        name='lp_pn_dspbind', **_LP_BASE, **_LP_QPROD,
        narrow_mult=True, narrow_mult_comment='  // exact narrow product',
        bind_op_mult='dsp', max_output_unroll=4),
    # Shared-basis activation: BASIS_LUT[E][deg+1] + NEURON_COEFF[H][deg+1],
    # with the small basis multiplies pushed into fabric to spare DSPs.
    'lp_shared_basis': BackendProfile(
        name='lp_shared_basis', **_LP_BASE, **_LP_QPROD,
        narrow_mult=True, narrow_mult_comment='  // exact narrow product, 1 DSP',
        max_output_unroll=4, activation_impl='shared_basis', act_bind_op='fabric'),
    # 14x4x2 only: shipped with the shared activation but with its linear
    # layers left un-patched. A genuine inconsistency in the hand-edited
    # source, not a design choice — see R5.
    'lp_shared_basis_x4': BackendProfile(
        name='lp_shared_basis_x4', **_LP_BASE, **_LP_QPROD,
        max_output_unroll=4, activation_impl='shared_basis', act_bind_op='fabric'),

    # --- transformer ---------------------------------------------------
    'bert_kv260': BackendProfile(
        name='bert_kv260', tcl_stages=('csim', 'csynth', 'cosim'),
        axi_config=('config_interface -m_axi_latency=64',
                    'config_interface -m_axi_max_bitwidth=256')),

    # --- low-power applied to the RULE front-end -------------------------
    # The same backend treatment as the FC profiles above — exact narrow
    # product, distributed ROM — which is the point: the profile axis is not
    # FC-specific. Feeds the R50/R29 rows of the XC7S15 deployment table.
    'lp_rules_dspopt': BackendProfile(
        name='lp_rules_dspopt', part='xc7s15ftgb196-1', clock_ns=15,
        tcl_style='env', tcl_tag='xc7s15', tb_guard_num_test=True,
        narrow_mult=True, narrow_mult_comment='  // 8x16 narrow mult, exact',
        prod_type='prod_t', prod_bits=(24, 16),
        prod_comment='   // exact int8 x fix<16,8> product (1 DSP48E1)'),
    # + move the rule ROM off block RAM. Only the 50-rule design needs this;
    # the 29-rule one already fits and keeps its BRAM binding.
    'lp_rules_dspopt_lutram': BackendProfile(
        name='lp_rules_dspopt_lutram', part='xc7s15ftgb196-1', clock_ns=15,
        tcl_style='env', tcl_tag='xc7s15', tb_guard_num_test=True,
        narrow_mult=True, narrow_mult_comment='  // 8x16 narrow mult, exact',
        prod_type='prod_t', prod_bits=(24, 16),
        prod_comment='   // exact int8 x fix<16,8> product (1 DSP48E1)',
        rom_impl='lutram', rom_pragma='reshape_dim3',
        rom_pragma_comment=(
            '    // Rule-weight ROM ({rom_dims} int8 = {rom_kb} KB) in distributed'
            ' LUTRAM instead of\n'
            '    // block RAM: frees 15 of the 20 BRAM18K on XC7S15'
            ' (rule ROM was 100% usage).')),
}


def get_profile(name_or_profile):
    """Resolve a profile name (or pass a BackendProfile through unchanged)."""
    if isinstance(name_or_profile, BackendProfile):
        return name_or_profile
    if name_or_profile is None:
        return PROFILES['kv260_default']
    try:
        return PROFILES[name_or_profile]
    except KeyError:
        raise ValueError(
            f"unknown profile '{name_or_profile}' "
            f"(available: {', '.join(sorted(PROFILES))})") from None


def assert_default_profile():
    """Guard the invariant that kv260_default carries no overrides.

    Every emitter branch that reads a profile field must reduce, under this
    profile, to the literal it replaced. If someone changes a dataclass
    default to serve a new target, this fires before `verify` has to.
    """
    base, ref = PROFILES['kv260_default'], BackendProfile()
    drift = [f.name for f in fields(BackendProfile)
             if f.name != 'name' and getattr(base, f.name) != getattr(ref, f.name)]
    if drift:
        raise AssertionError(
            f"kv260_default deviates from BackendProfile() defaults in: {drift}. "
            "The 67-project byte-identity depends on these being identical.")
    return True
