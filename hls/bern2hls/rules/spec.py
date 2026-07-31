"""Registry of the rule-classifier projects the artifact reproduces.

Scope is set by what the paper cites: the fallback ablation, the sparsity-K
sweep, and the tree-fallback architecture sweep. The older v1/v2/v3 hardware
sweeps are not cited anywhere and are out of scope (v3's dense generator is
in fact dead code — every shipped v3 project came from the sparse one).
"""

import os
from dataclasses import dataclass

ARTIFACT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# Bundled rule sets and fallbacks (tools/package_rule_inputs.py). Self-contained
# by default; falls back to the research tree if the bundle is absent.
RULE_MODELS_DIR = os.path.join(ARTIFACT_DIR, 'rule_models')


def model_root(default_root):
    """Prefer the packaged inputs, so a clone needs nothing outside itself."""
    return RULE_MODELS_DIR if os.path.isdir(RULE_MODELS_DIR) else default_root


def asset_path(rel):
    """Resolve one input, preferring the npz the packager wrote over a
    torch checkpoint (the rules front-end runs on numpy alone)."""
    if os.path.isdir(RULE_MODELS_DIR):
        npz = os.path.join(RULE_MODELS_DIR, os.path.splitext(rel)[0] + '.npz')
        if os.path.isfile(npz):
            return npz
        direct = os.path.join(RULE_MODELS_DIR, rel)
        if os.path.isfile(direct):
            return direct
    return None


@dataclass
class RuleSpec:
    name: str                       # project dir name
    rules_json: str                 # rel. to the rule-model root
    fallback_kind: str = 'none'     # lr|network|small_nn|tree|none
    fallback_asset: str = None      # rel. to the rule-model root
    phase1: str = 'dense'           # dense | sparse
    k_sparse: int = None
    nn_hidden: int = None           # emits NN_HID
    tree_max_depth: int = None      # emits TREE_MAX_DEPTH
    scope: str = 'full'             # full | fb_only
    top_fn: str = 'rule_classifier_top'
    hls_proj: str = ''
    num_test: int = 9045
    # Which projection the golden reference dots with. f4v uses the retained
    # top-k (`sparse_weights`); the dense tree_arch suite uses `weight_vector`.
    # Swapping these silently changes test_output_ref.txt.
    dot_source: str = 'sparse_weights'
    prose: str = 'f4v'
    profile: str = 'kv260_default'
    tcl_label: str = ''             # what the TCL's closing puts announces
    arch: str = ''                  # source architecture, when it differs from `name`
    ground_truth: str = None        # rel. to --gt-root, for verification


_F4V = '14x32x2_p90_k7_sca0.5_ca0.1'
_FB = {'network': ('network/fallback_network.pth', 32, None),
       'small_nn': ('small_nn/fallback_net.pt', 4, None),
       'tree': ('tree/fallback_tree_int.npz', None, 4),
       'lr': ('lr/rules_int8.json', None, None)}


def _f4v_specs():
    out = []
    for kind, (asset, hid, depth) in _FB.items():
        gt = f'rule_distillation/fallback_4_variance/hls/{kind}'
        out.append(RuleSpec(
            name=kind, rules_json=f'{_F4V}/network/rules_int8.json',
            fallback_kind=kind, fallback_asset=f'{_F4V}/{asset}',
            nn_hidden=hid, tree_max_depth=depth,
            hls_proj=f'fb_{kind}_hls', ground_truth=gt))
        out.append(RuleSpec(
            name=f'{kind}_fb_only', rules_json=f'{_F4V}/network/rules_int8.json',
            fallback_kind=kind, fallback_asset=f'{_F4V}/{asset}',
            nn_hidden=hid, tree_max_depth=depth,
            scope='fb_only', top_fn='fallback_top',
            hls_proj=f'fb_{kind}_only_hls', ground_truth=f'{gt}/fb_only'))
    return out


def _ksweep_specs():
    """K = 1..14 non-zero weights per condition over one 14x32x2 classifier.
    K is the only independent variable, so K_SPARSE is per-project."""
    stem = ('kd_fc_14x32x2_bern_deg3_alpha0.5_T2_lr0.006_wd0.0001_seed6')
    out = []
    for k in range(1, 15):
        out.append(RuleSpec(
            name=f'k{k}',
            rules_json=f'k_sweep_jsons_14x32x2_int8/{stem}_k{k}.json',
            fallback_kind='lr',
            fallback_asset=f'k_sweep_jsons_14x32x2_int8/{stem}_k{k}.json',
            phase1='sparse', k_sparse=k, prose='ksweep', num_test=200,
            hls_proj=f'k{k}_hls',
            ground_truth=f'rule_distillation/sparsity_sweep/hls/k{k}'))
    return out


def _tree_arch_specs():
    """CART fallback held fixed across five source architectures. These rule
    sets are dense (sparsified=False), so the golden reference dots with the
    full weight_vector rather than the retained top-k."""
    archs = ['14x16x2', '14x32x2', '14x128x2', '14x16x8x2', '14x32x16x2']
    out = []
    for a in archs:
        root = f'synthesis_tree/{a}_p90_dense_sca0.5_ca0.1/tree'
        out.append(RuleSpec(
            name=a, rules_json=f'{root}/rules_int8.json',
            fallback_kind='tree', fallback_asset=f'{root}/fallback_tree_int.npz',
            tree_max_depth=4, prose='tree_arch',
            dot_source='weight_vector',
            hls_proj=f'tree_{a}_hls',
            ground_truth=f'rule_distillation/fallback_4_variance_more_arch/tree_hls/{a}'))
    return out


def _rule_lowpower_specs():
    """The rule classifier retargeted to XC7S15 — the same backend treatment
    the FC low-power profiles apply (exact narrow product, distributed ROM),
    which is the evidence that the profile axis is not FC-specific.

    Feeds the R50/R29 rows of the XC7S15 deployment table: R50 is the 50-rule
    classifier (needs the ROM off block RAM), R29 the 29-rule one (already
    fits, keeps its BRAM binding).
    """
    lp = 'lowpower_fpga/rule_tree'
    return [
        RuleSpec(  # R50
            name='dsp_opt_fb4var', rules_json=f'{_F4V}/network/rules_int8.json',
            fallback_kind='tree', fallback_asset=f'{_F4V}/tree/fallback_tree_int.npz',
            tree_max_depth=4, prose='f4v',
            profile='lp_rules_dspopt_lutram',
            hls_proj='fb_tree_dspopt_${tag}_hls', tcl_label='tree',
            ground_truth=f'{lp}/dsp_opt_fb4var'),
        RuleSpec(  # R29
            name='dsp_opt_14x16x8x2',
            rules_json='synthesis_tree/14x16x8x2_p90_dense_sca0.5_ca0.1/tree/rules_int8.json',
            fallback_kind='tree',
            fallback_asset='synthesis_tree/14x16x8x2_p90_dense_sca0.5_ca0.1/tree/'
                           'fallback_tree_int.npz',
            tree_max_depth=4, prose='tree_arch', dot_source='weight_vector',
            arch='14x16x8x2', profile='lp_rules_dspopt',
            hls_proj='tree_14x16x8x2_dspopt_${tag}_hls', tcl_label='14x16x8x2 tree',
            ground_truth=f'{lp}/dsp_opt_14x16x8x2'),
    ]


SUITES = {
    'fallback_4_variance': _f4v_specs(),
    'rule_lowpower': _rule_lowpower_specs(),
    'sparsity_sweep': _ksweep_specs(),
    'tree_arch': _tree_arch_specs(),
}
