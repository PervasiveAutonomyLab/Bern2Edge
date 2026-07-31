"""rules: a released rule set -> complete Vitis HLS projects."""

import os
from pathlib import Path

from ..core.emit import write_file
from ..core.profiles import get_profile
from . import fallback as fb, refmodel as rm
from .codegen import rule_family as rf
from .rules_io import RuleModel
from .spec import RuleSpec, SUITES, asset_path, model_root

FALLBACK_KINDS = ('none', 'lr', 'tree', 'network', 'small_nn')

# Where each suite's inputs live if the packaged bundle is absent.
FALLBACK_ROOTS = {
    'fallback_4_variance': 'rule_distillation/fallback_4_variance',
    'sparsity_sweep': 'rule_distillation/sparsity_sweep',
    'tree_arch': 'rule_distillation/fallback_4_variance_more_arch',
    'rule_lowpower': 'rule_distillation/fallback_4_variance',
}


def _resolve(rel, suite, gt_root):
    return asset_path(rel) or os.path.join(gt_root, FALLBACK_ROOTS[suite], rel)


def _load_test_data(path):
    if not path:
        return None, None
    import numpy as np

    data = np.load(path, allow_pickle=True)
    x_key = 'X_test' if 'X_test' in data else 'X'
    y_key = 'y_test' if 'y_test' in data else 'y'
    return data[x_key].astype('float64'), data[y_key].astype(int)


def _emit_project(base, model, spec, params, profile, X=None, y=None):
    write_file(os.path.join(base, 'include/config.hpp'),
               rf.gen_config_hpp(model, spec))
    write_file(os.path.join(base, 'include/types.hpp'),
               rf.gen_types_hpp(spec, profile))
    write_file(os.path.join(base, 'include/rule_classifier.hpp'),
               rf.gen_hdr_hpp(spec))
    write_file(os.path.join(base, 'include/rule_rom.hpp'),
               rf.gen_rom_hpp(model, model.rom_tables(spec.k_sparse), spec, params))
    write_file(os.path.join(base, 'src/rule_classifier.cpp'),
               rf.gen_src_cpp(model, spec, profile))
    write_file(os.path.join(base, 'script/run_csynth.tcl'),
               rf.gen_tcl(spec, profile))
    if spec.scope != 'fb_only':
        write_file(os.path.join(base, 'tb/rule_classifier_tb.cpp'),
                   rf.gen_tb_cpp(spec, spec.num_test, profile))
        if X is not None:
            n = min(spec.num_test, len(X))
            pred = rm.golden_predict(model, X[:n], spec.fallback_kind,
                                     params, spec.dot_source)
            write_file(os.path.join(base, 'data/test_input.txt'),
                       ''.join(f'{v:.10f}\n' for v in X[:n].flatten()))
            write_file(os.path.join(base, 'data/test_labels.txt'),
                       ''.join(f'{int(v)}\n' for v in y[:n]))
            write_file(os.path.join(base, 'data/test_output_ref.txt'),
                       ''.join(f'{int(v)}\n' for v in pred))


def compile_rule_model(rules_json, out_dir, fallback_kind='none',
                       fallback_asset=None, profile='kv260_default',
                       test_data=None, name=None, num_test=9045,
                       fallback_only=False, prose='f4v', hls_proj=None,
                       arch='', force_dense=False):
    """Generate one HLS project from an arbitrary HLS-ready rule JSON.

    ``rules_json`` must contain the per-condition ``int8_scale`` values written
    by ``bern2edge.rule_extraction.quantize``. LR parameters live in that JSON;
    tree and Bernstein-network fallbacks use a separate asset.
    """
    if fallback_kind not in FALLBACK_KINDS:
        raise ValueError(
            f"unsupported fallback {fallback_kind!r}; choose from "
            f"{', '.join(FALLBACK_KINDS)}"
        )
    rules_json = str(Path(rules_json).resolve())
    if fallback_kind == 'lr':
        fallback_asset = rules_json
    elif fallback_kind != 'none' and not fallback_asset:
        raise ValueError(f"--fallback-model is required for {fallback_kind}")
    elif fallback_asset:
        fallback_asset = str(Path(fallback_asset).resolve())

    model = RuleModel(rules_json)
    phase1 = 'sparse' if model.is_sparse and not force_dense else 'dense'
    k_sparse = model.sparsity_k if phase1 == 'sparse' else None
    params = fb.load(fallback_kind, fallback_asset)
    nn_hidden = (
        int(params['W0'].shape[0])
        if fallback_kind in ('network', 'small_nn') else None
    )
    tree_depth = (
        int(model.fallback.get('max_depth', model.fallback.get('depth', 4)))
        if fallback_kind == 'tree' else None
    )
    project_name = name or Path(out_dir).name
    spec = RuleSpec(
        name=project_name,
        rules_json=rules_json,
        fallback_kind=fallback_kind,
        fallback_asset=fallback_asset,
        phase1=phase1,
        k_sparse=k_sparse,
        nn_hidden=nn_hidden,
        tree_max_depth=tree_depth,
        scope='fb_only' if fallback_only else 'full',
        top_fn='fallback_top' if fallback_only else 'rule_classifier_top',
        num_test=num_test,
        dot_source='sparse_weights' if phase1 == 'sparse' else 'weight_vector',
        prose=prose,
        arch=arch,
        hls_proj=hls_proj or f'{project_name}_hls',
        profile=profile,
    )
    X, y = _load_test_data(test_data)
    _emit_project(str(out_dir), model, spec, params, get_profile(profile), X, y)
    print(
        f"Generated {project_name}: {model.n_rules} rules, "
        f"max {model.max_conds} conditions, fallback={fallback_kind}"
    )
    if X is None:
        print("  No --test-data supplied: csynth is ready; csim data was not emitted.")
    return str(out_dir)


def compile_suite(suite, out_dir=None, only=None, test_data=None, gt_root='..'):
    """Generate every project in a suite. Returns the names written."""
    from ..fc.datasets import ARTIFACT_DIR
    specs = SUITES[suite]
    if only:
        specs = [s for s in specs if s.name == only]
        if not specs:
            raise ValueError(f"no project '{only}' in suite '{suite}'")
    out_dir = out_dir or os.path.join(ARTIFACT_DIR, 'generated', 'rules', suite)

    X, y = _load_test_data(test_data)

    written = []
    for i, spec in enumerate(specs):
        print(f"\n{'='*60}")
        print(f"  [{i+1}/{len(specs)}] {suite}/{spec.name}  "
              f"(fallback: {spec.fallback_kind}, {spec.phase1})")
        print(f"{'='*60}")
        model = RuleModel(_resolve(spec.rules_json, suite, gt_root))
        params = fb.load(spec.fallback_kind,
                         _resolve(spec.fallback_asset, suite, gt_root)
                         if spec.fallback_asset else None)
        profile = get_profile(spec.profile)
        base = os.path.join(out_dir, spec.name)

        _emit_project(base, model, spec, params, profile, X, y)
        print(f"  Done: {model.n_rules} rules, max {model.max_conds} conditions")
        written.append(spec.name)

    print(f"\nGenerated {len(written)} project(s) under {out_dir}")
    if X is None:
        print("  (no --test-data given, so data/ was not written; "
              "csim needs it, csynth does not)")
    return written
