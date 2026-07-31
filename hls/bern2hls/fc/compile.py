"""compile: .pth checkpoints -> complete Vitis HLS projects.

Two input modes:
  * curated (default): the DatasetSpec.models list — the exact checkpoints
    behind the shipped ground-truth projects.
  * --pth: arbitrary checkpoint files/dirs; each model's architecture,
    activation and Bernstein degree are extracted automatically from the
    checkpoint metadata (filename convention as fallback).

Test vectors come from the packaged artifact/test_vectors/<ds>/ .npy files
(the exact float64 arrays the original generators drew from the raw datasets).
"""

import glob
import os

import numpy as np

from .codegen import axi_family, rom_family
from .datasets import ARTIFACT_DIR, DATASET_SPECS
from .model_extract import check_cfg_against_metadata, make_cfg_from_pth
from ..core.emit import write_flat
from ..core.profiles import get_profile


def load_test_vectors(spec, profile=None):
    """Load the packaged vectors for this dataset, or the alternate set a
    profile asks for (the low-power accuracy testbench wants all 9045 adult
    samples, not the 20 the self-checking testbench uses)."""
    which = spec.name
    expect = spec.num_samples
    if profile is not None and profile.test_vector_set != 'default':
        which = profile.test_vector_set
        expect = None
    tv_dir = os.path.join(ARTIFACT_DIR, 'test_vectors', which)
    X = np.load(os.path.join(tv_dir, 'test_input.npy'))
    y = np.load(os.path.join(tv_dir, 'test_labels.npy'))
    if X.ndim != 2 or X.shape[1] != spec.input_dim:
        raise ValueError(f"Packaged test vectors '{which}' have shape {X.shape}, "
                         f"expected (N, {spec.input_dim})")
    if expect is not None and X.shape[0] != expect:
        raise ValueError(f"Packaged test vectors for {spec.name} have {X.shape[0]} "
                         f"samples, expected {expect}")
    return X, y


def resolve_pth_inputs(paths):
    """Expand --pth arguments (files or directories) into a list of .pth files."""
    files = []
    for p in paths:
        if os.path.isdir(p):
            files.extend(sorted(glob.glob(os.path.join(p, '*.pth'))))
        else:
            files.append(p)
    if not files:
        raise ValueError(f"No .pth files found under: {paths}")
    return files


def compile_dataset(dataset, out_dir=None, only=None, pth_inputs=None,
                    ckpt_dir=None, part=None, clock_ns=None, profile=None):
    """Generate HLS projects for one dataset. Returns list of generated names.

    `profile` selects a BackendProfile (name or object); `part`/`clock_ns`, if
    given, override that profile's tool settings.
    """
    profile = get_profile(profile)
    overrides = {}
    if part is not None:
        overrides['part'] = part
    if clock_ns is not None:
        overrides['clock_ns'] = clock_ns
    if overrides:
        profile = profile.derive(**overrides)
    spec = DATASET_SPECS[dataset]
    output_dir = out_dir or os.path.join(ARTIFACT_DIR, 'generated', spec.name)
    os.makedirs(output_dir, exist_ok=True)
    X_test, y_test = load_test_vectors(spec, profile)

    if pth_inputs:
        cfgs = [make_cfg_from_pth(p) for p in resolve_pth_inputs(pth_inputs)]
        for cfg in cfgs:
            if cfg['layer_sizes'][0] != spec.input_dim or cfg['layer_sizes'][-1] != spec.output_dim:
                raise ValueError(
                    f"{cfg['file']}: arch {cfg['layer_sizes']} does not match dataset "
                    f"{spec.name} ({spec.input_dim} -> {spec.output_dim})")
    else:
        cfgs = spec.models

    if only:
        cfgs = [c for c in cfgs if c['name'] == only]
        if not cfgs:
            raise ValueError(f"Model '{only}' not found for dataset {dataset}")

    family = rom_family if spec.family == 'rom' else axi_family
    generated = []
    for i, cfg in enumerate(cfgs):
        model_dir = cfg.get('_dir') or ckpt_dir or spec.default_ckpt_dir()
        print(f"\n{'='*60}")
        print(f"  [{i+1}/{len(cfgs)}] Generating: {cfg['name']}  ({spec.name}, {spec.family} family)")
        print(f"  Checkpoint dir: {model_dir}")
        print(f"  Profile: {profile.name}  ({profile.part} @ {profile.clock_ns} ns)")
        print(f"{'='*60}")
        check_cfg_against_metadata(cfg, os.path.join(model_dir, cfg['file']))
        family.generate_one_model(cfg, spec, model_dir, output_dir, X_test, y_test,
                                  profile=profile)
        generated.append(cfg['name'])

    # Profiles that share one data directory across a sweep (emit_data=False)
    # still need it to exist, or the emitted TCL's -tb paths dangle. The path is
    # resolved relative to <proj>/script/, which is Vitis' cwd.
    if generated and not profile.emit_data and profile.data_dir_mode.startswith('shared:'):
        rel = profile.data_dir_mode.split(':', 1)[1]
        shared = os.path.normpath(os.path.join(output_dir, generated[0], 'script', rel))
        os.makedirs(shared, exist_ok=True)
        write_flat(os.path.join(shared, 'test_input.txt'), X_test)
        with open(os.path.join(shared, 'test_labels.txt'), 'w') as f:
            for label in y_test:
                f.write(f'{label}\n')
        print(f"  Shared test data ({len(X_test)} samples) -> {shared}")

    print(f"\nGenerated {len(generated)} project(s) under {output_dir}")
    return generated
