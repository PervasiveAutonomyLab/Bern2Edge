"""Shared Bernstein math: normalization fusion, LUT building, NumPy golden reference.

Lifted verbatim from *-HW/sweep_test_all/generate_all_models.py (the three
copies are byte-identical). The reference deliberately mirrors the HLS
fixed-point LUT+lerp datapath so csim self-checks against it.
"""

import numpy as np
import torch

# Exact float32 bit patterns of torch.linspace(0, 1, 50) under the two torch
# generations that produced the shipped ground-truth projects. torch 2.0
# changed the linspace CPU kernel, shifting 8 of the 50 grid points by 1 ULP,
# which propagates into every Bernstein activation LUT (and from there into
# the golden reference outputs). The shipped adult and covertype projects were
# generated under torch 1.x, the shipped higgs projects under torch 2.x, so
# the grid variant is a per-dataset template property (DatasetSpec.lut_grid).
# Embedding both grids bit-exactly makes build_full_lut() reproduce the
# shipped LUT ROMs on any torch version. (Verified: bern_basis and the f32
# matmul are version-stable given an identical grid.)
_LUT_GRID_50_BITS = {
    'torch1': np.array([
        0x00000000, 0x3ca72f05, 0x3d272f05, 0x3d7ac688, 0x3da72f05, 0x3dd0fac6, 0x3dfac688, 0x3e124924, 0x3e272f05, 0x3e3c14e6,
        0x3e50fac6, 0x3e65e0a7, 0x3e7ac688, 0x3e87d634, 0x3e924924, 0x3e9cbc14, 0x3ea72f05, 0x3eb1a1f5, 0x3ebc14e6, 0x3ec687d6,
        0x3ed0fac6, 0x3edb6db6, 0x3ee5e0a7, 0x3ef05397, 0x3efac688, 0x3f029cbc, 0x3f07d634, 0x3f0d0fac, 0x3f124925, 0x3f17829d,
        0x3f1cbc15, 0x3f21f58d, 0x3f272f05, 0x3f2c687d, 0x3f31a1f5, 0x3f36db6e, 0x3f3c14e6, 0x3f414e5e, 0x3f4687d6, 0x3f4bc14e,
        0x3f50fac7, 0x3f56343f, 0x3f5b6db7, 0x3f60a730, 0x3f65e0a8, 0x3f6b1a20, 0x3f705398, 0x3f758d10, 0x3f7ac688, 0x3f800000,
    ], dtype=np.uint32),
    'torch2': np.array([
        0x00000000, 0x3ca72f05, 0x3d272f05, 0x3d7ac688, 0x3da72f05, 0x3dd0fac6, 0x3dfac688, 0x3e124924, 0x3e272f05, 0x3e3c14e6,
        0x3e50fac6, 0x3e65e0a7, 0x3e7ac688, 0x3e87d634, 0x3e924924, 0x3e9cbc15, 0x3ea72f05, 0x3eb1a1f5, 0x3ebc14e6, 0x3ec687d6,
        0x3ed0fac6, 0x3edb6db7, 0x3ee5e0a7, 0x3ef05397, 0x3efac688, 0x3f029cbc, 0x3f07d634, 0x3f0d0fad, 0x3f124925, 0x3f17829d,
        0x3f1cbc15, 0x3f21f58d, 0x3f272f05, 0x3f2c687e, 0x3f31a1f6, 0x3f36db6e, 0x3f3c14e6, 0x3f414e5e, 0x3f4687d6, 0x3f4bc14e,
        0x3f50fac7, 0x3f56343f, 0x3f5b6db7, 0x3f60a72f, 0x3f65e0a7, 0x3f6b1a1f, 0x3f705398, 0x3f758d10, 0x3f7ac688, 0x3f800000,
    ], dtype=np.uint32),
}


def lut_grid(num_entries, variant='torch1'):
    """Sampling grid for the activation LUT (bit-exact for the shipped size 50)."""
    if num_entries == 50:
        return torch.from_numpy(_LUT_GRID_50_BITS[variant].view(np.float32).copy())
    return torch.linspace(0, 1, num_entries)


def compute_fused(W, b, bounds):
    """Fuse normalization into weights/bias."""
    lower = bounds[:, 0]
    upper = bounds[:, 1]
    rng = upper - lower
    inv_range = 1.0 / (rng + 1e-8)
    W_fused = W * inv_range[:, np.newaxis]
    b_fused = (b - lower) * inv_range
    return W_fused, b_fused, inv_range, lower


def build_full_lut(bern_layer, num_entries, grid_variant='torch1'):
    """Build per-neuron LUT from a BernsteinLayer."""
    x = lut_grid(num_entries, grid_variant)
    basis = bern_layer.bern_basis(x)  # (E, deg+1)
    coeffs = bern_layer.bern_coeffs.data  # (H, deg+1)
    return (coeffs @ basis.T).detach().numpy()  # (H, E)


def ref_lut_lerp_layer(x_norm, lut, lut_size):
    """Apply per-neuron LUT + lerp activation (vectorized)."""
    x = np.clip(x_norm, 0, 1)
    scale = lut_size - 1
    pos = x * scale
    idx_lo = np.clip(np.floor(pos).astype(int), 0, lut_size - 2)
    idx_hi = idx_lo + 1
    frac = pos - idx_lo.astype(float)
    n_idx = np.arange(x.shape[0])
    return lut[n_idx, idx_lo] + frac * (lut[n_idx, idx_hi] - lut[n_idx, idx_lo])


def ref_inference(X, linears_data, luts, is_bern, lut_size):
    """Reference inference matching HLS computation.

    linears_data: [(W, b), ...] — already fused for bern layers
    luts: [lut_array_or_None, ...] for each hidden activation
    """
    N = X.shape[0]
    out_dim = linears_data[-1][0].shape[0]
    result = np.zeros((N, out_dim))

    for i in range(N):
        h = X[i].copy()
        num_linear = len(linears_data)
        for li in range(num_linear):
            W, b = linears_data[li]
            h = W @ h + b
            if li < num_linear - 1:
                if is_bern and luts[li] is not None:
                    h = ref_lut_lerp_layer(h, luts[li], lut_size)
                else:
                    h = np.maximum(h, 0)
        result[i] = h

    return result
