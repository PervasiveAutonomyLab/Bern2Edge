"""Weight loading and ROM emission for the transformer designs.

Reads the slim per-layer bundles produced by tools/slim_bert_weights.py, folds
the per-channel Bernstein input normalization into fc1 (so the hardware needs
no division), builds the activation tables, and writes the ROM headers.

The large matrices deliberately do NOT become headers: fc1/fc2 at 312x600 and
the four 312x312 attention projections exceed KV260 on-chip memory, so they are
written as text and streamed in over m_axi by the testbench. Only the small
per-channel parameters are compiled in.
"""

import os
from math import comb, erf

import numpy as np

BOUNDS_EPS = 1e-8          # matches the training-side normalization
GELU_CLAMP = (-8.0, 8.0)


def load_layer(npz_path):
    """Load one slim layer bundle. Keys are the tensor suffixes."""
    return {k: v.astype(np.float64) for k, v in np.load(npz_path).items()}


def fuse_bern(W1, b1, bounds):
    """Fold u = (z - lo)/(hi - lo) into fc1, so the kernel sees [0,1] directly."""
    lo, hi = bounds[:, 0], bounds[:, 1]
    inv_range = 1.0 / (hi - lo + BOUNDS_EPS)
    return W1 * inv_range[:, None], (b1 - lo) * inv_range, inv_range, lo


def bern_basis(u, degree):
    """(..., degree+1) Bernstein basis C(deg,k) u^k (1-u)^(deg-k)."""
    u = np.asarray(u)
    k = np.arange(degree + 1)
    nCk = np.array([comb(degree, int(kk)) for kk in k], dtype=np.float64)
    return nCk * (u[..., None] ** k) * ((1.0 - u[..., None]) ** (degree - k))


def build_bern_lut(coeffs, num_entries):
    """Per-channel activation table, sampled uniformly on [0,1]."""
    basis = bern_basis(np.linspace(0.0, 1.0, num_entries), coeffs.shape[1] - 1)
    return coeffs @ basis.T


def build_gelu_lut(num_entries, clamp=GELU_CLAMP):
    """One shared table over the clamp window, exact erf."""
    lo, hi = clamp
    z = np.linspace(lo, hi, num_entries)
    return np.array([0.5 * v * (1.0 + erf(v / np.sqrt(2.0))) for v in z])


# ---------------------------------------------------------------- ROM headers

def _guard(name, body):
    g = f'{name}_HPP'
    return (f'#ifndef {g}\n#define {g}\n\n'
            f'#include "types.hpp"\n#include "config.hpp"\n\n{body}\n#endif // {g}\n')


def gen_bias_rom(b1, b2, fused):
    tag = 'FUSED (b_fused[n]=(b[n]-lo[n])*inv_range[n])' if fused else 'plain'
    L = [f'// fc1 bias ROM ({tag})',
         'static const data_t FC1_BIAS_ROM[HIDDEN_DIM] = {']
    L += [f'    data_t({v:.10f}){"," if i < len(b1) - 1 else ""}'
          for i, v in enumerate(b1)]
    L += ['};', '', '// fc2 bias ROM',
          'static const data_t FC2_BIAS_ROM[OUTPUT_DIM] = {']
    L += [f'    data_t({v:.10f}){"," if i < len(b2) - 1 else ""}'
          for i, v in enumerate(b2)]
    L += ['};', '']
    return _guard('BIAS_ROM', '\n'.join(L))


def gen_bern_lut_rom(lut, num_entries, degree=15):
    H = lut.shape[0]
    out = [f'// Per-channel Bernstein activation LUT: {H} channels x '
           f'{num_entries} entries',
           f'// NEURON_ACT_LUT[n][i] = sum_k coeff[n,k]*C({degree},k)*x_i^k'
           f'*(1-x_i)^({degree}-k), x_i in [0,1]',
           'static const data_t NEURON_ACT_LUT[HIDDEN_DIM][NEURON_LUT_SIZE] = {']
    for n in range(H):
        row = f'    /* ch {n:4d} */ {{'
        for i in range(num_entries):
            if i and i % 10 == 0:
                row += '\n                   '
            row += f'{lut[n, i]:.10f}{", " if i < num_entries - 1 else ""}'
        out.append(row + f'}}{"," if n < H - 1 else ""}')
    out += ['};', '']
    return _guard('ACTIVATION_LUT_ROM', '\n'.join(out))


def gen_gelu_lut_rom(lut, num_entries, clamp=GELU_CLAMP):
    body = [f'// Shared GeLU activation LUT: {num_entries} entries over '
            f'[{clamp[0]},{clamp[1]}]',
            '// GELU_LUT[i] = gelu(lo + i*(hi-lo)/(N-1)), exact erf',
            'static const data_t GELU_LUT[GELU_LUT_SIZE] = {']
    line = ''
    for i in range(num_entries):
        if i % 8 == 0:
            line += '    '
        line += f'data_t({lut[i]:.10f}){", " if i < num_entries - 1 else ""}'
        if i % 8 == 7:
            body.append(line)
            line = ''
    body.append(line)
    body += ['};', '']
    return _guard('ACTIVATION_LUT_ROM', '\n'.join(body))


def gen_norm_rom(inv_range):
    L = ['// Normalization params (reference only; fused into fc1 weights/bias)',
         'static const data_t NEURON_INV_RANGE_ROM[HIDDEN_DIM] = {']
    L += [f'    data_t({v:.10f}){"," if i < len(inv_range) - 1 else ""}'
          for i, v in enumerate(inv_range)]
    L += ['};', '']
    return _guard('NORM_PARAMS_ROM', '\n'.join(L))


# ------------------------------------------------------------- weight streams

def ceildiv(a, b):
    return -(-a // b)


def pack_fc1_colvec(W, col_vec_size):
    """fc1 streams as column vectors of COL_VEC_SIZE outputs, zero-padded.

    The hidden layer uses an outer-product datapath: for each input element it
    broadcasts against a slab of outputs, so the weights must arrive grouped by
    input index, not by output row.
    """
    H, IN = W.shape
    n_vec = ceildiv(H, col_vec_size)
    out = np.zeros((IN, n_vec, col_vec_size), dtype=np.float64)
    for i in range(IN):
        for v in range(n_vec):
            lo = v * col_vec_size
            hi = min(lo + col_vec_size, H)
            out[i, v, :hi - lo] = W[lo:hi, i]
    return out


def write_flat(path, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        for v in np.asarray(data).flatten():
            f.write(f'{v:.10f}\n')
