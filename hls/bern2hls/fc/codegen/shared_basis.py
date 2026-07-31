"""Shared-basis factorization of the per-neuron Bernstein activation LUT.

The per-neuron form stores NEURON_ACT_LUT[H][E] — one sampled activation
curve per hidden neuron. But the curves are all combinations of the *same*
neuron-independent Bernstein basis:

    NEURON_ACT_LUT[n][e] = sum_k coeffs[n][k] * B_k(t_e)

so the table factors into BASIS_LUT[E][deg+1] + NEURON_COEFF[H][deg+1],
shrinking the ROM from H*E to E*(deg+1) + H*(deg+1) entries — 3.03x smaller
at H=16, and it takes the activation ROM off BRAM entirely on a BRAM-starved
part like the XC7S15.

Two details are load-bearing for reproducing the shipped ROMs byte-for-byte:

* The grid is the torch 2.x `linspace(0,1,50)`, NOT the torch 1.x grid the
  adult per-neuron LUTs were generated under — the shared-basis ROMs were
  produced later, under a newer torch. One dataset, two grids.
* The basis must be evaluated through `BernsteinLayer.bern_basis`, whose
  binomial goes via `lgamma`. Re-deriving it with `math.comb` in float32 does
  not reproduce the shipped values even on the identical grid (30 of 200
  cells differ in the 10-decimal rendering).

`lut_grid(50, 'torch2')` is bit-identical to a live `torch.linspace(0,1,50)`
on torch 2.x, so using the embedded grid keeps the output independent of the
installed torch version — the same trick `bern_math` uses for the per-neuron
LUTs.
"""

import numpy as np

from ..bern_math import lut_grid

# The factorization is exact in real arithmetic; this bounds the float32
# round-trip. The shipped ROMs land ~1e-7, comfortably inside.
RECON_TOL = 5e-6


def factor_shared_basis(bern_layer, num_entries, grid_variant='torch2'):
    """Return (basis, coeffs) with shapes (E, deg+1) and (H, deg+1)."""
    x = lut_grid(num_entries, grid_variant)
    basis = bern_layer.bern_basis(x).detach().numpy()
    coeffs = bern_layer.bern_coeffs.data.detach().numpy()
    return basis, coeffs


def check_reconstruction(basis, coeffs, reference_lut, tol=RECON_TOL):
    """Assert the factorization reproduces the per-neuron LUT it replaces.

    Guards against a silent grid or evaluation-path mismatch: if the basis
    were built on the wrong grid the product would still be a plausible set of
    curves, just not the ones the design was trained and verified against.
    """
    err = float(np.abs(coeffs @ basis.T - reference_lut).max())
    if err > tol:
        raise ValueError(
            f'shared-basis factorization does not reproduce the per-neuron LUT: '
            f'max|recon - lut| = {err:.3e} > {tol:.1e}. The sampling grid or the '
            f'basis evaluation path is wrong (see this module docstring).')
    return err


def _fmt_2d(name, arr):
    rows, cols = arr.shape
    out = [f'static const data_t {name}[{rows}][{cols}] = {{']
    for r in range(rows):
        row = ', '.join(f'{v:.10f}' for v in arr[r])
        out.append(f'    {{{row}}}{"," if r < rows - 1 else ""}')
    out.append('};\n')
    return '\n'.join(out)


def gen_shared_basis_rom(basis, coeffs, degree, num_entries):
    """Emit shared_basis_rom.hpp."""
    guard = 'SHARED_BASIS_ROM_HPP'
    hidden = coeffs.shape[0]
    body = (f'// Shared Bernstein basis (degree {degree}), sampled on [0,1] with '
            f'{num_entries} entries.\n'
            f'// BASIS_LUT[e][k] = C(deg,k) t^k (1-t)^(deg-k), t = e/(E-1). '
            f'Neuron-independent.\n')
    body += f'constexpr unsigned int BERN_DEG = {degree};\n\n'
    body += _fmt_2d('BASIS_LUT', basis)
    body += f'\n// Per-neuron Bernstein coefficients ({hidden} x {degree + 1}).\n'
    body += _fmt_2d('NEURON_COEFF', coeffs)
    return (f'#ifndef {guard}\n#define {guard}\n\n'
            f'#include "types.hpp"\n#include "config.hpp"\n\n'
            f'{body}\n#endif // {guard}\n')


def gen_activate_shared(profile):
    """The `activate_shared()` kernel that replaces `activate_lerp()`.

    lerp is linear in the table values, so lerping the basis and then combining
    gives the same result as lerping the pre-combined curve — the two forms are
    mathematically identical and differ only in fixed-point rounding order.
    """
    bind = profile.act_bind_op
    L = []
    L.append('// Shared-basis Bernstein activation: activation_n(x) = sum_k coeff[n][k] * B_k(x).')
    L.append('// lerp is linear in table values, so lerp(sum_k c_k B_k) == sum_k c_k lerp(B_k):')
    L.append('// mathematically identical to the per-neuron LUT; only fixed-point rounding')
    L.append('// order differs. ROM: H*E entries -> E*(deg+1) + H*(deg+1).')
    L.append('static data_t activate_shared(data_t x_norm, const data_t coeff[BERN_DEG + 1]) {')
    L.append('    #pragma HLS inline')
    L.append('    if (x_norm < data_t(0)) x_norm = data_t(0);')
    L.append('    if (x_norm > data_t(1)) x_norm = data_t(1);')
    L.append('    const data_t scale = data_t(NEURON_LUT_SIZE - 1);')
    L.append('    data_t pos = x_norm * scale;')
    L.append('    unsigned int idx_lo = (unsigned int)(pos);')
    L.append('    if (idx_lo >= NEURON_LUT_SIZE - 1) idx_lo = NEURON_LUT_SIZE - 2;')
    L.append('    unsigned int idx_hi = idx_lo + 1;')
    L.append('    data_t frac = pos - data_t(idx_lo);')
    L.append('    acc_t acc = 0;')
    L.append('    for (unsigned int k = 0; k <= BERN_DEG; k++) {')
    L.append('        #pragma HLS unroll')
    L.append('        data_t b_lo = BASIS_LUT[idx_lo][k];')
    L.append('        data_t b_hi = BASIS_LUT[idx_hi][k];')
    L.append('        data_t db = b_hi - b_lo;')
    if bind:
        L.append('        // small 16-bit-class multiplies: keep them out of the scarce DSP48E1s')
    L.append('        data_t lerp_inc;')
    if bind:
        L.append(f'        #pragma HLS BIND_OP variable=lerp_inc op=mul impl={bind}')
    L.append('        lerp_inc = frac * db;')
    L.append('        data_t b = b_lo + lerp_inc;')
    L.append(f'        {profile.prod_type} p;  // exact data_t x data_t product, no acc_t widening')
    if bind:
        L.append(f'        #pragma HLS BIND_OP variable=p op=mul impl={bind}')
    L.append('        p = coeff[k] * b;')
    L.append('        acc += p;')
    L.append('    }')
    L.append('    return data_t(acc);')
    L.append('}')
    return L
