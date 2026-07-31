#ifndef BERNSTEIN_HPP
#define BERNSTEIN_HPP

#include "types.hpp"
#include "config.hpp"

// ============================================================
// Per-channel degree-15 Bernstein activation, deployed as LUT + lerp.
// fc1 is fused with the per-channel normalisation offline, so the input here
// is already ~[0,1]. The degree-15 polynomial is baked into NEURON_ACT_LUT
// offline -> the hardware is a clamp + 2 ROM reads + 1 lerp, independent of
// the polynomial degree.
// ============================================================

data_t bernstein_activate_lerp(data_t x_norm, const data_t lut[NEURON_LUT_SIZE]);

void bernstein_layer(data_t out[HIDDEN_DIM], const data_t in[HIDDEN_DIM]);

#endif // BERNSTEIN_HPP
