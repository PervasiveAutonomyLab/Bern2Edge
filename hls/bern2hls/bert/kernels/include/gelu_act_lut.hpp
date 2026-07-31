#ifndef GELU_ACT_HPP
#define GELU_ACT_HPP

#include "types.hpp"
#include "config.hpp"

// ============================================================
// GeLU activation via a SHARED LUT + linear interpolation.
// Same lerp datapath as the Bernstein variant, but ONE table for all channels
// (GeLU is a single 1-D function). Clamp to [GELU_CLAMP_LO, GELU_CLAMP_HI],
// affine-map to the table index, read two neighbours, interpolate.
// ============================================================

data_t gelu_activate_lut(data_t x, const data_t lut[GELU_LUT_SIZE]);

void gelu_layer(data_t out[HIDDEN_DIM], const data_t in[HIDDEN_DIM]);

#endif // GELU_ACT_HPP
