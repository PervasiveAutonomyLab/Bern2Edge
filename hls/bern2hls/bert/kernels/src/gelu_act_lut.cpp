#include "../include/gelu_act.hpp"
#include "../include/activation_lut_rom.hpp"

// clamp to [LO,HI] -> affine map to [0,N-1] -> floor/ceil reads -> lerp
data_t gelu_activate_lut(data_t x, const data_t lut[GELU_LUT_SIZE])
{
#pragma HLS inline
    const data_t lo = data_t(GELU_CLAMP_LO);
    const data_t hi = data_t(GELU_CLAMP_HI);
    if (x < lo) x = lo;
    if (x > hi) x = hi;

    // pos in [0, N-1]: (x - lo) / (hi - lo) * (N-1). Reaches 511 -> needs idx_t
    // (overflows data_t's 8 integer bits).
    const idx_t inv_span_scale = idx_t((GELU_LUT_SIZE - 1) / (GELU_CLAMP_HI - GELU_CLAMP_LO));
    idx_t pos = idx_t(x - lo) * inv_span_scale;

    unsigned int idx_lo = pos.to_uint();          // floor (pos >= 0)
    if (idx_lo >= GELU_LUT_SIZE - 1) idx_lo = GELU_LUT_SIZE - 2;
    unsigned int idx_hi = idx_lo + 1;

    data_t frac   = data_t(pos - idx_t(idx_lo));  // in [0,1]
    data_t val_lo = lut[idx_lo];
    data_t val_hi = lut[idx_hi];
    return val_lo + frac * (val_hi - val_lo);
}

void gelu_layer(data_t out[HIDDEN_DIM], const data_t in[HIDDEN_DIM])
{
#pragma HLS inline off
GELU_LAYER:
    for (unsigned int n = 0; n < HIDDEN_DIM; n++) {
#pragma HLS pipeline II=1
        out[n] = gelu_activate_lut(in[n], GELU_LUT);
    }
}
