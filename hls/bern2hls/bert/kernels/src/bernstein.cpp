#include "../include/bernstein.hpp"
#include "../include/activation_lut_rom.hpp"

// clamp -> scaled position -> floor/ceil reads -> linear interpolation
data_t bernstein_activate_lerp(data_t x_norm, const data_t lut[NEURON_LUT_SIZE])
{
#pragma HLS inline
    if (x_norm < data_t(0)) x_norm = data_t(0);
    if (x_norm > data_t(1)) x_norm = data_t(1);

    // wide idx_t: x_norm*(N-1) reaches 255, overflowing data_t's 8 integer bits
    idx_t pos = idx_t(x_norm) * idx_t(NEURON_LUT_SIZE - 1);

    unsigned int idx_lo = pos.to_uint();          // floor (pos >= 0)
    if (idx_lo >= NEURON_LUT_SIZE - 1) idx_lo = NEURON_LUT_SIZE - 2;
    unsigned int idx_hi = idx_lo + 1;

    data_t frac   = data_t(pos - idx_t(idx_lo));  // in [0,1]
    data_t val_lo = lut[idx_lo];
    data_t val_hi = lut[idx_hi];
    return val_lo + frac * (val_hi - val_lo);
}

void bernstein_layer(data_t out[HIDDEN_DIM], const data_t in[HIDDEN_DIM])
{
#pragma HLS inline off
BERN_LAYER:
    for (unsigned int n = 0; n < HIDDEN_DIM; n++) {
#pragma HLS pipeline II=1
        out[n] = bernstein_activate_lerp(in[n], NEURON_ACT_LUT[n]);
    }
}
