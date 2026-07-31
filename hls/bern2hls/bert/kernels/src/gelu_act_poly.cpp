#include "../include/gelu_act.hpp"
#include "hls_math.h"

// erf via Abramowitz & Stegun 7.1.26 (max abs error ~1.5e-7).
//   erf(x) = sign(x) * (1 - (a1*t + a2*t^2 + ... + a5*t^5) * exp(-x^2)),
//   t = 1 / (1 + p*|x|)
static inline float erf_as(float x)
{
#pragma HLS inline
    const float p  = 0.3275911f;
    const float a1 = 0.254829592f, a2 = -0.284496736f, a3 = 1.421413741f;
    const float a4 = -1.453152027f, a5 = 1.061405429f;
    float sign = (x < 0.0f) ? -1.0f : 1.0f;
    float ax = hls::fabsf(x);
    float t = 1.0f / (1.0f + p * ax);
    float poly = ((((a5 * t + a4) * t + a3) * t + a2) * t + a1) * t;
    float y = 1.0f - poly * hls::expf(-ax * ax);
    return sign * y;
}

// gelu(z) = 0.5*z*(1 + erf(z/sqrt(2)))
data_t gelu_activate_poly(data_t x)
{
#pragma HLS inline
    float z = (float)x;
    float g = 0.5f * z * (1.0f + erf_as(z * 0.70710678118654752f));
    return data_t(g);
}

void gelu_layer(data_t out[HIDDEN_DIM], const data_t in[HIDDEN_DIM])
{
#pragma HLS inline off
GELU_LAYER:
    for (unsigned int n = 0; n < HIDDEN_DIM; n++) {
#pragma HLS pipeline II=1
        out[n] = gelu_activate_poly(in[n]);
    }
}
