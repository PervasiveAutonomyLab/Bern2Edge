#include "../include/layernorm.hpp"
#include "hls_math.h"

// LayerNorm over the 312 feature axis: biased variance (/D), eps=1e-12.
// Statistics are accumulated in float: sum of z^2 over 312 elements (z up to ~+-100
// after the FFN residual) reaches millions, which overflows any narrow ap_fixed.
// out[i] = gamma[i]*(x[i]-mean)/sqrt(var+eps) + beta[i]
void layernorm(data_t out[D_MODEL], const wide_t x[D_MODEL],
               const data_t gamma[D_MODEL], const data_t beta[D_MODEL])
{
#pragma HLS inline off
    const float inv_d = 1.0f / D_MODEL;

    float sum = 0.0f, sumsq = 0.0f;
LN_STATS:
    for (unsigned int i = 0; i < D_MODEL; i++) {
#pragma HLS pipeline II=1
        float v = (float)x[i];
        sum += v;
        sumsq += v * v;
    }
    float mean = sum * inv_d;
    float var  = sumsq * inv_d - mean * mean + LN_EPS;
    float rstd = 1.0f / hls::sqrt(var);

LN_APPLY:
    for (unsigned int i = 0; i < D_MODEL; i++) {
#pragma HLS pipeline II=1
        float xn = ((float)x[i] - mean) * rstd;
        out[i] = data_t((float)gamma[i] * xn + (float)beta[i]);
    }
}
