#ifndef LAYERNORM_HPP
#define LAYERNORM_HPP

#include "types.hpp"
#include "config.hpp"

// LayerNorm over the 312 feature axis: biased variance (/D), eps=1e-12.
// out[i] = gamma[i]*(x[i]-mean)/sqrt(var+eps) + beta[i]
// Input x is wide_t: the LN2 residual (x_ffn + fc2_out) reaches ~+-430 in h312, exceeding
// data_t's +-128. wide_t shares data_t's 24 fractional bits (lossless for the data_t residual)
// and adds integer headroom; stats are computed in float regardless.
void layernorm(data_t out[D_MODEL], const wide_t x[D_MODEL],
               const data_t gamma[D_MODEL], const data_t beta[D_MODEL]);

#endif // LAYERNORM_HPP
