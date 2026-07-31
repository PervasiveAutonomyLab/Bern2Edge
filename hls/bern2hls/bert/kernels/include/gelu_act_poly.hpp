#ifndef GELU_ACT_HPP
#define GELU_ACT_HPP

#include "types.hpp"
#include "config.hpp"

// ============================================================
// Direct GeLU: gelu(z) = 0.5*z*(1 + erf(z/sqrt(2))), exact erf form.
// erf is evaluated with the Abramowitz & Stegun 7.1.26 approximation
// (max error ~1.5e-7) in float — a synthesizable transcendental that exposes
// the exp/divide cost the Bernstein clamp+LUT replaces.
// ============================================================

data_t gelu_activate_poly(data_t x);

void gelu_layer(data_t out[HIDDEN_DIM], const data_t in[HIDDEN_DIM]);

#endif // GELU_ACT_HPP
