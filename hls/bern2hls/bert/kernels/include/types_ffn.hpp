#ifndef TYPES_HPP
#define TYPES_HPP

#include <ap_fixed.h>
#include <hls_vector.h>
#include "config.hpp"

// ============================================================
// Fixed-point types
// ============================================================
// data_t : activations / weights / fc1 output / activation output.
//          ap_fixed<32,8> -> 8 integer bits (+-128), 24 fractional bits.
// acc_t  : MAC accumulator, wide integer headroom.
// wide_t : fc2 OUTPUT storage. fc2 outputs reach ~+-200 (bern) / ~+-2151 (gelu);
//          this exceeds data_t's +-128, so the final output uses a wider type.
//          Used identically in both models so it is not a thumb on the scale.
typedef ap_fixed<FIXED_TOTAL_BITS, FIXED_INT_BITS> data_t;
typedef ap_fixed<48, 16> acc_t;
typedef ap_fixed<40, 16> wide_t;

// LUT position/index type: holds scaled positions up to the LUT size (256/512),
// which exceed data_t's 8 integer bits. 16 integer + 24 fractional bits.
typedef ap_fixed<40, 16> idx_t;

// AXI block: 8 x 32-bit = 256-bit transfer
typedef hls::vector<data_t, AXI_BLOCK_SIZE> axi_block_t;

#endif // TYPES_HPP
