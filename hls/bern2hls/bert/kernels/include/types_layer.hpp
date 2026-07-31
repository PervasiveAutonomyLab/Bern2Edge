#ifndef TYPES_HPP
#define TYPES_HPP

#include <ap_fixed.h>
#include <hls_vector.h>
#include "config.hpp"

// ---- FFN/linear types (reused from the FFN experiment) ----
typedef ap_fixed<FIXED_TOTAL_BITS, FIXED_INT_BITS> data_t;   // activations/weights
typedef ap_fixed<48, 16> acc_t;                              // MAC accumulator
typedef ap_fixed<40, 16> wide_t;                             // fc2 output (exceeds +-128)
typedef ap_fixed<40, 16> idx_t;                              // LUT position (256/512)
typedef hls::vector<data_t, AXI_BLOCK_SIZE> axi_block_t;     // 256-bit transfer

// ---- Attention / LayerNorm / softmax types ----
typedef ap_fixed<32, 12> score_t;    // pre-softmax scores reach ~+-160 -> 12 int bits
typedef ap_fixed<32, 10> ln_t;       // LayerNorm mean/var/rstd (var <= ~30)
typedef ap_fixed<24, 2>  exp_t;      // softmax exp output in (0,1]
typedef ap_fixed<32, 8>  sm_acc_t;   // softmax sum over <= SEQ_LEN terms

#endif // TYPES_HPP
