#ifndef FFN_MODEL_HPP
#define FFN_MODEL_HPP

#include "types.hpp"
#include "config.hpp"
#include "linear.hpp"
#include "bernstein.hpp"

// Bernstein FFN sub-block (core): fc1(312->600) -> bernstein -> fc2(600->312).
// Weights streamed from DRAM; activations + per-channel LUT on-chip.
void ffn_top(wide_t output[OUTPUT_DIM],
             const axi_block_t input[INPUT_BLOCKS],
             const axi_block_t fc1_weights[FC1_WEIGHT_AXI_BLOCKS],
             const axi_block_t fc2_weights[FC2_WEIGHT_BLOCKS]);

#endif // FFN_MODEL_HPP
