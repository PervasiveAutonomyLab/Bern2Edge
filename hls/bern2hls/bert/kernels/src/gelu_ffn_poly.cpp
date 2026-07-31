#include "../include/ffn_model.hpp"
#include "../include/bias_rom.hpp"

// ============================================================
// GeLU baseline FFN sub-block top (core: fc1 -> gelu(direct erf) -> fc2).
//   fc1: 312 -> 1200, plain weights/bias
//   act: GeLU evaluated directly (exact erf form)
//   fc2: 1200 -> 312, wide_t output
// Weights streamed from DRAM; no activation LUT.
// ============================================================

static data_t input_buf[INPUT_DIM];
static data_t hidden_buf[HIDDEN_DIM];
static data_t act_buf[HIDDEN_DIM];

void ffn_top(wide_t output[OUTPUT_DIM],
             const axi_block_t input[INPUT_BLOCKS],
             const axi_block_t fc1_weights[FC1_WEIGHT_AXI_BLOCKS],
             const axi_block_t fc2_weights[FC2_WEIGHT_BLOCKS])
{
#pragma HLS INTERFACE m_axi port=output       offset=slave bundle=gmem0 depth=OUTPUT_DIM
#pragma HLS INTERFACE m_axi port=input        offset=slave bundle=gmem1 depth=INPUT_BLOCKS
#pragma HLS INTERFACE m_axi port=fc1_weights  offset=slave bundle=gmem2 depth=FC1_WEIGHT_AXI_BLOCKS
#pragma HLS INTERFACE m_axi port=fc2_weights  offset=slave bundle=gmem3 depth=FC2_WEIGHT_BLOCKS
#pragma HLS INTERFACE s_axilite port=return bundle=control

#pragma HLS array_partition variable=act_buf cyclic factor=AXI_BLOCK_SIZE

    load_input(input_buf, input);
    fc1_stream(hidden_buf, input_buf, fc1_weights, FC1_BIAS_ROM);
    gelu_layer(act_buf, hidden_buf);
    fc2_stream(output, act_buf, fc2_weights, FC2_BIAS_ROM);
}
