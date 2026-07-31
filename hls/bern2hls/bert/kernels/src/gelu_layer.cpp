#include "../include/layer_model.hpp"
#include "../include/bias_rom.hpp"
#include "../include/ln_params_rom.hpp"

// ============================================================
// GeLU full encoder layer top.
//   attention_sublayer -> per-token [fc1 -> gelu -> fc2 -> +resid -> LN2]
// Identical to the Bernstein layer except the FFN activation (gelu_layer) and
// the FFN dims (HIDDEN_DIM=1200, plain fc1 bias).
// ============================================================

static data_t x_buf[SEQ_LEN][D_MODEL];
static data_t xffn_buf[SEQ_LEN][D_MODEL];
static data_t hidden_buf[HIDDEN_DIM];
static data_t act_buf[HIDDEN_DIM];

void layer_top(data_t output[SEQ_LEN * OUTPUT_DIM],
               const axi_block_t input[SEQ_LEN * INPUT_BLOCKS],
               const axi_block_t qkvo_weights[QKVO_AXI_BLOCKS],
               const axi_block_t fc1_weights[FC1_WEIGHT_AXI_BLOCKS],
               const axi_block_t fc2_weights[FC2_WEIGHT_BLOCKS])
{
#pragma HLS INTERFACE m_axi port=output        offset=slave bundle=gmem0 depth=SEQ_LEN*OUTPUT_DIM
#pragma HLS INTERFACE m_axi port=input         offset=slave bundle=gmem1 depth=SEQ_LEN*INPUT_BLOCKS
#pragma HLS INTERFACE m_axi port=qkvo_weights  offset=slave bundle=gmem2 depth=QKVO_AXI_BLOCKS
#pragma HLS INTERFACE m_axi port=fc1_weights   offset=slave bundle=gmem3 depth=FC1_WEIGHT_AXI_BLOCKS
#pragma HLS INTERFACE m_axi port=fc2_weights   offset=slave bundle=gmem4 depth=FC2_WEIGHT_BLOCKS
#pragma HLS INTERFACE s_axilite port=return bundle=control
#pragma HLS array_partition variable=x_buf cyclic factor=4 dim=1
#pragma HLS array_partition variable=x_buf cyclic factor=8 dim=2
#pragma HLS array_partition variable=act_buf cyclic factor=AXI_BLOCK_SIZE
#pragma HLS bind_storage variable=x_buf    type=ram_2p impl=lutram
#pragma HLS bind_storage variable=xffn_buf type=ram_2p impl=lutram

LOAD_TOK:
    for (unsigned int t = 0; t < SEQ_LEN; t++) {
        load_input(x_buf[t], input + t * INPUT_BLOCKS);
    }

    attention_sublayer(xffn_buf, x_buf, qkvo_weights);

FFN_TOK:
    for (unsigned int t = 0; t < SEQ_LEN; t++) {
        fc1_stream(hidden_buf, xffn_buf[t], fc1_weights, FC1_BIAS_ROM);
        gelu_layer(act_buf, hidden_buf);
        wide_t y[OUTPUT_DIM];
        fc2_stream(y, act_buf, fc2_weights, FC2_BIAS_ROM);

        // LN2 residual in wide_t: y (fc2 output) exceeds data_t's +-128 in the h312 model;
        // a data_t z2 would saturate and corrupt the LayerNorm variance for high-output tokens.
        wide_t z2[D_MODEL];
        for (unsigned int j = 0; j < D_MODEL; j++) {
#pragma HLS pipeline II=1
            z2[j] = wide_t(xffn_buf[t][j]) + y[j];
        }
        data_t out_row[OUTPUT_DIM];
        layernorm(out_row, z2, LN2_GAMMA_ROM, LN2_BETA_ROM);

        for (unsigned int j = 0; j < OUTPUT_DIM; j++) {
#pragma HLS pipeline II=1
            output[t * OUTPUT_DIM + j] = out_row[j];
        }
    }
}
