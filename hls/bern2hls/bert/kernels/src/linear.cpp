#include "../include/linear.hpp"

// ------------------------------------------------------------
// Load the 312-d input token vector from AXI blocks into a flat buffer.
// ------------------------------------------------------------
void load_input(data_t dst[INPUT_DIM], const axi_block_t src[INPUT_BLOCKS])
{
#pragma HLS inline off
LOAD_IN:
    for (unsigned int b = 0; b < INPUT_BLOCKS; b++) {
#pragma HLS pipeline II=1
        axi_block_t blk = src[b];
        for (unsigned int i = 0; i < AXI_BLOCK_SIZE; i++) {
#pragma HLS unroll
            unsigned int idx = b * AXI_BLOCK_SIZE + i;
            if (idx < INPUT_DIM) dst[idx] = blk[i];
        }
    }
}

// ------------------------------------------------------------
// fc1 (outer-product). Streams weights once in padded col_vec(32) order.
// All HIDDEN accumulators stay live; the 32-wide inner update is fully
// unrolled (32 parallel MACs) across cyclic-partitioned banks.
// ------------------------------------------------------------
void fc1_stream(data_t out[HIDDEN_DIM],
                const data_t in[INPUT_DIM],
                const axi_block_t fc1_w[FC1_WEIGHT_AXI_BLOCKS],
                const data_t bias[HIDDEN_DIM])
{
#pragma HLS inline off

    static acc_t acc[FC1_PADDED_HIDDEN];
#pragma HLS array_partition variable=acc cyclic factor=COL_VEC_SIZE

INIT_ACC:
    for (unsigned int n = 0; n < FC1_PADDED_HIDDEN; n++) {
#pragma HLS pipeline II=1
        acc[n] = (n < HIDDEN_DIM) ? acc_t(bias[n]) : acc_t(0);
    }

FC1_J:
    for (unsigned int j = 0; j < INPUT_DIM; j++) {
        data_t x = in[j];
    FC1_CV:
        for (unsigned int cv = 0; cv < FC1_COL_VECS_PER_COL; cv++) {
#pragma HLS pipeline II=4
            unsigned int base = (j * FC1_COL_VECS_PER_COL + cv) * AXI_PER_COL_VEC;
            data_t w[COL_VEC_SIZE];
#pragma HLS array_partition variable=w complete
            for (unsigned int a = 0; a < AXI_PER_COL_VEC; a++) {
                axi_block_t blk = fc1_w[base + a];
                for (unsigned int i = 0; i < AXI_BLOCK_SIZE; i++) {
#pragma HLS unroll
                    w[a * AXI_BLOCK_SIZE + i] = blk[i];
                }
            }
            for (unsigned int i = 0; i < COL_VEC_SIZE; i++) {
#pragma HLS unroll
                acc[cv * COL_VEC_SIZE + i] += x * w[i];
            }
        }
    }

WRITE_OUT:
    for (unsigned int n = 0; n < HIDDEN_DIM; n++) {
#pragma HLS pipeline II=1
        out[n] = data_t(acc[n]);
    }
}

// ------------------------------------------------------------
// fc2 (inner-product). Streams each weight row once; 8-wide MAC per beat.
// Output widened to wide_t (fc2 outputs exceed data_t's +-128 range).
// ------------------------------------------------------------
void fc2_stream(wide_t out[OUTPUT_DIM],
                const data_t act[HIDDEN_DIM],
                const axi_block_t fc2_w[FC2_WEIGHT_BLOCKS],
                const data_t bias[OUTPUT_DIM])
{
#pragma HLS inline off

FC2_O:
    for (unsigned int o = 0; o < OUTPUT_DIM; o++) {
        acc_t acc = acc_t(bias[o]);
        unsigned int row = o * HIDDEN_AXI_BLOCKS;
    FC2_AB:
        for (unsigned int ab = 0; ab < HIDDEN_AXI_BLOCKS; ab++) {
#pragma HLS pipeline II=1
            axi_block_t blk = fc2_w[row + ab];
            for (unsigned int i = 0; i < AXI_BLOCK_SIZE; i++) {
#pragma HLS unroll
                acc += act[ab * AXI_BLOCK_SIZE + i] * blk[i];
            }
        }
        out[o] = wide_t(acc);
    }
}
