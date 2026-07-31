#include "../include/attention.hpp"
#include "../include/softmax.hpp"
#include "../include/layernorm.hpp"
#include "../include/attn_bias_rom.hpp"
#include "../include/ln_params_rom.hpp"

// ============================================================
// Attention with MODERATE parallelism so it does not dominate the layer area.
// Knobs (kept small on purpose):
//   TOKEN_PAR : tokens accumulated in parallel in the projections   (4, was 16)
//   DOT_PAR   : head-dim lanes in the Q.K^T dot product             (2, was 26)
//   CTX_PAR   : key lanes in the context (prob.V) reduction         (TOKEN_PAR)
// Buffers are partitioned cyclically (dim1 by TOKEN_PAR, dim2 by 8) instead of
// a 16x26 = 416-way split — the previous over-partitioning is what blew up FF/LUT.
// Latency rises (attention is still a minority of the layer) but DSP/FF/LUT drop
// sharply, giving a more balanced per-module resource allocation.
// ============================================================
#define TOKEN_PAR 4
#define DOT_PAR   2

static data_t Q[SEQ_LEN][D_MODEL];
static data_t K[SEQ_LEN][D_MODEL];
static data_t V[SEQ_LEN][D_MODEL];
static data_t AO[SEQ_LEN][D_MODEL];
static data_t CTX[SEQ_LEN][D_MODEL];

constexpr unsigned int PROJ_ROW_BLOCKS = D_MODEL / AXI_BLOCK_SIZE;   // 39

// out[t][o] = sum_i in[t][i]*W[o][i] + bias[o].  Tokens processed in groups of
// TOKEN_PAR (weights re-read per group); within a group, TOKEN_PAR*8 MACs/cycle.
static void proj_stream(data_t out[SEQ_LEN][D_MODEL],
                        const data_t in[SEQ_LEN][D_MODEL],
                        const axi_block_t w[QKVO_AXI_BLOCKS],
                        unsigned int blk_off,
                        const data_t bias[D_MODEL])
{
#pragma HLS inline off
PROJ_TG:
    for (unsigned int tg = 0; tg < SEQ_LEN; tg += TOKEN_PAR) {
    PROJ_O:
        for (unsigned int o = 0; o < D_MODEL; o++) {
            acc_t acc[TOKEN_PAR];
#pragma HLS array_partition variable=acc complete
            for (unsigned int tp = 0; tp < TOKEN_PAR; tp++) {
#pragma HLS unroll
                acc[tp] = acc_t(bias[o]);
            }
            unsigned int row = blk_off + o * PROJ_ROW_BLOCKS;
        PROJ_AB:
            for (unsigned int ab = 0; ab < PROJ_ROW_BLOCKS; ab++) {
#pragma HLS pipeline II=1
                axi_block_t wb = w[row + ab];
                for (unsigned int i = 0; i < AXI_BLOCK_SIZE; i++) {
#pragma HLS unroll
                    data_t wv = wb[i];
                    unsigned int ii = ab * AXI_BLOCK_SIZE + i;
                    for (unsigned int tp = 0; tp < TOKEN_PAR; tp++) {
#pragma HLS unroll
                        acc[tp] += in[tg + tp][ii] * wv;
                    }
                }
            }
            for (unsigned int tp = 0; tp < TOKEN_PAR; tp++) {
#pragma HLS unroll
                out[tg + tp][o] = data_t(acc[tp]);
            }
        }
    }
}

void attention_sublayer(data_t xffn[SEQ_LEN][D_MODEL],
                        const data_t x[SEQ_LEN][D_MODEL],
                        const axi_block_t qkvo_w[QKVO_AXI_BLOCKS])
{
#pragma HLS inline off
    // moderate cyclic partitioning (function-scope pragmas on file-scope arrays)
#pragma HLS array_partition variable=Q   cyclic factor=TOKEN_PAR dim=1
#pragma HLS array_partition variable=Q   cyclic factor=8 dim=2
#pragma HLS array_partition variable=K   cyclic factor=TOKEN_PAR dim=1
#pragma HLS array_partition variable=K   cyclic factor=8 dim=2
#pragma HLS array_partition variable=V   cyclic factor=TOKEN_PAR dim=1
#pragma HLS array_partition variable=V   cyclic factor=8 dim=2
#pragma HLS array_partition variable=AO  cyclic factor=TOKEN_PAR dim=1
#pragma HLS array_partition variable=AO  cyclic factor=8 dim=2
#pragma HLS array_partition variable=CTX cyclic factor=TOKEN_PAR dim=1
#pragma HLS array_partition variable=CTX cyclic factor=8 dim=2
    // These are small scratch buffers — keep them in distributed LUT-RAM, not BRAM
    // (cyclic partitioning would otherwise over-bank them into many underfilled BRAMs).
#pragma HLS bind_storage variable=Q   type=ram_2p impl=lutram
#pragma HLS bind_storage variable=K   type=ram_2p impl=lutram
#pragma HLS bind_storage variable=V   type=ram_2p impl=lutram
#pragma HLS bind_storage variable=AO  type=ram_2p impl=lutram
#pragma HLS bind_storage variable=CTX type=ram_2p impl=lutram

    // 1. Q/K/V projections
    proj_stream(Q, x, qkvo_w, 0 * PROJ_AXI_BLOCKS, BQ_ROM);
    proj_stream(K, x, qkvo_w, 1 * PROJ_AXI_BLOCKS, BK_ROM);
    proj_stream(V, x, qkvo_w, 2 * PROJ_AXI_BLOCKS, BV_ROM);

    // 2. per head, per query: scaled dot-product -> softmax -> context
HEAD:
    for (unsigned int h = 0; h < NUM_HEADS; h++) {
        unsigned int base = h * HEAD_DIM;
    QUERY:
        for (unsigned int qi = 0; qi < SEQ_LEN; qi++) {
            score_t row[SEQ_LEN];
#pragma HLS array_partition variable=row complete
        KEY:
            for (unsigned int ki = 0; ki < SEQ_LEN; ki++) {
#pragma HLS pipeline II=1
                acc_t s = 0;
                for (unsigned int d = 0; d < HEAD_DIM; d++) {
#pragma HLS unroll factor=DOT_PAR
                    s += Q[qi][base + d] * K[ki][base + d];
                }
                row[ki] = score_t(s) * score_t(ATTN_SCALE);
            }
            data_t prob[SEQ_LEN];
#pragma HLS array_partition variable=prob cyclic factor=TOKEN_PAR
            softmax_row(prob, row);
        CTXD:
            for (unsigned int d = 0; d < HEAD_DIM; d++) {
#pragma HLS pipeline II=1
                acc_t c = 0;
                for (unsigned int ki = 0; ki < SEQ_LEN; ki++) {
#pragma HLS unroll factor=TOKEN_PAR
                    c += prob[ki] * V[ki][base + d];
                }
                CTX[qi][base + d] = data_t(c);
            }
        }
    }

    // 3. output projection Wo
    proj_stream(AO, CTX, qkvo_w, 3 * PROJ_AXI_BLOCKS, BO_ROM);

    // 4. residual + LayerNorm1
RESID_LN1:
    for (unsigned int t = 0; t < SEQ_LEN; t++) {
        // z1 in wide_t to match layernorm's widened input (LN1 residual is small, ~+-22,
        // but the type must agree; wide_t is lossless from data_t).
        wide_t z1[D_MODEL];
        for (unsigned int j = 0; j < D_MODEL; j++) {
#pragma HLS pipeline II=1
            z1[j] = wide_t(x[t][j]) + wide_t(AO[t][j]);
        }
        layernorm(xffn[t], z1, LN1_GAMMA_ROM, LN1_BETA_ROM);
    }
}
