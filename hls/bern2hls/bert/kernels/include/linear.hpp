#ifndef LINEAR_HPP
#define LINEAR_HPP

#include "types.hpp"
#include "config.hpp"

// ============================================================
// Streaming linear layers (weights read directly from DRAM via m_axi).
// Transformer FFN weight matrices (600x312, 1200x312) are far too large to
// buffer fully on-chip on the KV260, so they are streamed and consumed once.
// ============================================================

// fc1: outer-product form. Weights streamed in padded column-major col_vec(32)
//      order: for j in INPUT: for cv in cvpc: 4 AXI beats (32 weights).
constexpr unsigned int COL_VEC_SIZE        = 32;
constexpr unsigned int AXI_PER_COL_VEC     = COL_VEC_SIZE / AXI_BLOCK_SIZE;        // 4
constexpr unsigned int FC1_COL_VECS_PER_COL= ceildiv(HIDDEN_DIM, COL_VEC_SIZE);    // 19 (600) / 38 (1200)
constexpr unsigned int FC1_PADDED_HIDDEN   = FC1_COL_VECS_PER_COL * COL_VEC_SIZE;  // 608 / 1216
constexpr unsigned int FC1_WEIGHT_AXI_BLOCKS = INPUT_DIM * FC1_COL_VECS_PER_COL * AXI_PER_COL_VEC;

// fc2: inner-product form. Weights streamed row-major; HIDDEN must be a
//      multiple of AXI_BLOCK_SIZE (600/8=75, 1200/8=150).
constexpr unsigned int HIDDEN_AXI_BLOCKS = HIDDEN_DIM / AXI_BLOCK_SIZE;
constexpr unsigned int FC2_WEIGHT_BLOCKS = OUTPUT_DIM * HIDDEN_AXI_BLOCKS;

void load_input(data_t dst[INPUT_DIM], const axi_block_t src[INPUT_BLOCKS]);

// fc1: out[HIDDEN] = W1 * in + bias  (W1 streamed col-major col_vec)
void fc1_stream(data_t out[HIDDEN_DIM],
                const data_t in[INPUT_DIM],
                const axi_block_t fc1_w[FC1_WEIGHT_AXI_BLOCKS],
                const data_t bias[HIDDEN_DIM]);

// fc2: out[OUTPUT] = W2 * act + bias  (W2 streamed row-major), wide output
void fc2_stream(wide_t out[OUTPUT_DIM],
                const data_t act[HIDDEN_DIM],
                const axi_block_t fc2_w[FC2_WEIGHT_BLOCKS],
                const data_t bias[OUTPUT_DIM]);

#endif // LINEAR_HPP
