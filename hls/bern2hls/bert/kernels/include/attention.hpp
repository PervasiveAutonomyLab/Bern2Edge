#ifndef ATTENTION_HPP
#define ATTENTION_HPP

#include "types.hpp"
#include "config.hpp"

// Self-attention sublayer: Q/K/V projections, per-head scaled dot-product +
// softmax + context, output projection, residual, LayerNorm1.
//   x    : [SEQ_LEN][D_MODEL]  layer input (post-attention-LN of previous layer)
//   qkvo_w: streamed Wq|Wk|Wv|Wo (row-major, concatenated)
//   xffn : [SEQ_LEN][D_MODEL]  output = LN1(x + attn_out)  (FFN input)
void attention_sublayer(data_t xffn[SEQ_LEN][D_MODEL],
                        const data_t x[SEQ_LEN][D_MODEL],
                        const axi_block_t qkvo_w[QKVO_AXI_BLOCKS]);

#endif // ATTENTION_HPP
