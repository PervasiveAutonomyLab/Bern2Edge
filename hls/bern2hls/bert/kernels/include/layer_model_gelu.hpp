#ifndef LAYER_MODEL_HPP
#define LAYER_MODEL_HPP

#include "types.hpp"
#include "config.hpp"
#include "linear.hpp"
#include "gelu_act.hpp"
#include "attention.hpp"
#include "layernorm.hpp"

// One full TinyBERT encoder layer (GeLU FFN):
//   attention -> +residual -> LN1 -> FFN(312->1200->312) -> +residual -> LN2
void layer_top(data_t output[SEQ_LEN * OUTPUT_DIM],
               const axi_block_t input[SEQ_LEN * INPUT_BLOCKS],
               const axi_block_t qkvo_weights[QKVO_AXI_BLOCKS],
               const axi_block_t fc1_weights[FC1_WEIGHT_AXI_BLOCKS],
               const axi_block_t fc2_weights[FC2_WEIGHT_BLOCKS]);

#endif // LAYER_MODEL_HPP
