#ifndef SOFTMAX_HPP
#define SOFTMAX_HPP

#include "types.hpp"
#include "config.hpp"

// Stable softmax over one key-row of length SEQ_LEN: max-subtract, LUT exp,
// LUT reciprocal of the sum. Outputs probabilities in data_t.
void softmax_row(data_t prob[SEQ_LEN], const score_t row[SEQ_LEN]);

#endif // SOFTMAX_HPP
