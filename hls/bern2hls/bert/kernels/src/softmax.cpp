#include "../include/softmax.hpp"
#include "../include/softmax_lut_rom.hpp"

// Matches the numpy "faithful" softmax: nearest-index exp LUT over [DELTA_MIN,0]
// and nearest-index reciprocal LUT over [1,SEQ_LEN].
void softmax_row(data_t prob[SEQ_LEN], const score_t row[SEQ_LEN])
{
#pragma HLS inline off
    const idx_t exp_scale = idx_t((EXP_LUT_SIZE - 1) / (0.0f - DELTA_MIN));
    const idx_t inv_scale = idx_t((INV_LUT_SIZE - 1) / (float)(SEQ_LEN - 1));

    // 1. max over keys
    score_t m = row[0];
SM_MAX:
    for (unsigned int k = 1; k < SEQ_LEN; k++) {
#pragma HLS pipeline II=1
        if (row[k] > m) m = row[k];
    }

    // 2. exp(delta) via LUT, accumulate sum
    exp_t e[SEQ_LEN];
#pragma HLS array_partition variable=e complete
    sm_acc_t sum = 0;
SM_EXP:
    for (unsigned int k = 0; k < SEQ_LEN; k++) {
#pragma HLS pipeline II=1
        score_t d = row[k] - m;                       // <= 0
        if (d < score_t(DELTA_MIN)) d = score_t(DELTA_MIN);
        idx_t pos = (idx_t(d) - idx_t(DELTA_MIN)) * exp_scale;   // >= 0
        unsigned int ei = pos.to_uint();
        if (ei >= EXP_LUT_SIZE) ei = EXP_LUT_SIZE - 1;
        e[k] = EXP_LUT[ei];
        sum += e[k];
    }

    // 3. reciprocal of sum via LUT (sum in [1, SEQ_LEN])
    idx_t sp = (idx_t(sum) - idx_t(1)) * inv_scale;
    if (sp < idx_t(0)) sp = idx_t(0);
    unsigned int si = sp.to_uint();
    if (si >= INV_LUT_SIZE) si = INV_LUT_SIZE - 1;
    data_t inv = INV_LUT[si];

    // 4. normalize
SM_NORM:
    for (unsigned int k = 0; k < SEQ_LEN; k++) {
#pragma HLS pipeline II=1
        prob[k] = data_t(e[k]) * inv;
    }
}
