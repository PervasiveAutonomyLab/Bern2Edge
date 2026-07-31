/*
 * Full encoder-layer testbench (shared by all layer variants).
 * Loads attention (Wq|Wk|Wv|Wo) + FFN weights and the numpy golden, runs the
 * HLS layer_top, compares the SEQ_LEN×312 output (relative tolerance).
 */
#include <iostream>
#include <fstream>
#include <cmath>
#include <iomanip>
#include <string>
#include "../include/layer_model.hpp"

#ifndef NUM_TEST_SAMPLES
#define NUM_TEST_SAMPLES 2
#endif
#define REL_TOL 0.03

static float qkvo_f[QKVO_AXI_BLOCKS * AXI_BLOCK_SIZE];
static float fc1_f[FC1_WEIGHT_AXI_BLOCKS * AXI_BLOCK_SIZE];
static float fc2_f[FC2_WEIGHT_BLOCKS * AXI_BLOCK_SIZE];
static float in_f[NUM_TEST_SAMPLES * SEQ_LEN * INPUT_DIM];
static float ref_f[NUM_TEST_SAMPLES * SEQ_LEN * OUTPUT_DIM];

static axi_block_t qkvo_axi[QKVO_AXI_BLOCKS];
static axi_block_t fc1_axi[FC1_WEIGHT_AXI_BLOCKS];
static axi_block_t fc2_axi[FC2_WEIGHT_BLOCKS];
static axi_block_t in_axi[SEQ_LEN * INPUT_BLOCKS];
static data_t out_buf[SEQ_LEN * OUTPUT_DIM];

static bool load_values(const std::string& fn, float* d, int n) {
    std::ifstream f(fn);
    if (!f.is_open()) { std::cerr << "ERROR: cannot open " << fn << "\n"; return false; }
    for (int i = 0; i < n; i++)
        if (!(f >> d[i])) { std::cerr << "ERROR: short read " << fn << " (" << i << "/" << n << ")\n"; return false; }
    return true;
}
static std::string find_data_dir() {
    const std::string c[] = {"../../data","../../../data","../../../../data","../../../../../data","data","../data"};
    for (const auto& d : c) { std::ifstream f(d + "/qkv_o_weights.txt"); if (f.is_open()) { std::cout<<"[TB] data dir: "<<d<<"\n"; return d; } }
    return "../../data";
}

int main() {
    std::string dd = find_data_dir();
    std::cout << "Layer TB  SEQ=" << SEQ_LEN << " D=" << D_MODEL << " HIDDEN=" << HIDDEN_DIM
              << " samples=" << NUM_TEST_SAMPLES << "\n";

    if (!load_values(dd + "/qkv_o_weights.txt", qkvo_f, QKVO_AXI_BLOCKS * AXI_BLOCK_SIZE)) return 1;
    if (!load_values(dd + "/fc1_weights.txt",   fc1_f,  FC1_WEIGHT_AXI_BLOCKS * AXI_BLOCK_SIZE)) return 1;
    if (!load_values(dd + "/fc2_weights.txt",   fc2_f,  OUTPUT_DIM * HIDDEN_DIM)) return 1;
    if (!load_values(dd + "/test_input.txt",    in_f,   NUM_TEST_SAMPLES * SEQ_LEN * INPUT_DIM)) return 1;
    if (!load_values(dd + "/test_output_ref.txt", ref_f, NUM_TEST_SAMPLES * SEQ_LEN * OUTPUT_DIM)) return 1;

    for (unsigned b = 0; b < QKVO_AXI_BLOCKS; b++)
        for (unsigned i = 0; i < AXI_BLOCK_SIZE; i++) qkvo_axi[b][i] = data_t(qkvo_f[b*AXI_BLOCK_SIZE+i]);
    for (unsigned b = 0; b < FC1_WEIGHT_AXI_BLOCKS; b++)
        for (unsigned i = 0; i < AXI_BLOCK_SIZE; i++) fc1_axi[b][i] = data_t(fc1_f[b*AXI_BLOCK_SIZE+i]);
    for (unsigned idx = 0; idx < OUTPUT_DIM * HIDDEN_DIM; idx++)
        fc2_axi[idx/AXI_BLOCK_SIZE][idx%AXI_BLOCK_SIZE] = data_t(fc2_f[idx]);

    float g_max_rel = 0.f, g_min_cos = 1.f; int pass = 0;
    for (int s = 0; s < NUM_TEST_SAMPLES; s++) {
        // pack input tokens
        for (unsigned t = 0; t < SEQ_LEN; t++)
            for (unsigned b = 0; b < INPUT_BLOCKS; b++)
                for (unsigned i = 0; i < AXI_BLOCK_SIZE; i++) {
                    unsigned f = b*AXI_BLOCK_SIZE + i;
                    float v = (f < INPUT_DIM) ? in_f[(s*SEQ_LEN + t)*INPUT_DIM + f] : 0.f;
                    in_axi[t*INPUT_BLOCKS + b][i] = data_t(v);
                }

        layer_top(out_buf, in_axi, qkvo_axi, fc1_axi, fc2_axi);

        float max_err = 0.f, max_ref = 1e-9f, dot = 0, nh = 0, nr = 0;
        for (unsigned k = 0; k < SEQ_LEN * OUTPUT_DIM; k++) {
            float hv = (float)out_buf[k];
            float rv = ref_f[s*SEQ_LEN*OUTPUT_DIM + k];
            max_err = std::max(max_err, std::fabs(hv - rv));
            max_ref = std::max(max_ref, std::fabs(rv));
            dot += hv*rv; nh += hv*hv; nr += rv*rv;
        }
        float rel = max_err/max_ref, cos = dot/(std::sqrt(nh)*std::sqrt(nr)+1e-30f);
        g_max_rel = std::max(g_max_rel, rel); g_min_cos = std::min(g_min_cos, cos);
        if (rel < REL_TOL) pass++;
        std::cout << std::fixed << std::setprecision(6)
                  << "  sample " << s << ": max_abs_err=" << max_err << " max|ref|=" << max_ref
                  << " rel=" << rel << " cos=" << cos << (rel<REL_TOL?"  OK":"  WARN") << "\n";
    }
    std::cout << "----\n  passed " << pass << "/" << NUM_TEST_SAMPLES
              << "  worst_rel=" << g_max_rel << "  worst_cos=" << g_min_cos << "\n";
    std::cout << (pass==NUM_TEST_SAMPLES ? "RESULT: ALL PASSED\n" : "RESULT: SOME EXCEEDED TOL\n");
    return (pass==NUM_TEST_SAMPLES) ? 0 : 1;
}
