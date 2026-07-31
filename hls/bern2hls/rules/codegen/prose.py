"""Per-suite C++ text: file headers, type declarations and Phase-3 bodies.

Isolated from the emitter on purpose. The three shipped suites were forked
from one another and drifted in wording *and* whitespace — `types.hpp` differs
between `fallback_4_variance` and `sparsity_sweep` only in comment capitalization
and column alignment, for instance. None of that is derivable, so byte-identity
requires carrying it as data. Keeping it here means the emitter stays about
structure and this file is honest about being transcription.
"""

# ---------------------------------------------------------------- types.hpp

TYPES_HPP = {
    # fallback_4_variance / tree_arch: has nn_t (Bernstein-NN fallbacks), no idx_t
    'f4v': """\
#ifndef RULE_TYPES_HPP
#define RULE_TYPES_HPP

#include <ap_int.h>
#include <ap_fixed.h>
#include "config.hpp"

typedef ap_int<8>        w8_t;     // int8 quantized weight
typedef ap_fixed<16, 8>  data_t;   // inputs, band bounds, tree thresholds
typedef ap_fixed<32, 16> acc_t;    // accumulators, scales
typedef ap_fixed<32, 16> nn_t;     // full-precision NN params / activations
typedef unsigned int     purity_t; // purity * 10000

#endif
""",
    # tree_arch: no NN fallback, and stripped of comments and blank lines
    'tree_arch': """\
#ifndef RULE_TYPES_HPP
#define RULE_TYPES_HPP
#include <ap_int.h>
#include <ap_fixed.h>
#include "config.hpp"
typedef ap_int<8>        w8_t;
typedef ap_fixed<16, 8>  data_t;
typedef ap_fixed<32, 16> acc_t;
typedef unsigned int     purity_t;
#endif
""",
    # sparsity_sweep: has idx_t (sparse feature index), no nn_t (no NN fallback)
    'ksweep': """\
#ifndef RULE_TYPES_HPP
#define RULE_TYPES_HPP

#include <ap_int.h>
#include <ap_fixed.h>
#include "config.hpp"

typedef ap_int<8>       w8_t;       // Int8 quantized weight
typedef ap_uint<4>      idx_t;      // Feature index (0-13 fits in 4 bits)
typedef ap_fixed<16, 8> data_t;     // Inputs, band bounds (fix<16,8>)
typedef ap_fixed<32,16> acc_t;      // Accumulator, scale, intermediates
typedef unsigned int    purity_t;   // Purity * 10000

#endif
""",
}

HDR_HPP = {
    'f4v': """\
#ifndef RULE_CLASSIFIER_HPP
#define RULE_CLASSIFIER_HPP
#include "types.hpp"
#include "config.hpp"
void {top_fn}(int *result, const data_t x[N_FEATURES]);
#endif
""",
    'ksweep': """\
#ifndef RULE_CLASSIFIER_HPP
#define RULE_CLASSIFIER_HPP

#include "types.hpp"
#include "config.hpp"

void {top_fn}(
    int *result,
    const data_t x[N_FEATURES]
);

#endif
""",
}

# ROM layout: indent width, the word before the rule number, and the header
# comment above COND_W8. Cosmetic, but it is what byte-identity turns on.
ROM_STYLE = {
    'f4v': dict(i1='  ', i2='    ', word='rule',
                header='// ---- shared 50-rule int8 classifier ROM ----'),
    'tree_arch': dict(i1='  ', i2='    ', word='rule',
                      header='// ---- shared rule classifier ROM (dense int8) ----'),
    'tree_arch': dict(i1='  ', i2='    ', word='rule',
                      header='// ---- shared rule classifier ROM (dense int8) ----'),
    'ksweep': dict(i1='    ', i2='        ', word='Rule',
                   header='// Sparse int8 quantized condition weights [rule][cond][K_SPARSE]'),
}

# --------------------------------------------------------------- config.hpp

# The fallback-only tops need no rule tables at all, so their config drops
# N_RULES/MAX_CONDS (and the blank line with them).
CONFIG_HPP_FB_ONLY = """\
#ifndef RULE_CONFIG_HPP
#define RULE_CONFIG_HPP
constexpr unsigned int N_FEATURES = {n_features};
{extra}#endif
"""

CONFIG_HPP = {
    'tree_arch': """\
#ifndef RULE_CONFIG_HPP
#define RULE_CONFIG_HPP
constexpr unsigned int N_FEATURES = {n_features};
constexpr unsigned int N_RULES    = {n_rules};
constexpr unsigned int MAX_CONDS  = {max_conds};
{extra}#endif
""",
    'f4v': """\
#ifndef RULE_CONFIG_HPP
#define RULE_CONFIG_HPP

constexpr unsigned int N_FEATURES = {n_features};
constexpr unsigned int N_RULES    = {n_rules};
constexpr unsigned int MAX_CONDS  = {max_conds};
{extra}#endif
""",
    'ksweep': """\
#ifndef RULE_CONFIG_HPP
#define RULE_CONFIG_HPP

constexpr unsigned int N_FEATURES = {n_features};
constexpr unsigned int K_SPARSE   = {k_sparse};    // Non-zero weights per condition (= sweep K)
constexpr unsigned int N_RULES    = {n_rules};
constexpr unsigned int MAX_CONDS  = {max_conds};

#endif
""",
}

# ------------------------------------------------------------ Phase-3 bodies

PHASE3 = {
    'lr': """\
        acc_t lr_sum = 0;
        for (unsigned int i = 0; i < N_FEATURES; i++) {
            #pragma HLS UNROLL
            lr_sum += acc_t(LR_W8[i]) * acc_t(x_local[i]);
        }
        acc_t logit = lr_sum * LR_SCALE + acc_t(LR_B_EFF);
        best_label = (logit > acc_t(0)) ? 1 : 0;""",

    'network': """\
        nn_t logit0 = NN_B2[0], logit1 = NN_B2[1];
        FB_NEURON:
        for (unsigned int j = 0; j < NN_HID; j++) {
            #pragma HLS PIPELINE II=1
            nn_t z = NN_B0[j];
            for (unsigned int i = 0; i < N_FEATURES; i++) {
                #pragma HLS UNROLL
                z += NN_W0[j][i] * nn_t(x_local[i]);
            }
            nn_t u = (z - NN_LO[j]) * NN_INVR[j];
            if (u < nn_t(0)) u = nn_t(0);
            if (u > nn_t(1)) u = nn_t(1);
            nn_t omu = nn_t(1) - u;
            nn_t b0 = omu*omu*omu;
            nn_t b1 = nn_t(3)*u*omu*omu;
            nn_t b2 = nn_t(3)*u*u*omu;
            nn_t b3 = u*u*u;
            nn_t a = NN_COEFF[j][0]*b0 + NN_COEFF[j][1]*b1
                   + NN_COEFF[j][2]*b2 + NN_COEFF[j][3]*b3;
            logit0 += NN_W2[0][j] * a;
            logit1 += NN_W2[1][j] * a;
        }
        best_label = (logit1 > logit0) ? 1 : 0;""",

    'small_nn': """\
        nn_t a_buf[NN_HID];
        FB_NEURON:
        for (unsigned int j = 0; j < NN_HID; j++) {
            #pragma HLS PIPELINE II=1
            acc_t acc = 0;
            for (unsigned int i = 0; i < N_FEATURES; i++) {
                #pragma HLS UNROLL
                acc += acc_t(NN_W0Q[j][i]) * acc_t(x_local[i]);
            }
            nn_t z = nn_t(acc * NN_S0) + NN_B0[j];
            nn_t u = (z - NN_LO[j]) * NN_INVR[j];
            if (u < nn_t(0)) u = nn_t(0);
            if (u > nn_t(1)) u = nn_t(1);
            nn_t omu = nn_t(1) - u;
            nn_t b0 = omu*omu*omu;
            nn_t b1 = nn_t(3)*u*omu*omu;
            nn_t b2 = nn_t(3)*u*u*omu;
            nn_t b3 = u*u*u;
            a_buf[j] = NN_COEFF[j][0]*b0 + NN_COEFF[j][1]*b1
                     + NN_COEFF[j][2]*b2 + NN_COEFF[j][3]*b3;
        }
        acc_t o0 = 0, o1 = 0;
        for (unsigned int j = 0; j < NN_HID; j++) {
            #pragma HLS UNROLL
            o0 += acc_t(NN_W2Q[0][j]) * acc_t(a_buf[j]);
            o1 += acc_t(NN_W2Q[1][j]) * acc_t(a_buf[j]);
        }
        nn_t logit0 = nn_t(o0 * NN_S2) + NN_B2[0];
        nn_t logit1 = nn_t(o1 * NN_S2) + NN_B2[1];
        best_label = (logit1 > logit0) ? 1 : 0;""",

    'tree': """\
        int node = 0;
        FB_TREE:
        for (unsigned int dep = 0; dep < TREE_MAX_DEPTH + 1; dep++) {
            #pragma HLS PIPELINE II=1
            if (TREE_FEAT[node] == -2) break;
            if (x_local[TREE_FEAT[node]] <= TREE_THR[node])
                node = TREE_LEFT[node];
            else
                node = TREE_RIGHT[node];
        }
        best_label = TREE_LEAF[node];""",
}

# ------------------------------------------------------- src header comments

SRC_HEADER = {
    'f4v': """\
// ============================================
// Rule classifier + {variant} fallback
//   Phase 1: dense int8 per-condition dot products
//   Phase 2: max-purity rule evaluation
//   Phase 3: {variant} fallback for uncovered inputs
// ============================================""",
}

FB_ONLY_HEADER = ('// Fallback block ONLY (Phase-3 of the {variant} variant)'
                  ' — for isolated resource cost.')

# ------------------------------------------------------- full source template
#
# One per suite rather than one parameterized template: the suites are forks,
# and they diverge in the function signature layout, pragma column alignment
# and comment blocks, not just in the Phase-1 inner loop.

SRC_TMPL = {
    'tree_arch': """\
#include "../include/rule_classifier.hpp"
#include "../include/rule_rom.hpp"

// Rule classifier + CART tree fallback ({tag})
//   Phase 1: dense int8 per-condition dot products
//   Phase 2: max-purity rule evaluation
//   Phase 3: tree fallback (comparisons only) for uncovered inputs
void {top_fn}(int *result, const data_t x[N_FEATURES]) {{
    #pragma HLS INTERFACE s_axilite port=result bundle=control
    #pragma HLS INTERFACE s_axilite port=x      bundle=control
    #pragma HLS INTERFACE s_axilite port=return bundle=control

    // Force the rule-weight ROM into block RAM (uniform across archs);
    // reshape dim=3 so all N_FEATURES weights read in parallel from one wide word.
    #pragma HLS ARRAY_RESHAPE variable=COND_W8 complete dim=3
    #pragma HLS BIND_STORAGE  variable=COND_W8 type=rom_1p impl=bram

    data_t x_local[N_FEATURES];
    #pragma HLS ARRAY_PARTITION variable=x_local complete
    for (unsigned int i = 0; i < N_FEATURES; i++) {{
        #pragma HLS UNROLL
        x_local[i] = x[i];
    }}

    // ---- Phase 1: per-condition dot products ----
    acc_t z_vals[N_RULES][MAX_CONDS];
    COMPUTE_DOT:
    for (unsigned int r = 0; r < N_RULES; r++) {{
        for (unsigned int c = 0; c < MAX_CONDS; c++) {{
            #pragma HLS PIPELINE II=1
            acc_t z = 0;
            for (unsigned int i = 0; i < N_FEATURES; i++) {{
                #pragma HLS UNROLL
                z += acc_t(COND_W8[r][c][i]) * acc_t(x_local[i]);
            }}
            z_vals[r][c] = z * COND_SCALE[r][c];
        }}
    }}

    // ---- Phase 2: max-purity rule evaluation ----
    int best_label = -1;
    purity_t best_purity = 0;
    EVAL_RULES:
    for (unsigned int r = 0; r < N_RULES; r++) {{
        #pragma HLS PIPELINE II=1
        bool fires = true;
        for (unsigned int c = 0; c < MAX_CONDS; c++) {{
            #pragma HLS UNROLL
            if (c < RULE_N_COND[r]) {{
                bool lo_ok = !COND_HAS_LO[r][c] || (z_vals[r][c] >= acc_t(COND_BAND_LO[r][c]));
                bool hi_ok = !COND_HAS_HI[r][c] || (z_vals[r][c] <  acc_t(COND_BAND_HI[r][c]));
                if (!lo_ok || !hi_ok) fires = false;
            }}
        }}
        if (fires && RULE_PURITY[r] > best_purity) {{
            best_label  = RULE_LABEL[r];
            best_purity = RULE_PURITY[r];
        }}
    }}

    // ---- Phase 3: {variant} fallback ----
    if (best_label == -1) {{
{phase3}
    }}

    *result = best_label;
}}
""",
    'f4v': """\
#include "../include/rule_classifier.hpp"
#include "../include/rule_rom.hpp"

// ============================================
// Rule classifier + {variant} fallback
//   Phase 1: dense int8 per-condition dot products
//   Phase 2: max-purity rule evaluation
//   Phase 3: {variant} fallback for uncovered inputs
// ============================================
void {top_fn}(int *result, const data_t x[N_FEATURES]) {{
    #pragma HLS INTERFACE s_axilite port=result bundle=control
    #pragma HLS INTERFACE s_axilite port=x      bundle=control
    #pragma HLS INTERFACE s_axilite port=return bundle=control

    data_t x_local[N_FEATURES];
    #pragma HLS ARRAY_PARTITION variable=x_local complete
    for (unsigned int i = 0; i < N_FEATURES; i++) {{
        #pragma HLS UNROLL
        x_local[i] = x[i];
    }}

    // ---- Phase 1: per-condition dot products ----
    acc_t z_vals[N_RULES][MAX_CONDS];
    COMPUTE_DOT:
    for (unsigned int r = 0; r < N_RULES; r++) {{
        for (unsigned int c = 0; c < MAX_CONDS; c++) {{
            #pragma HLS PIPELINE II=1
            acc_t z = 0;
            for (unsigned int i = 0; i < N_FEATURES; i++) {{
                #pragma HLS UNROLL
                z += acc_t(COND_W8[r][c][i]) * acc_t(x_local[i]);
            }}
            z_vals[r][c] = z * COND_SCALE[r][c];
        }}
    }}

    // ---- Phase 2: max-purity rule evaluation ----
    int best_label = -1;
    purity_t best_purity = 0;
    EVAL_RULES:
    for (unsigned int r = 0; r < N_RULES; r++) {{
        #pragma HLS PIPELINE II=1
        bool fires = true;
        for (unsigned int c = 0; c < MAX_CONDS; c++) {{
            #pragma HLS UNROLL
            if (c < RULE_N_COND[r]) {{
                bool lo_ok = !COND_HAS_LO[r][c] || (z_vals[r][c] >= acc_t(COND_BAND_LO[r][c]));
                bool hi_ok = !COND_HAS_HI[r][c] || (z_vals[r][c] <  acc_t(COND_BAND_HI[r][c]));
                if (!lo_ok || !hi_ok) fires = false;
            }}
        }}
        if (fires && RULE_PURITY[r] > best_purity) {{
            best_label  = RULE_LABEL[r];
            best_purity = RULE_PURITY[r];
        }}
    }}

    // ---- Phase 3: {variant} fallback ----
    if (best_label == -1) {{
{phase3}
    }}

    *result = best_label;
}}
""",
    'ksweep': """\
#include "../include/rule_classifier.hpp"
#include "../include/rule_rom.hpp"

// ============================================
// SPARSE Int8 Rule Classifier — Hybrid Architecture
// ============================================
// Phase 1: Per-condition SPARSE int8 dot products (pipelined)
//   z[r][c] = (sum w8[j] * x[idx[j]], j=0..K_SPARSE-1) * scale
// Phase 2: Rule evaluation (pipelined)
//   Check band conditions, track max-purity match
// Phase 3: LR fallback for uncovered samples
//   logit = (sum lr_w8[i] * x[i]) * lr_scale + b_eff
// ============================================

void {top_fn}(
    int *result,
    const data_t x[N_FEATURES]
) {{
    #pragma HLS INTERFACE s_axilite port=result bundle=control
    #pragma HLS INTERFACE s_axilite port=x     bundle=control
    #pragma HLS INTERFACE s_axilite port=return bundle=control

    // ---- Force the sparse weight ROM into BRAM ----
    // Bank along the K (innermost) axis -> K independent BRAMs, each holding the
    // [N_RULES][MAX_CONDS] = {banked} entries for one non-zero slot. The Phase-1 loop
    // reads slot j of condition (r,c) from bank j, addressed sequentially by (r,c),
    // so each bank takes exactly ONE read/cycle: K parallel reads, no port conflict,
    // II=1 preserved. BRAM usage now scales with K (the weight footprint).
    #pragma HLS ARRAY_PARTITION variable=COND_W8 dim=3 type=complete
    #pragma HLS BIND_STORAGE    variable=COND_W8 type=rom_1p impl=bram

    // ---- Local copy of input ----
    data_t x_local[N_FEATURES];
    #pragma HLS ARRAY_PARTITION variable=x_local complete
    for (unsigned int i = 0; i < N_FEATURES; i++) {{
        #pragma HLS UNROLL
        x_local[i] = x[i];
    }}

    // ---- Phase 1: Compute all condition dot products (SPARSE) ----
    acc_t z_vals[N_RULES][MAX_CONDS];

    COMPUTE_DOT:
    for (unsigned int r = 0; r < N_RULES; r++) {{
        for (unsigned int c = 0; c < MAX_CONDS; c++) {{
            #pragma HLS PIPELINE II=1
            acc_t z = 0;
            for (unsigned int j = 0; j < K_SPARSE; j++) {{
                #pragma HLS UNROLL
                z += acc_t(COND_W8[r][c][j]) * acc_t(x_local[COND_IDX[r][c][j]]);
            }}
            z_vals[r][c] = z * COND_SCALE[r][c];
        }}
    }}

    // ---- Phase 2: Evaluate rules ----
    int best_label = -1;
    purity_t best_purity = 0;

    EVAL_RULES:
    for (unsigned int r = 0; r < N_RULES; r++) {{
        #pragma HLS PIPELINE II=1

        bool fires = true;
        for (unsigned int c = 0; c < MAX_CONDS; c++) {{
            #pragma HLS UNROLL
            if (c < RULE_N_COND[r]) {{
                bool lo_ok = !COND_HAS_LO[r][c] || (z_vals[r][c] >= acc_t(COND_BAND_LO[r][c]));
                bool hi_ok = !COND_HAS_HI[r][c] || (z_vals[r][c] <  acc_t(COND_BAND_HI[r][c]));
                if (!lo_ok || !hi_ok) fires = false;
            }}
        }}

        if (fires && RULE_PURITY[r] > best_purity) {{
            best_label  = RULE_LABEL[r];
            best_purity = RULE_PURITY[r];
        }}
    }}

    // ---- Phase 3: LR fallback (dense — only ~2 zeros of 14) ----
    if (best_label == -1) {{
{phase3}
    }}

    *result = best_label;
}}
""",
}


# --------------------------------------------------------------- testbenches
# Per-suite again: the k-sweep testbench prints K_SPARSE, uses a 200-sample
# subset, and declares its helpers non-static. Transcribed once for 14 projects.
TB_KSWEEP = """\
#include <iostream>
#include <fstream>
#include <iomanip>
#include <string>
#include "../include/rule_classifier.hpp"

#define NUM_TEST {num_test}

bool load_floats(const std::string& fn, float* data, int count) {{
    std::ifstream f(fn);
    if (!f.is_open()) {{ std::cerr << "Cannot open " << fn << std::endl; return false; }}
    for (int i = 0; i < count; i++)
        if (!(f >> data[i])) {{ std::cerr << "Short read " << fn << std::endl; return false; }}
    return true;
}}

bool load_ints(const std::string& fn, int* data, int count) {{
    std::ifstream f(fn);
    if (!f.is_open()) {{ std::cerr << "Cannot open " << fn << std::endl; return false; }}
    for (int i = 0; i < count; i++)
        if (!(f >> data[i])) {{ std::cerr << "Short read " << fn << std::endl; return false; }}
    return true;
}}

std::string find_data_dir() {{
    const std::string candidates[] = {{
        "../data", "../../data", "../../../data",
        "../../../../data", "../../../../../data", "data",
    }};
    for (const auto& dir : candidates) {{
        std::ifstream f(dir + "/test_input.txt");
        if (f.is_open()) {{ f.close(); return dir; }}
    }}
    return "../data";
}}

static float test_inputs_f[NUM_TEST * N_FEATURES];
static int   test_labels[NUM_TEST];
static int   test_ref_preds[NUM_TEST];

int main() {{
    std::string data_dir = find_data_dir();

    std::cout << "========================================" << std::endl;
    std::cout << "  Sparse Int8 Rule Classifier Testbench" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << "  N_FEATURES=" << N_FEATURES
              << "  K_SPARSE=" << K_SPARSE
              << "  N_RULES=" << N_RULES
              << "  MAX_CONDS=" << MAX_CONDS << std::endl;
    std::cout << "  Samples=" << NUM_TEST << std::endl;
    std::cout << "  Data dir: " << data_dir << std::endl << std::endl;

    if (!load_floats(data_dir + "/test_input.txt",
                     test_inputs_f, NUM_TEST * N_FEATURES)) return 1;
    if (!load_ints(data_dir + "/test_labels.txt",
                   test_labels, NUM_TEST)) return 1;
    if (!load_ints(data_dir + "/test_output_ref.txt",
                   test_ref_preds, NUM_TEST)) return 1;

    int hls_correct = 0, ref_correct = 0, agree = 0, mismatch = 0;

    for (int s = 0; s < NUM_TEST; s++) {{
        data_t x[N_FEATURES];
        for (unsigned int j = 0; j < N_FEATURES; j++)
            x[j] = data_t(test_inputs_f[s * N_FEATURES + j]);

        int hls_pred;
        rule_classifier_top(&hls_pred, x);

        int gt = test_labels[s], ref_p = test_ref_preds[s];
        if (hls_pred == gt)    hls_correct++;
        if (ref_p == gt)       ref_correct++;
        if (hls_pred == ref_p) agree++; else mismatch++;

        if (hls_pred != ref_p) {{
            std::cout << "  S" << std::setw(3) << s
                      << ": HLS=" << hls_pred << " ref=" << ref_p
                      << " gt=" << gt << " **MISMATCH**" << std::endl;
        }}
    }}

    std::cout << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << "  Summary" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << std::fixed << std::setprecision(2);
    std::cout << "  HLS accuracy:  " << hls_correct << "/" << NUM_TEST
              << " (" << (float)hls_correct/NUM_TEST*100 << "%)" << std::endl;
    std::cout << "  Ref accuracy:  " << ref_correct << "/" << NUM_TEST
              << " (" << (float)ref_correct/NUM_TEST*100 << "%)" << std::endl;
    std::cout << "  HLS==Ref:      " << agree << "/" << NUM_TEST
              << " (" << (float)agree/NUM_TEST*100 << "%)" << std::endl;
    std::cout << "  Mismatches:    " << mismatch << std::endl;
    std::cout << "========================================" << std::endl;

    if (mismatch == 0)
        std::cout << "RESULT: ALL TESTS PASSED" << std::endl;
    else
        std::cout << "RESULT: " << mismatch << " mismatches (int8 quantization)" << std::endl;

    return (mismatch > NUM_TEST * 10 / 100) ? 1 : 0;
}}
"""


# tree_arch reuses the f4v testbench but drops N_FEATURES from the banner.
TB_TREE_ARCH_BANNER = ('    std::cout << "N_RULES=" << N_RULES << " MAX_CONDS=" << MAX_CONDS\n'
                       '              << " samples=" << NUM_TEST << std::endl;')
TB_F4V_BANNER = ('    std::cout << "N_FEATURES=" << N_FEATURES << " N_RULES=" << N_RULES\n'
                 '              << " MAX_CONDS=" << MAX_CONDS << " samples=" << NUM_TEST << std::endl;')
