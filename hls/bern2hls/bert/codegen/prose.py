"""config.hpp templates, one per (scope, activation).

Dimensions and table sizes are substituted; the trailing comments are carried
verbatim because they drift between bundles in wording and column alignment
(the h312 files align one space differently from h600) and are not derivable.
"""

CONFIG_FFN = {
    'bern': '''\
#ifndef CONFIG_HPP
#define CONFIG_HPP

// ============================================================
// Bernstein FFN sub-block (core: fc1 -> bern act -> fc2), TinyBERT layer 0
// Architecture: {in_dim} -> {hidden} -> {out_dim}, per-channel degree-15 Bernstein via LUT+lerp
// Weights are STREAMED from DRAM (m_axi); only activations + the per-channel
// LUT live on-chip. The degree-15 polynomial is baked into the LUT offline, so
// the hardware cost is independent of the polynomial degree.
// ============================================================

constexpr unsigned int ceildiv(unsigned int a, unsigned int b) {{ return (a + b - 1) / b; }}

// ---- Layer dimensions ----
constexpr unsigned int INPUT_DIM  = {in_dim};   // FFN-block input  (token vector)
constexpr unsigned int HIDDEN_DIM = {hidden};{hidden_note}
constexpr unsigned int OUTPUT_DIM = {out_dim};   // fc2 output

// ---- Bernstein activation (reference; evaluated offline into the LUT) ----
constexpr unsigned int BERNSTEIN_DEGREE = 15;
constexpr unsigned int NUM_BERN_COEFFS  = BERNSTEIN_DEGREE + 1;   // 16 control points
constexpr unsigned int NEURON_LUT_SIZE  = {lut_size};{lut_note}

// ---- Fixed-point / AXI ----
constexpr unsigned int FIXED_TOTAL_BITS = 32;
constexpr unsigned int FIXED_INT_BITS   = 8;
constexpr unsigned int AXI_WIDTH        = 256;
constexpr unsigned int AXI_BLOCK_SIZE   = AXI_WIDTH / FIXED_TOTAL_BITS;   // 8

// ---- Derived block / streaming counts ----
constexpr unsigned int INPUT_BLOCKS  = ceildiv(INPUT_DIM,  AXI_BLOCK_SIZE);   // 39
constexpr unsigned int OUTPUT_BLOCKS = ceildiv(OUTPUT_DIM, AXI_BLOCK_SIZE);   // 39

#endif // CONFIG_HPP
''',
    'gelu_lut': '''\
#ifndef CONFIG_HPP
#define CONFIG_HPP

// ============================================================
// GeLU baseline FFN sub-block (core: fc1 -> gelu act -> fc2), TinyBERT layer 0
// Architecture: {in_dim} -> {hidden} -> {out_dim}, GeLU via a SHARED LUT + lerp.
// GeLU is one shared 1-D function (unlike Bernstein's per-channel functions),
// so a single small ROM serves all {hidden} channels. Weights streamed from DRAM.
// ============================================================

constexpr unsigned int ceildiv(unsigned int a, unsigned int b) {{ return (a + b - 1) / b; }}

// ---- Layer dimensions ----
constexpr unsigned int INPUT_DIM  = {in_dim};
constexpr unsigned int HIDDEN_DIM = {hidden};{hidden_note}
constexpr unsigned int OUTPUT_DIM = {out_dim};

// ---- GeLU shared LUT ----
constexpr unsigned int GELU_LUT_SIZE = 512;          // shared entries over the clamp window
constexpr float        GELU_CLAMP_LO = -8.0f;        // <0.3% of pre-activations fall outside
constexpr float        GELU_CLAMP_HI =  8.0f;

// ---- Fixed-point / AXI ----
constexpr unsigned int FIXED_TOTAL_BITS = 32;
constexpr unsigned int FIXED_INT_BITS   = 8;
constexpr unsigned int AXI_WIDTH        = 256;
constexpr unsigned int AXI_BLOCK_SIZE   = AXI_WIDTH / FIXED_TOTAL_BITS;   // 8

// ---- Derived ----
constexpr unsigned int INPUT_BLOCKS  = ceildiv(INPUT_DIM,  AXI_BLOCK_SIZE);   // 39
constexpr unsigned int OUTPUT_BLOCKS = ceildiv(OUTPUT_DIM, AXI_BLOCK_SIZE);   // 39

#endif // CONFIG_HPP
''',
    'gelu_poly': '''\
#ifndef CONFIG_HPP
#define CONFIG_HPP

// ============================================================
// GeLU baseline FFN sub-block (core: fc1 -> gelu act -> fc2), TinyBERT layer 0
// Architecture: {in_dim} -> {hidden} -> {out_dim}, GeLU evaluated DIRECTLY (exact erf form),
// exposing the cost of the transcendental that the Bernstein LUT replaces.
// Weights streamed from DRAM.
// ============================================================

constexpr unsigned int ceildiv(unsigned int a, unsigned int b) {{ return (a + b - 1) / b; }}

// ---- Layer dimensions ----
constexpr unsigned int INPUT_DIM  = {in_dim};
constexpr unsigned int HIDDEN_DIM = {hidden};{hidden_note}
constexpr unsigned int OUTPUT_DIM = {out_dim};

// ---- Fixed-point / AXI ----
constexpr unsigned int FIXED_TOTAL_BITS = 32;
constexpr unsigned int FIXED_INT_BITS   = 8;
constexpr unsigned int AXI_WIDTH        = 256;
constexpr unsigned int AXI_BLOCK_SIZE   = AXI_WIDTH / FIXED_TOTAL_BITS;   // 8

// ---- Derived ----
constexpr unsigned int INPUT_BLOCKS  = ceildiv(INPUT_DIM,  AXI_BLOCK_SIZE);   // 39
constexpr unsigned int OUTPUT_BLOCKS = ceildiv(OUTPUT_DIM, AXI_BLOCK_SIZE);   // 39

#endif // CONFIG_HPP
''',
}


CONFIG_LAYER = {
    'bern': '''\
#ifndef CONFIG_HPP
#define CONFIG_HPP

// ============================================================
// Single TinyBERT encoder layer (Bernstein-FFN variant):
//   attention (12 heads) + residual + LN1  ->  FFN({in_dim}->{hidden}->{out_dim}) + residual + LN2
// Weights streamed from DRAM; attention is identical across all layer variants.
// ============================================================

constexpr unsigned int ceildiv(unsigned int a, unsigned int b) {{ return (a + b - 1) / b; }}

// ---- Sequence length (compile-time; override with -DSEQ_LEN=.. for cosim) ----
#ifndef SEQ_LEN
#define SEQ_LEN 16
#endif

// ---- Model dims ----
constexpr unsigned int D_MODEL   = 312;
constexpr unsigned int NUM_HEADS = 12;
constexpr unsigned int HEAD_DIM  = 26;        // 312 / 12
constexpr float        ATTN_SCALE = 0.196116135f;  // 1/sqrt(26)
constexpr float        LN_EPS     = 1e-12f;

// ---- FFN dims (Bernstein {in_dim}->{hidden}->{out_dim}{ffn_note}) ----
constexpr unsigned int INPUT_DIM  = {in_dim};
constexpr unsigned int HIDDEN_DIM = {hidden};
constexpr unsigned int OUTPUT_DIM = {out_dim};
constexpr unsigned int BERNSTEIN_DEGREE = 15;
constexpr unsigned int NUM_BERN_COEFFS  = BERNSTEIN_DEGREE + 1;
constexpr unsigned int NEURON_LUT_SIZE  = {lut_size};

// ---- Softmax LUTs ----
constexpr unsigned int EXP_LUT_SIZE = 256;
constexpr unsigned int INV_LUT_SIZE = 256;
constexpr float        DELTA_MIN    = -16.0f;   // exp domain after max-subtraction

// ---- Fixed-point / AXI ----
constexpr unsigned int FIXED_TOTAL_BITS = 32;
constexpr unsigned int FIXED_INT_BITS   = 8;
constexpr unsigned int AXI_WIDTH        = 256;
constexpr unsigned int AXI_BLOCK_SIZE   = AXI_WIDTH / FIXED_TOTAL_BITS;   // 8

// ---- Derived ----
constexpr unsigned int INPUT_BLOCKS  = ceildiv(INPUT_DIM,  AXI_BLOCK_SIZE);   // 39
constexpr unsigned int OUTPUT_BLOCKS = ceildiv(OUTPUT_DIM, AXI_BLOCK_SIZE);   // 39
// Q/K/V/Wo are each D_MODEL x D_MODEL, streamed row-major (proj_stream)
constexpr unsigned int PROJ_ELEMS       = D_MODEL * D_MODEL;                  // 97344
constexpr unsigned int PROJ_AXI_BLOCKS  = PROJ_ELEMS / AXI_BLOCK_SIZE;        // 12168
constexpr unsigned int QKVO_AXI_BLOCKS  = 4 * PROJ_AXI_BLOCKS;                // Wq|Wk|Wv|Wo

#endif // CONFIG_HPP
''',
    'gelu_lut': '''\
#ifndef CONFIG_HPP
#define CONFIG_HPP

// ============================================================
// Single TinyBERT encoder layer (GeLU shared-LUT FFN variant):
//   attention (12 heads) + residual + LN1  ->  FFN({in_dim}->{hidden}->{out_dim}) + residual + LN2
// Attention/softmax/LayerNorm code is identical to the Bernstein variant.
// ============================================================

constexpr unsigned int ceildiv(unsigned int a, unsigned int b) {{ return (a + b - 1) / b; }}

#ifndef SEQ_LEN
#define SEQ_LEN 16
#endif

// ---- Model dims ----
constexpr unsigned int D_MODEL   = 312;
constexpr unsigned int NUM_HEADS = 12;
constexpr unsigned int HEAD_DIM  = 26;
constexpr float        ATTN_SCALE = 0.196116135f;   // 1/sqrt(26)
constexpr float        LN_EPS     = 1e-12f;

// ---- FFN dims (GeLU {in_dim}->{hidden}->{out_dim}{ffn_note}) ----
constexpr unsigned int INPUT_DIM  = {in_dim};
constexpr unsigned int HIDDEN_DIM = {hidden};
constexpr unsigned int OUTPUT_DIM = {out_dim};

// ---- GeLU shared LUT ----
constexpr unsigned int GELU_LUT_SIZE = 512;
constexpr float        GELU_CLAMP_LO = -8.0f;
constexpr float        GELU_CLAMP_HI =  8.0f;

// ---- Softmax LUTs ----
constexpr unsigned int EXP_LUT_SIZE = 256;
constexpr unsigned int INV_LUT_SIZE = 256;
constexpr float        DELTA_MIN    = -16.0f;

// ---- Fixed-point / AXI ----
constexpr unsigned int FIXED_TOTAL_BITS = 32;
constexpr unsigned int FIXED_INT_BITS   = 8;
constexpr unsigned int AXI_WIDTH        = 256;
constexpr unsigned int AXI_BLOCK_SIZE   = AXI_WIDTH / FIXED_TOTAL_BITS;   // 8

// ---- Derived ----
constexpr unsigned int INPUT_BLOCKS  = ceildiv(INPUT_DIM,  AXI_BLOCK_SIZE);   // 39
constexpr unsigned int OUTPUT_BLOCKS = ceildiv(OUTPUT_DIM, AXI_BLOCK_SIZE);   // 39
constexpr unsigned int PROJ_ELEMS       = D_MODEL * D_MODEL;
constexpr unsigned int PROJ_AXI_BLOCKS  = PROJ_ELEMS / AXI_BLOCK_SIZE;
constexpr unsigned int QKVO_AXI_BLOCKS  = 4 * PROJ_AXI_BLOCKS;

#endif // CONFIG_HPP
''',
    'gelu_poly': '''\
#ifndef CONFIG_HPP
#define CONFIG_HPP

// ============================================================
// Single TinyBERT encoder layer (GeLU direct-erf FFN variant):
//   attention (12 heads) + residual + LN1  ->  FFN({in_dim}->{hidden}->{out_dim}) + residual + LN2
// Attention/softmax/LayerNorm code is identical to the Bernstein variant.
// GeLU evaluated directly (exact erf form), exposing the transcendental cost.
// ============================================================

constexpr unsigned int ceildiv(unsigned int a, unsigned int b) {{ return (a + b - 1) / b; }}

#ifndef SEQ_LEN
#define SEQ_LEN 16
#endif

constexpr unsigned int D_MODEL   = 312;
constexpr unsigned int NUM_HEADS = 12;
constexpr unsigned int HEAD_DIM  = 26;
constexpr float        ATTN_SCALE = 0.196116135f;
constexpr float        LN_EPS     = 1e-12f;

constexpr unsigned int INPUT_DIM  = {in_dim};
constexpr unsigned int HIDDEN_DIM = {hidden};{hidden_note}
constexpr unsigned int OUTPUT_DIM = {out_dim};

constexpr unsigned int EXP_LUT_SIZE = 256;
constexpr unsigned int INV_LUT_SIZE = 256;
constexpr float        DELTA_MIN    = -16.0f;

constexpr unsigned int FIXED_TOTAL_BITS = 32;
constexpr unsigned int FIXED_INT_BITS   = 8;
constexpr unsigned int AXI_WIDTH        = 256;
constexpr unsigned int AXI_BLOCK_SIZE   = AXI_WIDTH / FIXED_TOTAL_BITS;

constexpr unsigned int INPUT_BLOCKS  = ceildiv(INPUT_DIM,  AXI_BLOCK_SIZE);
constexpr unsigned int OUTPUT_BLOCKS = ceildiv(OUTPUT_DIM, AXI_BLOCK_SIZE);
constexpr unsigned int PROJ_ELEMS       = D_MODEL * D_MODEL;
constexpr unsigned int PROJ_AXI_BLOCKS  = PROJ_ELEMS / AXI_BLOCK_SIZE;
constexpr unsigned int QKVO_AXI_BLOCKS  = 4 * PROJ_AXI_BLOCKS;

#endif // CONFIG_HPP
''',
}
