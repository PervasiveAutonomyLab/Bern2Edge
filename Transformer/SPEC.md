# BernBert — network & kernel spec (paper TABLE XII)

TinyBERT4 sequence classifier (SST-2, binary) in which **all four transformer
FFN layers have their GeLU MLP replaced by a narrow `312 → H → 312` FFN**.
`models/` ships four distilled variants — two activations × two widths — that
are identical except for the FFN sub-block:

| Release             | Activation     | Width H | SST-2 acc  |
| ------------------- | -------------- | ------- | ---------- |
| `release_bern_h600` | Bernstein d=15 | 600     | **0.9048** |
| `release_bern_h312` | Bernstein d=15 | 312     | **0.9002** |
| `release_gelu_h600` | GeLU           | 600     | 0.9002     |
| `release_gelu_h312` | GeLU           | 312     | 0.8911     |

The Bernstein variants are the LUT-synthesis targets; the GeLU variants are the
matched-width baselines to compare against. Everything else is stock BERT.

This document is the reference for the network and the two FFN kernels. For how
to reproduce the accuracies, see [README.md](README.md).

- Base model: `huawei-noah/TinyBERT_General_4L_312D`
- Original GeLU teacher (H=1200) is 14,350,874 params / 0.9037 acc for reference.

## 1. Full network (standard BERT — implement as usual)

```
input_ids, attention_mask
      │
  Embeddings:  word(30522×312) + position(512×312) + token_type(2×312), then LayerNorm
      │
  4 × Transformer encoder layer:
        ├─ Multi-head self-attention (12 heads, dim 312) + residual + LayerNorm   ← standard
        └─ FFN sub-block                                                          ← REPLACED (section 2)
      │
  Pooler (dense 312×312 + tanh) on [CLS]
      │
  Classifier (dense 312×2)  →  logits
```

Hidden D=312, heads=12 (head dim 26), eval seq length 64, vocab 30522. The
attention/embeddings/pooler/classifier weights are in each release under their
usual HuggingFace `bert.*` names and need no special handling.

## 2. FFN sub-block (the replaced part) — per layer `ℓ`

Input `x ∈ ℝ³¹²` is the post-attention, post-LayerNorm token vector entering the
FFN block (identical interface to the original BERT FFN). Output is the layer
output `∈ ℝ³¹²`. `H` is the intermediate width: **312 or 600** depending on the
release. All four variants share this skeleton:

```
h = fc1.W · x + fc1.b                 # Linear 312 → H   (fc1.weight: H×312)
a = act(h)                            # per-channel activation, H → H
y = fc2.W · a + fc2.b                 # Linear H → 312   (fc2.weight: 312×H)
out = LayerNorm( y + x )              # dropout is identity at inference
```

`act` = **Bernstein** (§3) for the `bern_*` releases, **GeLU** (§4) for the
`gelu_*` releases. `LayerNorm` is over the 312-dim axis:
`((z−mean)/sqrt(var+eps))·γ + β`, **biased** variance (divide by N=312),
**eps = 1e-12**, params `ln.weight (γ)`, `ln.bias (β)`.

## 3. Bernstein activation — the lookup-table kernel

Applied **independently per channel** `c = 0..H−1`. Each channel has its own
input window `[lo_c, hi_c]` and its own 16 control points. Degree `n = 15`.

```
u_c   = (h_c − lo_c) / (hi_c − lo_c + 1e-8)     # affine map into [0,1]
u_c   = clamp(u_c, 0, 1)                         # SATURATING — out-of-window pins to 0/1
a_c   = Σ_{k=0}^{15}  coeff[c,k] · C(15,k) · u_c^k · (1 − u_c)^(15−k)
```

- `C(15,k)` = the fixed vector
  `[1,15,105,455,1365,3003,5005,6435,6435,5005,3003,1365,455,105,15,1]`.
- Each channel is a smooth 1-D function `f_c: [0,1] → ℝ`. To deploy as a LUT:
  precompute `f_c` at the desired resolution and replace the `u^k…` polynomial
  with `clamp → quantize(u_c) → table_read`; keep the per-channel affine pre-step
  and `lo_c, hi_c`. (No table ships here — pick resolution/format on the HW side.)
- Convex-hull bound: `min_k coeff[c,k] ≤ a_c ≤ max_k coeff[c,k]` (range sizing).

## 4. GeLU control activation

The `gelu_*` releases replace `act` with the exact erf GeLU (nn.GELU default):

```
a = 0.5 · h · (1 + erf(h / √2))
```

This is the transcendental the Bernstein LUT replaces — at the **same** width, so
the area/power/latency comparison is purely activation-vs-activation.

## 5. Weights

Each release ships a PyTorch `state_dict` (`models/release_*.pt`, load via
`bernbert.py`) plus a `.meta.json` giving act / hidden / degree / replaced
layers / verified accuracy. Per replaced
layer `ℓ` (slot into `replaced_layers = [0,1,2,3]`; the FFN submodules are named
`bern_ffns.ℓ.*` in **all** releases — historical name; H = 312 or 600):

| Tensor                          | Shape          | Bern | GeLU | Meaning                       |
| ------------------------------- | -------------- | :--: | :--: | ----------------------------- |
| `bern_ffns.ℓ.fc1.weight/bias`   | (H,312)/(H,)   |  ✓   |  ✓   | Linear 312→H                  |
| `bern_ffns.ℓ.bern.input_bounds` | (H, 2)         |  ✓   |  —   | per-channel `[lo, hi]`        |
| `bern_ffns.ℓ.bern.bern_coeffs`  | (H, 16)        |  ✓   |  —   | per-channel 16 control points |
| `bern_ffns.ℓ.fc2.weight/bias`   | (312,H)/(312,) |  ✓   |  ✓   | Linear H→312                  |
| `bern_ffns.ℓ.ln.weight/bias`    | (312,)         |  ✓   |  ✓   | LayerNorm γ/β                 |

(`bern.nCk`, `bern._basis_indices`, `bern._deg_tensor` are constant buffers — the
binomials/indices above; not learned. The GeLU FFN has no activation parameters.)

## 6. Checking an implementation

- `python load_and_run.py <variant> "<sentence>"` runs the whole network and
  prints the logits — feed the same sentence to your implementation and compare.
  Variants: `bern_h312`, `bern_h600`, `gelu_h312`, `gelu_h600`.
- `python eval_release.py` re-evaluates every variant on the SST-2 dev split and
  checks it against the accuracy recorded when the release was built.

> The HLS compiler losslessly extracts torch-free per-layer bundles from these
> releases and generates the matching FFN or encoder-layer kernels. See
> `hls/README.md`.

## 7. Comparison (the accuracy row of TABLE XII)

FFN params are the 4× per-layer budget (Linear + activation; LayerNorm excluded,
it is shared). Δ is vs the GeLU teacher; total-Δ equals FFN-Δ because only the
FFN sub-block changes.

| Model               | Activation  | Total params | FFN params (×4) |               Δ FFN |  SST-2 acc |
| ------------------- | ----------- | -----------: | --------------: | ------------------: | ---------: |
| Teacher (H=1200)    | erf GeLU    |   14,350,874 |       3,001,248 |                   — |     0.9037 |
| `release_bern_h600` | deg-15 poly |   12,889,274 |       1,539,648 | −1,461,600 (−48.7%) | **0.9048** |
| `release_gelu_h600` | erf GeLU    |   12,850,874 |       1,501,248 | −1,500,000 (−50.0%) |     0.9002 |
| `release_bern_h312` | deg-15 poly |   12,150,842 |         801,216 | −2,200,032 (−73.3%) | **0.9002** |
| `release_gelu_h312` | erf GeLU    |   12,130,874 |         781,248 | −2,220,000 (−74.0%) |     0.8911 |

At matched width Bernstein beats GeLU (h=312: +0.91pp; h=600: +0.46pp). Bern
h=312 matches GeLU h=600 (both 0.9002) at ~half the FFN params, and Bern h=600
beats the teacher at −48.7% FFN params.
