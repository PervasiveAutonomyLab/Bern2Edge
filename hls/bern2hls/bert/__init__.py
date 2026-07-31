"""Front-end 3: TinyBERT encoder blocks with Bernstein activations.

Not a full .pt -> HLS compiler: attention, softmax and LayerNorm are
hand-tuned kernels kept as a fixed library. What is generated is the part the
study varies — the activation, its ROMs, the dimensions, and the weight
streams — so the comparison across activations is reproducible.
"""
