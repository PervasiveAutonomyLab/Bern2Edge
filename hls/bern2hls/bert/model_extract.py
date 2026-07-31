"""Losslessly slice one encoder layer from a Bern2Edge Transformer checkpoint."""

from __future__ import annotations

from pathlib import Path

import numpy as np

ATTENTION_KEYS = (
    "self.query.weight", "self.query.bias",
    "self.key.weight", "self.key.bias",
    "self.value.weight", "self.value.bias",
    "output.dense.weight", "output.dense.bias",
    "output.LayerNorm.weight", "output.LayerNorm.bias",
)


def _numpy(tensor):
    return tensor.detach().cpu().numpy()


def extract_layer(checkpoint_path, output_path, layer=0):
    """Write the torch-free layer bundle consumed by the HLS compiler.

    Supports the four clean ``release_*.pt`` checkpoints and the
    ``teacher_gelu_9037.pt`` checkpoint shipped under ``Transformer/models``.
    Every written array is checked against its source tensor after the NPZ is
    reopened.
    """
    import torch

    checkpoint_path = Path(checkpoint_path)
    output_path = Path(output_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    teacher = "state_dict" not in checkpoint
    state = checkpoint["bert"] if teacher else checkpoint["state_dict"]
    out = {}

    if teacher:
        prefix = f"bert.encoder.layer.{layer}."
        mapping = {
            "fc1.weight": prefix + "intermediate.dense.weight",
            "fc1.bias": prefix + "intermediate.dense.bias",
            "fc2.weight": prefix + "output.dense.weight",
            "fc2.bias": prefix + "output.dense.bias",
            "ln.weight": prefix + "output.LayerNorm.weight",
            "ln.bias": prefix + "output.LayerNorm.bias",
        }
        attention_prefix = prefix + "attention."
    else:
        meta = checkpoint.get("meta", {})
        replaced = meta.get("replaced_layers", [])
        if replaced and layer not in replaced:
            raise ValueError(
                f"layer {layer} is not replaced in {checkpoint_path.name}: {replaced}"
            )
        slot = replaced.index(layer) if replaced else layer
        prefix = f"bern_ffns.{slot}."
        mapping = {
            "fc1.weight": prefix + "fc1.weight",
            "fc1.bias": prefix + "fc1.bias",
            "fc2.weight": prefix + "fc2.weight",
            "fc2.bias": prefix + "fc2.bias",
            "ln.weight": prefix + "ln.weight",
            "ln.bias": prefix + "ln.bias",
        }
        if meta.get("act") == "bern":
            mapping.update({
                "bern.input_bounds": prefix + "bern.input_bounds",
                "bern.bern_coeffs": prefix + "bern.bern_coeffs",
            })
        attention_prefix = f"bert.bert.encoder.layer.{layer}.attention."

    for output_name, state_name in mapping.items():
        if state_name not in state:
            raise KeyError(f"{checkpoint_path}: missing {state_name}")
        out[output_name] = _numpy(state[state_name])
    for suffix in ATTENTION_KEYS:
        state_name = attention_prefix + suffix
        if state_name not in state:
            raise KeyError(f"{checkpoint_path}: missing {state_name}")
        out["attn." + suffix] = _numpy(state[state_name])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, **out)
    with np.load(output_path) as saved:
        for name, array in out.items():
            if not np.array_equal(saved[name], array):
                raise AssertionError(f"NPZ round-trip mismatch for {name}")
    return output_path
