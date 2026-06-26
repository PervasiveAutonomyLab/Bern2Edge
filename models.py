"""
Model definitions for the Bern2Edge Cover Type experiments.

  * TeacherMLP : a standard ReLU MLP used as the knowledge-distillation teacher.
  * FCModel    : the compact student network whose hidden activations are either
                 ReLU or a learnable Bernstein-polynomial layer (`BernsteinLayer`).

Both produce raw logits (no softmax); use them with `nn.CrossEntropyLoss`.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from bernstein import BernsteinLayer


class TeacherMLP(nn.Module):
    """
    Teacher network:  (in) -> [Linear -> ReLU -> Dropout] x L -> Linear -> (logits)
    """

    def __init__(self, d_in, d_layers, dropout, d_out):
        super().__init__()
        # Accept either a single dropout value or one value per hidden layer.
        if isinstance(dropout, (float, int)):
            dropouts = [float(dropout)] * len(d_layers)
        else:
            dropouts = list(dropout)
            assert len(dropouts) == len(d_layers)
        act_fn = nn.ReLU

        layers = []
        prev = d_in
        for width, p in zip(d_layers, dropouts):
            layers.append(nn.Linear(prev, width, bias=True))
            layers.append(act_fn())
            layers.append(nn.Dropout(p))
            prev = width

        self.blocks = nn.Sequential(*layers)
        self.head = nn.Linear(prev, d_out, bias=True)

    def forward(self, x):
        x = self.blocks(x)
        return self.head(x)  # logits


class AdultTeacherMLP(nn.Module):
    """
    Knowledge-distillation teacher for the ordinal-encoded Adult dataset.

    A flat  [Linear -> ReLU -> (Dropout)] x L -> Linear(., n_classes)  stack.
    Kept distinct from TeacherMLP because the published Adult teacher checkpoint
    stores its weights under ``net.*`` keys (this Sequential layout); it
    reproduces the original ``adult_v2_kd/teacher.py:MLP`` exactly so that
    ``adult_teacher_ordinal.pt`` loads without remapping.

    Args:
        input_dim : number of input features (14 for ordinal Adult).
        d_layers  : hidden widths, e.g. [42, 503, 503, 503, 111].
        dropout   : dropout probability between hidden layers (0.0 = off).
        n_classes : number of output logits (2 for Adult).
    """

    def __init__(self, input_dim, d_layers, dropout=0.0, n_classes=2):
        super().__init__()
        layers = []
        in_dim = input_dim
        for out_dim in d_layers:
            layers.append(nn.Linear(in_dim, out_dim))
            layers.append(nn.ReLU())
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
            in_dim = out_dim
        layers.append(nn.Linear(in_dim, n_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)  # logits


class FCModel(nn.Module):
    """
    Compact student network used to compare Bernstein vs. ReLU activations.

    Args:
        layer_sizes : full width list, e.g. [54, 64, 32, 7] (input ... output).
        degree      : Bernstein polynomial degree (ignored when act != "bern").
        act         : "bern" for BernsteinLayer activations, otherwise ReLU.
        last_bern   : if True, append a degree-1 BernsteinLayer on the output.
        dropout     : dropout probability between hidden layers (0.0 = off).
    """

    def __init__(self, layer_sizes, degree, act="bern", last_bern=False, dropout=0.0):
        super().__init__()

        layers = []
        for i, _ in enumerate(layer_sizes[:-1]):
            # Linear layer mapping width[i] -> width[i+1].
            layers.append(nn.Linear(layer_sizes[i], layer_sizes[i + 1]))

            # Insert an activation after every layer except the final (output) one.
            if i < (len(layer_sizes) - 2):
                if act == "bern":
                    layers.append(BernsteinLayer([layer_sizes[i + 1]], degree))
                else:
                    layers.append(nn.ReLU())
                if dropout > 0.0:
                    layers.append(nn.Dropout(p=dropout))

        if last_bern:
            # Optional degree-1 Bernstein layer on the logits.
            layers.append(BernsteinLayer([layer_sizes[-1]], 1))

        self.layers = nn.ModuleList(layers)
        self.net = nn.Sequential(*self.layers)

    def get_bern_layers(self):
        """Return all BernsteinLayer instances in forward order."""
        return [m for m in self.layers if isinstance(m, BernsteinLayer)]

    def set_frozen_up_to(self, k):
        """
        Enable input-bound normalization for Bernstein layers 0..k (inclusive);
        later layers run without bounds. Used by progressive calibration.
        """
        bern_layers = self.get_bern_layers()
        for i, bl in enumerate(bern_layers):
            bl.use_bounds = (i <= k)

    @torch.no_grad()
    def calibrate_one_bern_layer_from_data(
        self, loader, device, layer_idx, max_batches=200, p_lo=0.01, p_hi=0.99, min_width=0.5
    ):
        """
        Estimate per-neuron input bounds for a single Bernstein layer from data
        (using empirical quantiles), then enable bound normalization on it.
        Other layers are left unchanged.
        """
        self.eval()
        bern_layers = self.get_bern_layers()
        if len(bern_layers) == 0:
            self.train()
            return

        assert 0 <= layer_idx < len(bern_layers), \
            f"layer_idx out of range (0..{len(bern_layers) - 1})"

        target = bern_layers[layer_idx]
        # Disable bounds on the target while calibrating so we observe raw inputs.
        target.use_bounds = False

        low = None
        high = None

        def hook(module, inp, out):
            nonlocal low, high
            x = inp[0].detach()
            lo = torch.quantile(x, p_lo, dim=0)
            hi = torch.quantile(x, p_hi, dim=0)
            # Accumulate conservatively (widest range) across batches.
            if low is None:
                low, high = lo, hi
            else:
                low = torch.minimum(low, lo)
                high = torch.maximum(high, hi)

        handle = target.register_forward_hook(hook)
        batches = 0
        for batch in loader:
            data = batch[0].to(device)
            _ = self(data)  # triggers the hook at the target layer
            batches += 1
            if batches >= max_batches:
                break
        handle.remove()

        low = low.to(device)
        high = high.to(device)

        # Ensure low < high everywhere (swap any inverted quantiles).
        swap_mask = low > high
        low[swap_mask], high[swap_mask] = high[swap_mask].clone(), low[swap_mask].clone()

        # Enforce a minimum (and a sane maximum) per-neuron width.
        width = high - low
        width = torch.clamp(width, min=min_width)
        max_width = 40.0   # no neuron should need a range wider than 40 std devs
        width = width.clamp(max=max_width)
        mid = 0.5 * (high + low)
        low = mid - 0.5 * width
        high = mid + 0.5 * width

        # Write the calibrated bounds and turn on bound normalization.
        target.input_bounds.data.copy_(torch.stack([low, high], dim=-1))
        target.use_bounds = True

        mean_width = (high - low).mean().item()
        print(
            f"[Calibrated Bernstein layer {layer_idx}] "
            f"low ∈ [{low.min():.4f}, {low.max():.4f}], "
            f"high ∈ [{high.min():.4f}, {high.max():.4f}], "
            f"mean width = {mean_width:.4f}"
        )
        self.train()

    def range_penalty(self):
        """Mean out-of-bounds penalty across all Bernstein layers (0 if none)."""
        bern_layers = self.get_bern_layers()
        if len(bern_layers) == 0:
            return torch.zeros((), device=next(self.parameters()).device)
        pen = torch.zeros((), device=next(self.parameters()).device)
        for bl in bern_layers:
            pen = pen + bl.range_penalty
        return pen / len(bern_layers)

    def forward(self, x):
        x = x.view(x.size(0), -1)   # flatten input
        return self.net(x)
