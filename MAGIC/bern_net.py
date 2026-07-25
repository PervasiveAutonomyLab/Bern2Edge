"""Self-contained Bernstein student network + certified interval propagation (MAGIC).

Local model core for the MAGIC experiments. `BernsteinLayer` + `FCModel` here are the
exact classes that produced the shipped `student_model_weights/*.pth` (state_dict-compatible)
and additionally expose `subinterval_bounds`, the Bernstein interval-bound-propagation
primitive that the repo-wide `bernstein.py` does not carry. Kept local (rather than importing
the shared `bernstein.py`/`models.py`) so certification is self-contained and the shipped
weights reproduce their exact numbers. Helpers:
  - load_student(ckpt_path)      : rebuild + load a saved_models_kd/*.pth student
  - nn_predict(model, Z)         : forward + argmax labels on normalized inputs
  - nn_certify_box(model, z_lo, z_hi): SOUND interval propagation of an input box
        [z_lo, z_hi] (normalized space) -> per-point certified label {0,1,-1=INC}.
        Linear layers via interval arithmetic; BernsteinLayer via subinterval_bounds
        (clamping z into input_bounds first, the NaN/soundness fix); final Linear is
        margin-projected (binary) for tightness, mirroring certified_tiling.propagate_box_bern.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


# ===================== Bernstein activation layer (+ interval bounds) =====================
class BernsteinLayer(nn.Module):

  def __init__(self, in_shape, degree: int):
    super().__init__()
    self._clamp_total = 0
    self._clamp_count = 0
    self._xnorm_min = float("inf")
    self._xnorm_max = float("-inf")
    self._xnorm_sum = 0.0
    self._xnorm_count = 0
    self.range_penalty = torch.tensor(0.0)
    self.pre_act_penalty = torch.tensor(0.0)

    self.degree = degree
    self.in_shape = in_shape

    _basis_indices_tensor = torch.arange(degree + 1).reshape((-1, degree + 1))
    _deg_tensor = torch.tensor([degree]).reshape(-1, 1)
    nCk_tensor = self.binom(_deg_tensor, _basis_indices_tensor)

    input_bounds = torch.zeros((*in_shape, 2))

    self.register_buffer("input_bounds", input_bounds)
    self.register_buffer("_basis_indices", _basis_indices_tensor)
    self.register_buffer("_deg_tensor", _deg_tensor)
    self.register_buffer("nCk", nCk_tensor)

    self.use_bounds = True

    ramp = torch.linspace(0, 1, degree + 1)
    bern_coeffs = ramp.expand(*in_shape, degree + 1).clone()
    bern_coeffs += torch.randn_like(bern_coeffs) * 0.01
    self.bern_coeffs = nn.Parameter(bern_coeffs)

  def reset_xnorm_stats(self):
      self._xnorm_min = float("inf")
      self._xnorm_max = float("-inf")
      self._xnorm_sum = 0.0
      self._xnorm_count = 0

  def get_xnorm_stats(self):
      if self._xnorm_count == 0:
          return 0.0, 0.0, 0.0
      mean = self._xnorm_sum / self._xnorm_count
      return self._xnorm_min, mean, self._xnorm_max

  @property
  def bern_bounds(self):
    lb, _ = self.bern_coeffs.min(axis=-1, keepdim=True)
    ub, _ = self.bern_coeffs.max(axis=-1, keepdim=True)
    return torch.concat((lb, ub), dim=-1)

  def binom(self, n, k):
    nCk = torch.lgamma(n + 1) - torch.lgamma(k + 1) - torch.lgamma(n - k + 1)
    nCk = torch.exp(nCk)
    nCk = torch.floor(nCk + 0.5)
    return nCk

  def bern_basis(self, x):
    y = x.unsqueeze(-1)
    basis = (
        self.nCk
        * (y) ** self._basis_indices
        * (1 - y) ** (self._deg_tensor - self._basis_indices)
    )
    return basis

  def subinterval_bounds(self, bounds):
        """Compute Bernstein coeffs for interval [alpha,beta]"""
        in_bounds = self.input_bounds.unsqueeze(0)
        bounds = (bounds - in_bounds[..., 0:1]) / (
            in_bounds[..., 1:] - in_bounds[..., 0:1]
        )
        alpha = bounds[..., 0].unsqueeze(-1).clamp(0.0, 1.0)
        beta = bounds[..., 1].unsqueeze(-1).clamp(0.0, 1.0)
        zero_to_beta_coeffs = torch.zeros(
            bounds.shape[0],
            *self.in_shape,
            self.degree + 1,
            self.degree + 2,
            device=alpha.device,
        )
        zero_to_beta_coeffs[..., 0, :-1] = self.bern_coeffs
        for i in range(1, zero_to_beta_coeffs.shape[-2]):
            zero_to_beta_coeffs[..., i, 1:] = (1 - beta) * zero_to_beta_coeffs[
                ..., i - 1, :-1
            ].clone() + beta * zero_to_beta_coeffs[..., i - 1, 1:].clone()

        zero_to_beta_coeffs = zero_to_beta_coeffs[..., :-1].diagonal(dim1=-2, dim2=-1)

        gamma = alpha / beta
        alpha_to_beta_coeffs = torch.zeros(
            bounds.shape[0],
            *self.in_shape,
            self.degree + 1,
            self.degree + 2,
            device=alpha.device,
        )
        alpha_to_beta_coeffs[..., 0, :-1] = zero_to_beta_coeffs
        for i in range(1, alpha_to_beta_coeffs.shape[-2]):
            alpha_to_beta_coeffs[..., i, 1:] = (1 - gamma) * alpha_to_beta_coeffs[
                ..., i - 1, :-1
            ].clone() + gamma * alpha_to_beta_coeffs[..., i - 1, 1:].clone()

        new_coeffs_lb_ub = alpha_to_beta_coeffs[..., -2]
        lb = new_coeffs_lb_ub.min(axis=-1, keepdim=True)[0]
        ub = new_coeffs_lb_ub.max(axis=-1, keepdim=True)[0]
        return torch.concat((lb, ub), -1)

  def forward(self, x):
    if self.training:
        pre_act_penalty = (x ** 2).mean()
        self.pre_act_penalty = pre_act_penalty

    if self.use_bounds:
        x_norm = (x - self.input_bounds[..., 0]) / (
            self.input_bounds[..., 1] - self.input_bounds[..., 0] + 1e-8
        )
    else:
        x_norm = x

    if self.training:
        with torch.no_grad():
            clamped = (x_norm < 0.0) | (x_norm > 1.0)
            self._clamp_total += clamped.sum().item()
            self._clamp_count += clamped.numel()

    if self.training:
        with torch.no_grad():
            self._xnorm_min = min(self._xnorm_min, x_norm.min().item())
            self._xnorm_max = max(self._xnorm_max, x_norm.max().item())
            self._xnorm_sum += x_norm.sum().item()
            self._xnorm_count += x_norm.numel()
    if self.training:
      below = F.relu(-x_norm)
      above = F.relu(x_norm - 1.0)
      self.range_penalty = (below + above).mean()
    else:
        self.range_penalty = x_norm.new_zeros(())
        self.pre_act_penalty = x_norm.new_zeros(())

    x = torch.clamp(x_norm, 0.0, 1.0)
    basis = self.bern_basis(x)
    out = basis * self.bern_coeffs
    out = out.sum(axis=-1)
    return out


# ===================== Compact Bernstein/ReLU student network =====================
class FCModel(nn.Module):
    def __init__(self, layer_sizes, degree, act="bern", last_bern=False, dropout=0.0):
        super().__init__()
        layers = []
        for i, l_size in enumerate(layer_sizes[:-1]):
            layers.append(nn.Linear(layer_sizes[i], layer_sizes[i + 1]))
            if i < (len(layer_sizes) - 2):
                if act == "bern":
                    layers.append(BernsteinLayer([layer_sizes[i + 1]], degree))
                else:
                    layers.append(nn.ReLU())
                if dropout > 0.0:
                    layers.append(nn.Dropout(p=dropout))
        if last_bern:
            layers.append(BernsteinLayer([layer_sizes[-1]], 1))
        self.layers = nn.ModuleList(layers)
        self.net = nn.Sequential(*self.layers)

    def get_bern_layers(self):
        return [m for m in self.layers if isinstance(m, BernsteinLayer)]

    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.net(x)


# ===================== helpers (new) =====================
def load_student(ckpt_path):
    """Rebuild + load a saved_models_kd/*.pth student. Returns (model.eval(), raw_dict)."""
    raw = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    deg = raw.get("degree", None)
    model = FCModel(
        layer_sizes=raw["arch"],
        degree=deg if deg is not None else 3,
        act=raw.get("activation", "bern"),
        last_bern=raw.get("last_bern", False),
        dropout=0.0,
    )
    model.load_state_dict(raw["state_dict"])
    model.eval()
    return model, raw


@torch.no_grad()
def nn_predict(model, Z):
    """Forward + argmax labels on normalized inputs Z (np array [N, D])."""
    import numpy as np
    logits = model(torch.as_tensor(np.asarray(Z), dtype=torch.float32))
    return logits.argmax(dim=1).numpy()


@torch.no_grad()
def nn_certify_box(model, z_lo, z_hi):
    """SOUND certified label for each input box [z_lo, z_hi] (np arrays [N, D]).

    Returns int array in {0, 1, -1}: the class provably output by the network for the
    whole box, or -1 (INConclusive). Linear layers use interval arithmetic; Bernstein
    layers use subinterval_bounds after clamping z into input_bounds (eps margin avoids
    the divide-by-zero / NaN); nn.ReLU layers use exact per-neuron [relu(lo), relu(hi)]
    (matched IBP for the ReLU baseline); the final Linear is margin-projected (binary).
    """
    import numpy as np
    lo = torch.as_tensor(np.asarray(z_lo), dtype=torch.float32)
    hi = torch.as_tensor(np.asarray(z_hi), dtype=torch.float32)
    layers = list(model.layers)
    *body, last = layers
    for layer in body:
        if isinstance(layer, nn.Linear):
            W, b = layer.weight, layer.bias
            Wp, Wn = W.clamp(min=0), W.clamp(max=0)
            lo, hi = lo @ Wp.t() + hi @ Wn.t() + b, hi @ Wp.t() + lo @ Wn.t() + b
        elif isinstance(layer, BernsteinLayer):
            ib = layer.input_bounds
            eps = (ib[:, 1] - ib[:, 0]) * 1e-7
            clo = torch.clamp(lo, min=ib[:, 0], max=ib[:, 1] - eps)
            chi = torch.clamp(hi, min=ib[:, 0] + eps, max=ib[:, 1])
            hb = layer.subinterval_bounds(torch.stack([clo, chi], dim=-1))  # [N,H,2]
            lo, hi = hb[..., 0], hb[..., 1]
        elif isinstance(layer, nn.ReLU):
            # exact per-neuron range over the box, matched to Bernstein's per-neuron
            # subinterval_bounds (both ignore the same cross-neuron correlations).
            lo, hi = lo.clamp(min=0), hi.clamp(min=0)
        elif isinstance(layer, nn.Dropout):
            pass
        else:
            raise TypeError(f"unsupported layer in IBP: {type(layer)}")
    assert isinstance(last, nn.Linear) and last.out_features == 2, \
        "nn_certify_box expects a binary Linear output head"
    W, b = last.weight, last.bias
    Wm = W[1] - W[0]                  # margin = logit1 - logit0
    bm = b[1] - b[0]
    Wmp, Wmn = Wm.clamp(min=0), Wm.clamp(max=0)
    m_lo = lo @ Wmp + hi @ Wmn + bm
    m_hi = hi @ Wmp + lo @ Wmn + bm
    cert = torch.full((lo.shape[0],), -1, dtype=torch.long)
    cert[m_lo > 0] = 1                # margin certainly > 0 -> class 1
    cert[m_hi < 0] = 0                # margin certainly < 0 -> class 0
    return cert.numpy()
