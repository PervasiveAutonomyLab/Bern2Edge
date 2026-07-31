"""Model classes, checkpoint loading and automatic config extraction.

BernsteinLayer/FCModel are the canonical copies (lifted from
adult-HW/sweep_test_all/generate_all_models.py). Training checkpoints save
state_dicts with both `layers.*` and `net.*` key prefixes (the SW-side FCModel
registers the same submodules under a ModuleList and a Sequential); loading
with strict=False into this FCModel picks up `layers.*` and ignores the rest,
yielding identical weights.
"""

import os
import re

import numpy as np
import torch
import torch.nn as nn


class BernsteinLayer(nn.Module):
    def __init__(self, in_shape, degree: int):
        super().__init__()
        self.degree = degree
        self.in_shape = in_shape
        _basis_indices_tensor = torch.arange(degree + 1).reshape((-1, degree + 1))
        _deg_tensor = torch.tensor([degree]).reshape(-1, 1)
        nCk_tensor = self._binom(_deg_tensor, _basis_indices_tensor)
        input_bounds = torch.zeros((*in_shape, 2))
        self.register_buffer("input_bounds", input_bounds)
        self.register_buffer("_basis_indices", _basis_indices_tensor)
        self.register_buffer("_deg_tensor", _deg_tensor)
        self.register_buffer("nCk", nCk_tensor)
        self.use_bounds = False
        bern_coeffs = torch.empty(*in_shape, degree + 1)
        nn.init.xavier_uniform_(bern_coeffs)
        self.bern_coeffs = nn.Parameter(bern_coeffs)

    @staticmethod
    def _binom(n, k):
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

    def forward(self, x):
        if self.use_bounds:
            x = (x - self.input_bounds[..., 0]) / (
                self.input_bounds[..., 1] - self.input_bounds[..., 0] + 1e-8)
            x = torch.clamp(x, 0.0, 1.0)
        else:
            x = torch.clamp(x, 0.0, 1.0)
        basis = self.bern_basis(x)
        out = basis * self.bern_coeffs
        out = out.sum(axis=-1)
        return out


class FCModel(nn.Module):
    def __init__(self, layer_sizes, degree, act="bern", input_bounds=None, last_bern=False):
        super().__init__()
        if input_bounds is None:
            input_bounds = torch.cat((torch.zeros(layer_sizes[0], 1),
                                      torch.ones(layer_sizes[0], 1)), dim=-1)
        self.register_buffer("input_bounds", input_bounds.detach().clone())
        layers = []
        for i, l_size in enumerate(layer_sizes[:-1]):
            layers.append(nn.Linear(layer_sizes[i], layer_sizes[i + 1]))
            if i < (len(layer_sizes) - 2):
                if act == "bern":
                    layers.append(BernsteinLayer(
                        in_shape=(layer_sizes[i + 1],), degree=degree))
                elif act == "relu":
                    layers.append(nn.ReLU())
        if last_bern:
            layers.append(BernsteinLayer(
                in_shape=(layer_sizes[-1],), degree=degree))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


# =====================================================================
# Automatic config extraction (checkpoint metadata first, filename fallback)
# =====================================================================

# Lifted from adult-SW/sweep_test_sw/sweep_test.py
FILENAME_RE = re.compile(
    r"^kd_fc_(?P<arch>[\dx]+)"
    r"_(?P<act>bern|relu)"
    r"_deg(?P<deg>[^_]+)"
    r"_alpha(?P<alpha>[^_]+)"
    r"_T(?P<T>[^_]+)"
    r"_lr(?P<lr>[^_]+)"
    r"_wd(?P<wd>[^_]+)"
    r"_seed(?P<seed>\d+)\.pth$"
)


def parse_filename(fname):
    """Parse a kd_fc_* checkpoint filename into {layer_sizes, act, degree}, or None."""
    m = FILENAME_RE.match(os.path.basename(fname))
    if m is None:
        return None
    d = m.groupdict()
    return {
        'layer_sizes': list(map(int, d['arch'].split('x'))),
        'act': d['act'],
        'degree': int(d['deg']) if d['deg'] != 'NA' else None,
    }


def infer_from_state_dict(sd):
    """Infer {layer_sizes, act, degree} from state_dict tensor shapes alone."""
    prefix = 'layers.' if any(k.startswith('layers.') for k in sd) else 'net.'
    lin = {}
    bern_degs = []
    pat = re.compile(rf'^{re.escape(prefix)}(\d+)\.weight$')
    for k, v in sd.items():
        m = pat.match(k)
        if m and v.dim() == 2:
            lin[int(m.group(1))] = tuple(v.shape)
        if k.startswith(prefix) and k.endswith('.bern_coeffs'):
            bern_degs.append(v.shape[-1] - 1)
    if not lin:
        return None
    idxs = sorted(lin)
    layer_sizes = [lin[idxs[0]][1]] + [lin[i][0] for i in idxs]
    act = 'bern' if bern_degs else 'relu'
    return {'layer_sizes': layer_sizes, 'act': act,
            'degree': bern_degs[0] if bern_degs else None}


def extract_model_config(pth_path, ckpt=None):
    """Derive {layer_sizes, act, degree} for a checkpoint.

    Priority: (1) metadata embedded in the checkpoint dict (arch/activation/degree,
    written by the KD training scripts), (2) the kd_fc_* filename convention,
    (3) inference from state_dict tensor shapes. Raises ValueError if all fail.
    A handful of checkpoints carry a malformed activation field (e.g. 'bern,relu'
    from a training-script logging bug), hence the validity check before trusting it.
    """
    if ckpt is None:
        ckpt = torch.load(pth_path, map_location='cpu')
    if (isinstance(ckpt, dict) and 'arch' in ckpt
            and ckpt.get('activation') in ('bern', 'relu')):
        act = ckpt['activation']
        degree = ckpt.get('degree') if act == 'bern' else None
        return {
            'layer_sizes': list(ckpt['arch']),
            'act': act,
            'degree': int(degree) if degree is not None else None,
        }
    parsed = parse_filename(pth_path)
    if parsed is not None:
        return parsed
    sd = ckpt['state_dict'] if isinstance(ckpt, dict) and 'state_dict' in ckpt else ckpt
    if isinstance(sd, dict):
        inferred = infer_from_state_dict(sd)
        if inferred is not None:
            return inferred
    raise ValueError(
        f"Cannot determine model config for {pth_path}: no usable embedded metadata, "
        f"filename does not match the kd_fc_* convention, and state_dict shapes "
        f"could not be interpreted.")


def model_display_name(config):
    """Project name in the sweep convention: {act}[_d<deg>]_{arch}."""
    arch = 'x'.join(str(s) for s in config['layer_sizes'])
    if config['act'] == 'bern':
        return f"bern_d{config['degree']}_{arch}"
    return f"relu_{arch}"


def make_cfg_from_pth(pth_path):
    """Build a generator cfg dict (name/layer_sizes/act/degree/file) from a .pth."""
    config = extract_model_config(pth_path)
    return {
        'name': model_display_name(config),
        'layer_sizes': config['layer_sizes'],
        'act': config['act'],
        'degree': config['degree'],
        'file': os.path.basename(pth_path),
        '_dir': os.path.dirname(os.path.abspath(pth_path)),
    }


def check_cfg_against_metadata(cfg, model_path):
    """Cross-check a curated cfg entry against checkpoint-embedded metadata."""
    try:
        meta = extract_model_config(model_path)
    except (ValueError, FileNotFoundError):
        return
    mismatches = []
    if list(meta['layer_sizes']) != list(cfg['layer_sizes']):
        mismatches.append(f"arch {meta['layer_sizes']} != {cfg['layer_sizes']}")
    if meta['act'] != cfg['act']:
        mismatches.append(f"act {meta['act']} != {cfg['act']}")
    if cfg['act'] == 'bern' and meta['degree'] != cfg['degree']:
        mismatches.append(f"degree {meta['degree']} != {cfg['degree']}")
    if mismatches:
        raise ValueError(f"Config/checkpoint mismatch for {cfg['name']} "
                         f"({model_path}): " + '; '.join(mismatches))


# =====================================================================
# Model loading and parameter extraction
# (lifted from *-HW/sweep_test_all/generate_all_models.py)
# =====================================================================

def load_model(cfg, model_dir, input_dim):
    """Load a trained PyTorch model for any architecture."""
    layer_sizes = cfg['layer_sizes']
    act = cfg['act']
    degree = cfg['degree'] if cfg['degree'] else 3  # FCModel needs a degree even for ReLU
    model_path = os.path.join(model_dir, cfg['file'])

    input_bounds = torch.cat((torch.zeros(input_dim, 1), torch.ones(input_dim, 1)), dim=-1)
    model = FCModel(layer_sizes=layer_sizes, degree=degree, act=act,
                    input_bounds=input_bounds, last_bern=False).to('cpu')
    ckpt = torch.load(model_path, map_location='cpu')
    sd = ckpt['state_dict'] if isinstance(ckpt, dict) and 'state_dict' in ckpt else ckpt
    model.load_state_dict(sd, strict=False)
    model.eval()
    if act == 'bern':
        for m in model.modules():
            if isinstance(m, BernsteinLayer):
                m.use_bounds = True
    return model


def extract_params(model, cfg):
    """Extract all layer parameters from a multi-layer model.

    Returns:
      linears: [(W, b), ...] for each nn.Linear
      bern_layers: [BernsteinLayer or None, ...] for each hidden activation
    """
    layers = list(model.layers)
    linears = []
    bern_layers = []

    for layer in layers:
        if isinstance(layer, torch.nn.Linear):
            W = layer.weight.data.numpy()
            b = layer.bias.data.numpy()
            linears.append((W, b))
        elif isinstance(layer, BernsteinLayer):
            bern_layers.append(layer)
        # nn.ReLU has no parameters

    num_hidden = len(cfg['layer_sizes']) - 2
    while len(bern_layers) < num_hidden:
        bern_layers.append(None)

    return linears, bern_layers
