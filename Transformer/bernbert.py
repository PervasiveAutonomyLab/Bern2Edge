"""
Clean, self-contained Bernstein-FFN TinyBERT for hardware synthesis.

`BernBertForSequenceClassification` wraps a standard TinyBERT4 classifier but
replaces each chosen transformer layer's GeLU FFN with an explicit
`BernsteinFFN` module. Unlike the training scripts (which monkey-patch a closure
and keep the dead GeLU weights in the state_dict), this module:

  * exposes the Bernstein FFN math explicitly in `BernsteinFFN.forward`,
  * registers the FFNs as proper submodules (clean names: `bern_ffns.{i}.*`),
  * removes the original GeLU FFN parameters (intermediate/output.dense) so the
    saved weights contain nothing dead.

Per replaced layer the FFN is:   312 --Linear--> H --Bernstein(deg)--> H --Linear--> 312
followed by  dropout, residual add, and LayerNorm (carried over from the
original BertOutput — these stay because the residual+norm are part of the block).

The novel-for-HW part is `BernsteinFFN.forward` / `BernsteinLayer.forward`
(bern2edge/bernstein.py): per-feature affine normalisation into [0,1] (with
clamp), then a degree-`d` Bernstein polynomial evaluated from `d+1` learned
control points. That polynomial is what becomes a lookup table at inference.
"""

import os
import sys

import torch
import torch.nn as nn

# Repository root on the path so `bern2edge` imports resolve regardless of CWD.
_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_HERE)
for _p in (_HERE, _REPO_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from bern2edge.bernstein import BernsteinLayer  # noqa: E402

MODEL_NAME = "huawei-noah/TinyBERT_General_4L_312D"
HIDDEN_DIM = 312


class BernsteinFFN(nn.Module):
    """Drop-in replacement for one TinyBERT FFN block (incl. residual+LayerNorm).

    forward(attention_output) -> layer_output, matching BertLayer.feed_forward_chunk.
    """

    def __init__(self, hidden: int, degree: int, hidden_dim: int = HIDDEN_DIM,
                 dropout: float = 0.0, ln_eps: float = 1e-12) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.hidden = hidden
        self.degree = degree
        self.fc1 = nn.Linear(hidden_dim, hidden)          # 312 -> H
        self.bern = BernsteinLayer([hidden], degree)       # per-feature polynomial
        self.fc2 = nn.Linear(hidden, hidden_dim)           # H -> 312
        self.dropout = nn.Dropout(dropout)
        self.ln = nn.LayerNorm(hidden_dim, eps=ln_eps)

    def forward(self, attention_output: torch.Tensor) -> torch.Tensor:
        h = self.fc1(attention_output)
        h = self.bern(h)
        h = self.fc2(h)
        h = self.dropout(h)
        return self.ln(h + attention_output)

    @torch.no_grad()
    def load_from_fc_state(self, fc_state: dict, ln_state: dict) -> None:
        """Copy weights from a training-time FCModel state_dict (`fc{i}` key of a
        Stage-3 checkpoint) and the original BertOutput LayerNorm state."""
        self.fc1.weight.copy_(fc_state["net.0.weight"])
        self.fc1.bias.copy_(fc_state["net.0.bias"])
        self.fc2.weight.copy_(fc_state["net.2.weight"])
        self.fc2.bias.copy_(fc_state["net.2.bias"])
        bern_state = {k.split("net.1.")[1]: v for k, v in fc_state.items()
                      if k.startswith("net.1.")}
        self.bern.load_state_dict(bern_state)
        self.ln.weight.copy_(ln_state["weight"])
        self.ln.bias.copy_(ln_state["bias"])


class GeluFFN(nn.Module):
    """Clean small-GeLU FFN block — the exp14 control (same shape as BernsteinFFN
    but a plain GeLU activation instead of the polynomial). Used for the smaller
    GeLU substituted variants (e.g. 312 -> 312 -> 312), NOT the original h=1200
    teacher FFN.

    forward(attention_output) -> layer_output, matching BertLayer.feed_forward_chunk.
    """

    def __init__(self, hidden: int, hidden_dim: int = HIDDEN_DIM,
                 dropout: float = 0.0, ln_eps: float = 1e-12) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.hidden = hidden
        self.fc1 = nn.Linear(hidden_dim, hidden)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.ln = nn.LayerNorm(hidden_dim, eps=ln_eps)

    def forward(self, attention_output: torch.Tensor) -> torch.Tensor:
        h = self.fc1(attention_output)
        h = self.act(h)
        h = self.fc2(h)
        h = self.dropout(h)
        return self.ln(h + attention_output)

    @torch.no_grad()
    def load_from_fc_state(self, fc_state: dict, ln_state: dict) -> None:
        """Copy from a training-time FCModel(gelu) state (net.0=fc1, net.2=fc2;
        net.1 is the parameter-free GELU)."""
        self.fc1.weight.copy_(fc_state["net.0.weight"])
        self.fc1.bias.copy_(fc_state["net.0.bias"])
        self.fc2.weight.copy_(fc_state["net.2.weight"])
        self.fc2.bias.copy_(fc_state["net.2.bias"])
        self.ln.weight.copy_(ln_state["weight"])
        self.ln.bias.copy_(ln_state["bias"])


def _identity_ffn(hidden_dim: int) -> nn.Module:
    """A no-op stand-in so HF BertLayer keeps its attribute but holds no params."""
    return nn.Identity()


class BernBertForSequenceClassification(nn.Module):
    """TinyBERT4 sequence classifier with Bernstein FFNs on selected layers."""

    def __init__(self, replaced_layers, hidden: int, degree: int,
                 num_labels: int = 2, act: str = "bern") -> None:
        super().__init__()
        from transformers.models.bert.modeling_bert import BertForSequenceClassification
        self.replaced_layers = list(replaced_layers)
        self.hidden = hidden
        self.degree = degree
        self.act = act

        self.bert = BertForSequenceClassification.from_pretrained(
            MODEL_NAME, num_labels=num_labels, ignore_mismatched_sizes=True)

        # The replaced FFN is either a Bernstein polynomial FFN (default) or the
        # exp14 GeLU control of the same width. The submodule list keeps the name
        # `bern_ffns` for state_dict compatibility regardless of activation.
        if act == "gelu":
            self.bern_ffns = nn.ModuleList([
                GeluFFN(hidden) for _ in self.replaced_layers])
        else:
            self.bern_ffns = nn.ModuleList([
                BernsteinFFN(hidden, degree) for _ in self.replaced_layers])

        # Wire each Bernstein FFN into its layer and drop the dead GeLU weights.
        for slot, li in enumerate(self.replaced_layers):
            layer = self.bert.bert.encoder.layer[li]
            ffn = self.bern_ffns[slot]
            # Plain-function wiring (NOT an nn.Module attribute) avoids
            # double-registering the FFN under the bert sub-tree.
            layer.feed_forward_chunk = (lambda f: (lambda x: f(x)))(ffn)
            layer.intermediate = _identity_ffn(HIDDEN_DIM)
            layer.output = _identity_ffn(HIDDEN_DIM)

    def forward(self, input_ids, attention_mask=None, token_type_ids=None):
        return self.bert(input_ids=input_ids, attention_mask=attention_mask,
                         token_type_ids=token_type_ids).logits

    # ── construction from a training-time Stage-3 checkpoint ────────────────
    @classmethod
    def from_stage3_checkpoint(cls, ckpt_path: str, map_location="cpu"):
        """Build from a Stage-3 / exp12c / exp13d_v2 checkpoint.

        Such a checkpoint is {"bert": <full classifier state>, "fc{i}": <FCModel
        state>, ...}. The original BertOutput LayerNorm weights are read from the
        "bert" state; everything else FFN comes from the fc states.
        """
        ckpt = torch.load(ckpt_path, map_location=map_location)
        assert "bert" in ckpt, "expected a Stage-3 checkpoint with a 'bert' key"
        fc_keys = sorted(k for k in ckpt if k.startswith("fc"))
        replaced = [int(k[2:]) for k in fc_keys]

        # Infer act/hidden/degree from the first FCModel's shapes. A Bernstein
        # FCModel has net.1.bern_coeffs; a GeLU control does not (net.1 is GELU).
        f0 = ckpt[fc_keys[0]]
        hidden = f0["net.0.weight"].shape[0]
        is_bern = "net.1.bern_coeffs" in f0
        act = "bern" if is_bern else "gelu"
        degree = f0["net.1.bern_coeffs"].shape[-1] - 1 if is_bern else None

        model = cls(replaced_layers=replaced, hidden=hidden,
                    degree=degree if is_bern else 0, act=act)

        # Load all the BERT weights EXCEPT the FFN params of the replaced layers
        # (those layers are now Identity). FFN params of NON-replaced layers are
        # kept — they still run the original GeLU FFN.
        dead_prefixes = tuple(
            f"bert.encoder.layer.{li}.{sub}"
            for li in replaced for sub in ("intermediate.", "output."))
        bert_state = {k: v for k, v in ckpt["bert"].items()
                      if not k.startswith(dead_prefixes)}
        missing, unexpected = model.bert.load_state_dict(bert_state, strict=False)
        # Acceptable "missing" keys are only the Identity-replaced FFN params.
        bad = [m for m in missing if not m.startswith(dead_prefixes)]
        assert not bad, f"unexpected missing keys: {bad[:8]}"

        for slot, li in enumerate(replaced):
            ln_state = {
                "weight": ckpt["bert"][f"bert.encoder.layer.{li}.output.LayerNorm.weight"],
                "bias":   ckpt["bert"][f"bert.encoder.layer.{li}.output.LayerNorm.bias"],
            }
            model.bern_ffns[slot].load_from_fc_state(ckpt[fc_keys[slot]], ln_state)

        return model


def param_summary(model: "BernBertForSequenceClassification") -> dict:
    total = sum(p.numel() for p in model.parameters())
    ffn = sum(p.numel() for ffn in model.bern_ffns for p in ffn.parameters())
    return {"total_params": total, "bern_ffn_params": ffn,
            "replaced_layers": model.replaced_layers,
            "hidden": model.hidden, "degree": model.degree}
