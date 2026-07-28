"""
Shared utilities for the transformer FFN-substitution pipeline (paper TABLE XII).

Everything the three stages have in common: GLUE/SST-2 data loading, FFN I/O
capture via forward hooks, Bernstein bound calibration, teacher loading, FFN
monkey-patching, and logging.

The reference model is TinyBERT4 (huawei-noah/TinyBERT_General_4L_312D):
4 layers, hidden=312, FFN intermediate=1200, GeLU.

The substituted FFN is a plain `bern2edge.models.FCModel` (312 -> H -> 312) built
with the "ramp" Bernstein initialization -- see BERN_INIT below.
"""

import os
import sys
import random
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# Repository root on the path so `bern2edge` imports resolve regardless of CWD.
_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_HERE)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from bern2edge.models import FCModel  # noqa: E402

MODEL_NAME = "huawei-noah/TinyBERT_General_4L_312D"
HIDDEN_DIM = 312  # TinyBERT4 hidden size (FFN in/out width)

# Bernstein coefficient init for this experiment. Unlike the tabular experiments
# (which train from random "xavier" coefficients), the FFN student is trained to
# *match* an existing GeLU FFN, so it starts from a near-identity ramp. This is
# the init the shipped TABLE XII weights were trained with -- do not change it
# without retraining.
BERN_INIT = "ramp"


def models_path(*parts: str) -> str:
    """Absolute path inside Transformer/models/, independent of CWD."""
    return os.path.join(_HERE, "models", *parts)


def results_path(*parts: str) -> str:
    """Absolute path inside Transformer/results/, independent of CWD."""
    return os.path.join(_HERE, "results", *parts)


def require_gpu(what: str) -> None:
    """Training stages need a GPU; rendering and evaluation do not."""
    if not torch.cuda.is_available():
        raise SystemExit(
            f"{what} requires a CUDA GPU (none visible).\n"
            "Reproducing TABLE XII from the shipped weights does not:\n"
            "  python Transformer/make_table_xii.py   # render, seconds\n"
            "  python Transformer/eval_release.py     # recompute accuracy (CPU ok)"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Logging + reproducibility
# ─────────────────────────────────────────────────────────────────────────────

class Tee:
    """Mirror stdout to a log file (line-buffered)."""

    def __init__(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self._f = open(path, "w", buffering=1)
        self._stdout = sys.stdout

    def write(self, data: str) -> None:
        self._stdout.write(data)
        self._f.write(data)

    def flush(self) -> None:
        self._stdout.flush()
        self._f.flush()

    def fileno(self) -> int:
        return self._stdout.fileno()


def set_seed(seed: int = 42) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ─────────────────────────────────────────────────────────────────────────────
# Data
# ─────────────────────────────────────────────────────────────────────────────

def build_tokenizer():
    from transformers.models.bert.tokenization_bert import BertTokenizer
    return BertTokenizer.from_pretrained(MODEL_NAME)


# GLUE task registry: text column key(s), label count, and validation split.
# Single-key tasks are single-sentence; two-key tasks are sentence-pair.
GLUE_TASKS = {
    "sst2": {"keys": ("sentence",),              "num_labels": 2, "val_split": "validation"},
    "mrpc": {"keys": ("sentence1", "sentence2"), "num_labels": 2, "val_split": "validation"},
    "rte":  {"keys": ("sentence1", "sentence2"), "num_labels": 2, "val_split": "validation"},
    "qnli": {"keys": ("question", "sentence"),   "num_labels": 2, "val_split": "validation"},
}


def make_glue_loaders(tokenizer, task: str, batch_size: int, max_length: int,
                      with_labels: bool, num_workers: int = 4,
                      train_subset: Optional[int] = None):
    """Return (train_loader, val_loader) over a GLUE task in GLUE_TASKS.

    with_labels=False  -> isolation stages (only need input_ids/attention_mask).
    with_labels=True   -> classification stage (need labels for CE/eval).
    train_subset       -> cap the train split to the first N examples (used to
                          bound Stage-2 cache memory on large tasks like QNLI);
                          None keeps the full split.

    Tokenization is single-sentence for one-key tasks and sentence-pair for
    two-key tasks (the only data difference between SST-2 and MRPC/RTE/QNLI).
    """
    if task not in GLUE_TASKS:
        raise ValueError(f"unknown GLUE task '{task}'; known: {sorted(GLUE_TASKS)}")
    cfg = GLUE_TASKS[task]
    keys = cfg["keys"]

    from datasets import load_dataset
    ds = load_dataset("glue", task)
    train_split = ds["train"]
    if train_subset is not None:
        n = min(train_subset, len(train_split))
        train_split = train_split.select(range(n))
    val_split = ds[cfg["val_split"]]

    def collate(batch):
        if len(keys) == 1:
            texts = ([b[keys[0]] for b in batch],)
        else:
            texts = ([b[keys[0]] for b in batch], [b[keys[1]] for b in batch])
        enc = tokenizer(
            *texts, max_length=max_length,
            padding="max_length", truncation=True, return_tensors="pt",
        )
        if with_labels:
            enc["labels"] = torch.tensor([b["label"] for b in batch],
                                         dtype=torch.long)
        return enc

    train_loader = DataLoader(train_split, batch_size=batch_size,
                              collate_fn=collate, shuffle=True,
                              num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_split, batch_size=batch_size,
                            collate_fn=collate, shuffle=False,
                            num_workers=num_workers, pin_memory=True)
    return train_loader, val_loader


def make_sst2_loaders(tokenizer, batch_size: int, max_length: int,
                      with_labels: bool, num_workers: int = 4):
    """Backward-compatible SST-2 wrapper around make_glue_loaders."""
    return make_glue_loaders(tokenizer, "sst2", batch_size, max_length,
                             with_labels, num_workers=num_workers)


# ─────────────────────────────────────────────────────────────────────────────
# Model loading
# ─────────────────────────────────────────────────────────────────────────────

def load_general_tinybert(device: torch.device):
    """Frozen general-pretrained TinyBERT (BertModel) — Stage-1 target."""
    from transformers.models.bert.modeling_bert import BertModel
    m = BertModel.from_pretrained(MODEL_NAME).to(device).eval()
    for p in m.parameters():
        p.requires_grad_(False)
    return m


def load_fresh_teacher(device: torch.device, num_labels: int = 2):
    """Fresh (general-pretrained) BertForSequenceClassification head — the
    starting point for task fine-tuning (see pipeline/finetune_teacher.py)."""
    from transformers.models.bert.modeling_bert import BertForSequenceClassification
    return BertForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=num_labels, ignore_mismatched_sizes=True
    ).to(device)


def load_finetuned_teacher(ckpt_path: str, device: torch.device,
                           freeze: bool = True, num_labels: int = 2):
    """Load a task-fine-tuned BertForSequenceClassification teacher.

    Checkpoints saved by the fine-tuning scripts store the classifier state_dict
    under the "bert" key (see exp8d/exp12c/finetune_teacher). num_labels must
    match the task the checkpoint was trained on.
    """
    from transformers.models.bert.modeling_bert import BertForSequenceClassification
    m = BertForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=num_labels, ignore_mismatched_sizes=True
    ).to(device)
    state = torch.load(ckpt_path, map_location=device)
    m.load_state_dict(state["bert"] if "bert" in state else state)
    if freeze:
        m.eval()
        for p in m.parameters():
            p.requires_grad_(False)
    return m


def encoder_layers(model):
    """Return the encoder layer ModuleList for either a BertModel or a
    BertForSequenceClassification."""
    if hasattr(model, "bert"):
        return model.bert.encoder.layer
    return model.encoder.layer


def build_fc(hidden: int, degree: int, device: torch.device,
             act: str = "bern") -> FCModel:
    """A single 312 -> hidden -> 312 FFN approximator.

    Bernstein coefficients use the near-identity ramp init (BERN_INIT); `init` is
    ignored by FCModel for the non-Bernstein activations.
    """
    return FCModel(layer_sizes=[HIDDEN_DIM, hidden, HIDDEN_DIM],
                   degree=degree, act=act, init=BERN_INIT).to(device)


def load_fc(ckpt_path: str, hidden: int, degree: int,
            device: torch.device, act: str = "bern") -> FCModel:
    fc = build_fc(hidden, degree, device, act)
    fc.load_state_dict(torch.load(ckpt_path, map_location=device))
    return fc


# ─────────────────────────────────────────────────────────────────────────────
# FFN I/O capture (isolation stages)
# ─────────────────────────────────────────────────────────────────────────────

def capture_ffn_io(model, layer, batch: dict, device: torch.device):
    """Run one forward pass and capture the FFN block input/output for `layer`.

    FFN input  = input to layer.intermediate (post-attention, post-LayerNorm).
    FFN output = output of layer.output.dense (pre-residual, pre-LayerNorm) —
    this is exactly what the Bernstein FCModel must approximate.
    Returns (ffn_in, ffn_out) flattened to (N, HIDDEN_DIM).
    """
    captured: dict = {}
    h1 = layer.intermediate.register_forward_hook(
        lambda m, i, o: captured.update({"in": i[0]}))
    h2 = layer.output.dense.register_forward_hook(
        lambda m, i, o: captured.update({"out": o}))
    with torch.no_grad():
        model(**{k: v.to(device) for k, v in batch.items()
                 if k in ("input_ids", "attention_mask", "token_type_ids")})
    h1.remove()
    h2.remove()
    return (captured["in"].reshape(-1, HIDDEN_DIM),
            captured["out"].reshape(-1, HIDDEN_DIM))


def cache_ffn_io(model, layer, loader, device: torch.device):
    """Capture FFN (input, output) for *every* batch once and concatenate.

    The teacher is frozen and the data fixed, so the FFN I/O is identical every
    epoch — caching it eliminates the redundant teacher forward pass that
    dominates isolation-stage runtime (see tasks/lessons.md L10). Returns two
    tensors (X, Y) of shape (N, HIDDEN_DIM) on `device`. Training then iterates
    token-vector slices of these instead of re-running the model. Opt-in: the
    default online path is unchanged, so existing results are unaffected.
    """
    xs, ys = [], []
    with torch.no_grad():
        for batch in loader:
            fin, fout = capture_ffn_io(model, layer, batch, device)
            xs.append(fin)
            ys.append(fout)
    return torch.cat(xs, dim=0), torch.cat(ys, dim=0)


def iter_cached(X, Y, tok_bs: int, shuffle: bool, device: torch.device):
    """Yield (x, y) minibatches of `tok_bs` rows from cached tensors X, Y.

    Same per-step token count as the online path (batch_size * max_length), so
    the number and size of optimizer steps match; only the teacher forward is
    skipped.
    """
    n = X.size(0)
    order = torch.randperm(n, device=device) if shuffle else None
    for i in range(0, n, tok_bs):
        idx = order[i:i + tok_bs] if shuffle else slice(i, i + tok_bs)
        yield X[idx], Y[idx]


def build_cal_loader(model, layer, train_loader, device: torch.device,
                     n_batches: int = 100, batch_size: int = 4096):
    """Collect FFN inputs into a TensorDataset for bound calibration."""
    chunks = []
    for i, batch in enumerate(train_loader):
        if i >= n_batches:
            break
        ffn_in, _ = capture_ffn_io(model, layer, batch, device)
        chunks.append(ffn_in.cpu())
    cal_flat = torch.cat(chunks, dim=0)
    cal_ds = TensorDataset(cal_flat, cal_flat)
    return DataLoader(cal_ds, batch_size=batch_size, shuffle=True), cal_flat


def calibrate_all_bern_layers(fc: FCModel, cal_loader, device: torch.device,
                              max_batches: int = 50, p_lo: float = 0.01,
                              p_hi: float = 0.99, min_width: float = 0.5) -> None:
    for idx in range(len(fc.get_bern_layers())):
        fc.calibrate_one_bern_layer_from_data(
            cal_loader, device, idx, max_batches=max_batches,
            p_lo=p_lo, p_hi=p_hi, min_width=min_width)


def init_bounds(fc: FCModel, lo: float = -5.0, hi: float = 5.0) -> None:
    for bl in fc.get_bern_layers():
        bl.input_bounds[..., 0].fill_(lo)
        bl.input_bounds[..., 1].fill_(hi)
        bl.use_bounds = True


def reset_bern_stats(fc: FCModel) -> None:
    for bl in fc.get_bern_layers():
        bl._clamp_total = 0
        bl._clamp_count = 0
        bl.reset_xnorm_stats()


def clamp_ratio(fc: FCModel) -> float:
    c = n = 0
    for bl in fc.get_bern_layers():
        c += bl._clamp_total
        n += bl._clamp_count
    return c / n if n > 0 else 0.0


def bern_grad_norm(fc: FCModel) -> float:
    total = 0.0
    for name, p in fc.named_parameters():
        if p.grad is not None and "bern_coeffs" in name:
            total += p.grad.norm().item() ** 2
    return total ** 0.5


# ─────────────────────────────────────────────────────────────────────────────
# FFN substitution (Stage 3)
# ─────────────────────────────────────────────────────────────────────────────

def patch_ffn(model, layer_idx: int, fc: FCModel) -> None:
    """Replace layer_idx's GeLU FFN (intermediate -> output.dense) with `fc`.

    Residual add, dropout and LayerNorm are preserved from the original
    BertOutput. Mirrors the substitution used in exp12c/exp13d.
    """
    layer = encoder_layers(model)[layer_idx]

    def _patched(attention_output):
        B, L, D = attention_output.shape
        ffn_out = fc(attention_output.reshape(B * L, D)).reshape(B, L, D)
        ffn_out = layer.output.dropout(ffn_out)
        return layer.output.LayerNorm(ffn_out + attention_output)

    layer.feed_forward_chunk = _patched


def batch_inputs(batch: dict, device: torch.device) -> dict:
    """Model kwargs from a batch, including token_type_ids when present.

    Sentence-pair tasks (MRPC/RTE/QNLI) carry segment ids that the FFN-capture
    in Stage 2 already uses; passing them here keeps the deployment distribution
    consistent with what Stage 2 matched (single-sentence tasks have all-zero
    segments, so this is a no-op for SST-2)."""
    kw = {"input_ids": batch["input_ids"].to(device),
          "attention_mask": batch["attention_mask"].to(device)}
    if "token_type_ids" in batch:
        kw["token_type_ids"] = batch["token_type_ids"].to(device)
    return kw


@torch.no_grad()
def evaluate_classifier(model, val_loader, device: torch.device) -> float:
    model.eval()
    correct = total = 0
    for batch in val_loader:
        out = model(**batch_inputs(batch, device))
        # HF classifiers return a ModelOutput (.logits); the clean deploy module
        # returns the logits tensor directly.
        logits = out.logits if hasattr(out, "logits") else out
        correct += (logits.argmax(-1) == batch["labels"].to(device)).sum().item()
        total += batch["labels"].size(0)
    return correct / total
