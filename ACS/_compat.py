"""
_compat.py
==========
Thin compatibility shim that lets the ACS distribution-shift scripts reuse
Bern2Edge's shared modules (``models``, ``bernstein``, ``kdtrain``,
``rule_extraction``) unchanged.

Bern2Edge refactored the original research pipeline's free functions +
module-global hyperparameters into an ``ExtractionConfig``-parameterized API. The
rule-extraction / evaluation math is byte-equivalent, only the *call structure*
changed. This module re-exposes the old interface the ported scripts expect:

  * ``make_loader``  — the tiny TensorDataset+DataLoader helper that used to live
                       in ``teacher.py`` (absent from Bern2Edge).
  * ``MLP``          — alias of ``models.AdultTeacherMLP`` (the teacher checkpoints
                       store weights under ``net.*``, matching this class).
  * ``ad``           — an ``activation_distill``-style facade: module-global
                       hyperparameters (``N_FIXED_GRID`` ...) + ``build_neuron_regimes``
                       / ``generate_candidate_rules`` / ``rule_mask`` delegating to
                       ``rule_extraction``.
  * ``M``            — a ``run_all_architectures_network_purity``-style facade:
                       ``cascade_greedy_cover`` / ``train_fallback_bern`` /
                       ``_predict_with_rules`` + the ``FB_*`` fallback constants.
  * ``kd_train_models`` — signature adapter over ``kdtrain.kd_train_models``
                       (Bern2Edge dropped ``class_names`` / ``fold_idx`` / ``save_dir``).

Nothing here edits Bern2Edge's shared modules, so the published Adult / CoverType /
HIGGS results are untouched. See ``README.md`` for the reuse map.

Numerical note: the checkpoint -> Table XI path (load student, extract, CART
fallback, evaluate) is byte-identical to the original pipeline. Only *from-scratch
retraining* can differ within seed std, because Bern2Edge's ``BernsteinLayer``
uses xavier (vs the original ramp) init and its KD warmup bounds are +/-7 (vs
+/-5). Table XI is therefore reproduced exactly from the shipped checkpoints; the
retrain path is provided for completeness and lands within the reported std.
"""

import types

import torch
from torch.utils.data import TensorDataset, DataLoader

# ── Reused Bern2Edge shared code ──────────────────────────────────────────────
from bern2edge.models import AdultTeacherMLP as MLP          # teacher (net.* checkpoint layout)
from bern2edge.rule_extraction import bern_regimes as _bern_regimes
from bern2edge.rule_extraction import extraction as _extraction
import bern2edge.kdtrain as _kdtrain


# ── make_loader (verbatim from the original teacher.py) ───────────────────────
def make_loader(X, y, batch_size, shuffle=True):
    ds = TensorDataset(
        torch.tensor(X, dtype=torch.float32),
        torch.tensor(y, dtype=torch.long),
    )
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)


# ── `ad` facade (activation_distill-style module globals) ─────────────────────
ad = types.ModuleType('activation_distill_compat')
ad.N_FIXED_GRID = _bern_regimes.N_FIXED_GRID     # 5
ad.PURITY_THRESHOLD = 0.90
ad.MIN_COVERAGE = 2
ad.MAX_DEPTH = 3
ad.rule_mask = _extraction.rule_mask


def _ad_build_neuron_regimes(model, n_fixed_grid=None):
    grid = ad.N_FIXED_GRID if n_fixed_grid is None else n_fixed_grid
    return _bern_regimes.build_neuron_regimes(model, grid)


def _ad_generate_candidate_rules(neurons, Z, y):
    # Original signature reads the module globals; Bern2Edge takes them explicitly.
    return _extraction.generate_candidate_rules(
        neurons, Z, y, ad.MAX_DEPTH, ad.PURITY_THRESHOLD, ad.MIN_COVERAGE)


ad.build_neuron_regimes = _ad_build_neuron_regimes
ad.generate_candidate_rules = _ad_generate_candidate_rules


# ── `M` facade (network-purity module: cover + fallback + predict) ────────────
M = types.ModuleType('network_purity_compat')
# Fallback hyperparameters (settable by callers exactly as on the original module;
# these defaults match rule_extraction.ExtractionConfig, i.e. the paper setting).
M.FB_HIDDEN = 4
M.FB_DEGREE = 3
M.FB_LR = 5e-3
M.FB_BATCH_SIZE = 256
M.FB_WARMUP_EPOCHS = 2
M.FB_MAX_EPOCHS = 400
M.FB_PATIENCE = 40
M.FB_RANGE_PEN_W = 1e-2

# cascade_greedy_cover: identical signature (all_rules, y, purity_stages,
# conflict_alpha=, same_cov_alpha=), drop-in.
M.cascade_greedy_cover = _extraction.cascade_greedy_cover


def _M_train_fallback_bern(X, y, input_dim, device='cpu'):
    """Original module read FB_* globals; Bern2Edge takes an ExtractionConfig.
    Build one from the (possibly caller-overridden) M.FB_* attributes."""
    cfg = _extraction.ExtractionConfig(
        fb_hidden=M.FB_HIDDEN, fb_degree=M.FB_DEGREE, fb_lr=M.FB_LR,
        fb_batch_size=M.FB_BATCH_SIZE, fb_warmup_epochs=M.FB_WARMUP_EPOCHS,
        fb_max_epochs=M.FB_MAX_EPOCHS, fb_patience=M.FB_PATIENCE,
        fb_range_pen_w=M.FB_RANGE_PEN_W)
    return _extraction.train_fallback_bern(X, y, input_dim, cfg, device=device)


def _M_predict_with_rules(rules, Z, X, fallback_pred=None):
    """Original default strategy was 'early_stop' (highest-coverage rule wins,
    then fallback); Bern2Edge keeps exactly that path as `_predict_dense`."""
    return _extraction._predict_dense(rules, Z, X, fallback_pred)


M.train_fallback_bern = _M_train_fallback_bern
M._predict_with_rules = _M_predict_with_rules


# ── kd_train_models signature adapter ─────────────────────────────────────────
def kd_train_models(*args, class_names=None, fold_idx=None, save_dir=None, **kwargs):
    """Adapter over ``kdtrain.kd_train_models``.

    Bern2Edge's version dropped ``class_names`` / ``fold_idx`` / ``save_dir`` (it
    saves checkpoints under ``./saved_models_kd`` and returns their paths in the
    ``model_path`` column). We accept and ignore those three so the ported call
    sites are unchanged; callers read ``model_path`` from the returned DataFrame,
    so the on-disk location is irrelevant.
    """
    return _kdtrain.kd_train_models(*args, **kwargs)
