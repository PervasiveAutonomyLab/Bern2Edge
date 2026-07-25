"""
visualize_neurons.py
--------------------
Plot a handful of shape-diverse Bernstein activation neurons with their regime
breakpoints overlaid — a one-figure illustration of the geometry the rule
extractor reasons about.

The regime math (grid resolution + breakpoints) is imported from
`bern_regimes`, so this figure is guaranteed to use the SAME `N_FIXED_GRID`
as the extractor (no drift between what we show and what we extract).

Run:
    python -m bern2edge.rule_extraction.visualize_neurons \
        --ckpt Adult/rule_checkpoints/kd_fc_14x64x2_bern_deg3_alpha0.5_T2_lr0.006_wd0.0001_seed6.pth

Writes  bern_neuron_plots_<ckpt_stem>/01_neuron_polynomials.pdf  next to the
checkpoint's directory (or --out-dir).
"""

import argparse
import os
from collections import defaultdict

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from ..models import FCModel
from .bern_regimes import (N_FIXED_GRID, classify_motif_extended,
                           compute_regime_breakpoints, eval_bern_poly)

# Relative --ckpt paths are resolved against the repo root (bern2edge/rule_extraction/ -> ../..),
# so the documented command works from anywhere.
_REPO_DIR = os.path.abspath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, os.pardir))

# IEEE-style rendering: serif fonts, embedded (Type-42) fonts, true vector PDF.
# Figures are sized to the physical IEEE column width so the fonts print legibly.
plt.rcParams.update({
    "font.family":      "serif",
    "font.serif":       ["Times New Roman", "Times", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "pdf.fonttype":     42,
    "ps.fonttype":      42,
    "axes.linewidth":   0.6,
    "lines.linewidth":  1.2,
})

# Colours / labels per motif (classify_motif_extended's full label set).
MOTIF_COLORS = {
    'monotone_up':   '#2196F3',
    'monotone_down': '#F44336',
    'bump':          '#4CAF50',
    'valley':        '#FF9800',
    'sigmoid_up':    '#7B1FA2',
    'sigmoid_down':  '#AD1457',
    'complex':       '#607D8B',
}
MOTIF_LABELS = {
    'monotone_up':   'Monotone ↑',
    'monotone_down': 'Monotone ↓',
    'bump':          'Bump',
    'valley':        'Valley',
    'sigmoid_up':    'Sigmoid ↑',
    'sigmoid_down':  'Sigmoid ↓',
    'complex':       'Complex',
}
PRIORITY = ['monotone_up', 'monotone_down', 'bump', 'valley', 'sigmoid_up', 'sigmoid_down']


def load_bern_layer(ckpt_path):
    """Load an FCModel checkpoint and return its first Bernstein layer's coeffs."""
    raw = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model = FCModel(layer_sizes=raw["arch"], degree=raw["degree"],
                    act=raw["activation"], last_bern=False, dropout=0.0)
    model.load_state_dict(raw["state_dict"])
    model.eval()
    bl = model.get_bern_layers()[0]
    return bl.bern_coeffs.detach().cpu().numpy()          # [N_neurons, degree+1]


def select_diverse_neurons(coeffs, n_select=6):
    """Pick up to `n_select` neurons spanning as many distinct motifs as possible,
    then fill any remaining slots from the most-populated motifs."""
    global_std = coeffs.std()
    motifs = [classify_motif_extended(coeffs[i], global_std) for i in range(len(coeffs))]
    by_motif = defaultdict(list)
    for i, m in enumerate(motifs):
        by_motif[m].append(i)

    selected, used = [], set()
    for motif in PRIORITY:                       # one per motif, in priority order
        if len(selected) >= n_select:
            break
        for idx in by_motif.get(motif, []):
            if idx not in used:
                selected.append((idx, motif))
                used.add(idx)
                break
    # Fill remaining slots from the largest motif groups first.
    for motif in sorted(by_motif, key=lambda m: -len(by_motif[m])):
        for idx in by_motif.get(motif, []):
            if len(selected) >= n_select:
                break
            if idx not in used:
                selected.append((idx, motif))
                used.add(idx)
    selected.sort(key=lambda x: x[0])            # order by neuron index
    return selected


def plot_neurons(coeffs, selected, out_path):
    """2x3 grid of the selected neurons' activation curves with regime breakpoints."""
    x01 = np.linspace(0.0, 1.0, 300)
    TITLE_FS, LABEL_FS, TICK_FS, LEGEND_FS = 8, 8, 7, 7

    # 3.5 in = IEEE single-column text width; the figure prints at this size.
    fig, axes = plt.subplots(2, 3, figsize=(3.5, 2.4), sharex=True, sharey=True)
    appearing = []
    for ax_idx, (ax, (i, motif)) in enumerate(zip(axes.flat, selected)):
        row, col = divmod(ax_idx, 3)
        color = MOTIF_COLORS.get(motif, '#607D8B')
        if motif not in appearing:
            appearing.append(motif)

        ax.plot(x01, eval_bern_poly(coeffs[i], x01), color=color, lw=1.4)
        # Breakpoints live in normalised t-space, which equals the x01 axis here.
        for t in compute_regime_breakpoints(coeffs[i], motif, N_FIXED_GRID):
            ax.axvline(t, color='black', lw=0.6, linestyle='--', alpha=0.55)

        ax.set_xlim(0, 1)
        ax.set_ylim(-0.05, 1.05)
        ax.set_title(f"Neuron {i}", fontsize=TITLE_FS, pad=2)
        ax.set_xticks([0, 0.5, 1])
        ax.set_yticks([0, 0.5, 1])
        ax.tick_params(axis='both', labelsize=TICK_FS, length=2, pad=1.5)
        if row != 1:
            ax.tick_params(axis='x', labelbottom=False)
        if col != 0:
            ax.tick_params(axis='y', labelleft=False)

    fig.supxlabel("Normalized input", fontsize=LABEL_FS, y=0.155)
    fig.supylabel("Output", fontsize=LABEL_FS, x=0.015)

    handles = [mpatches.Patch(color=MOTIF_COLORS[m], label=MOTIF_LABELS[m]) for m in appearing]
    fig.legend(handles=handles, loc="lower center", ncol=min(len(handles), 4),
               fontsize=LEGEND_FS, frameon=False, bbox_to_anchor=(0.5, 0.02),
               columnspacing=1.0, handletextpad=0.4, handlelength=1.2)
    fig.subplots_adjust(left=0.115, right=0.985, top=0.92, bottom=0.27,
                        wspace=0.24, hspace=0.34)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    default_ckpt = os.path.join(
        "Adult", "rule_checkpoints",
        "kd_fc_14x64x2_bern_deg3_alpha0.5_T2_lr0.006_wd0.0001_seed6.pth")
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", default=default_ckpt,
                    help="path to a Bernstein FCModel checkpoint (.pth)")
    ap.add_argument("--out-dir", default=None,
                    help="output directory (default: bern_neuron_plots_<ckpt_stem>/ "
                         "beside the checkpoint)")
    args = ap.parse_args()

    ckpt_path = args.ckpt if os.path.isabs(args.ckpt) else os.path.join(_REPO_DIR, args.ckpt)
    ckpt_stem = os.path.splitext(os.path.basename(ckpt_path))[0]
    out_dir = args.out_dir or os.path.join(os.path.dirname(ckpt_path),
                                           f"bern_neuron_plots_{ckpt_stem}")
    os.makedirs(out_dir, exist_ok=True)

    coeffs = load_bern_layer(ckpt_path)
    print(f"Loaded {len(coeffs)} neurons (degree {coeffs.shape[1] - 1}) "
          f"from {os.path.basename(ckpt_path)}; grid={N_FIXED_GRID}")
    selected = select_diverse_neurons(coeffs, n_select=6)
    print(f"Selected neurons: {selected}")

    out_path = os.path.join(out_dir, "01_neuron_polynomials.pdf")
    plot_neurons(coeffs, selected, out_path)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
