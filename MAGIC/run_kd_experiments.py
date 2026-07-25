"""
MAGIC — knowledge-distillation training driver (paper TABLE V students).

Reproduces the student weights in ``student_model_weights/`` behind TABLE V (and the
TABLE X certification). For every ``(architecture, activation, alpha, T)`` config in
``RUNS`` it distils a compact ``FCModel`` student from the frozen ``teacher_magic.pt``
MLP across seeds ``{0,1,2,3,42}`` on the fixed 80/20 split, and saves one checkpoint per
(config, seed) using the published ``kd_fc_*.pth`` naming.

Reuses the shared training library (``kdtrain``) and the shared MAGIC loader
(``data.magic_dataloaders``); the student model core (``bern_net.FCModel``) and the
teacher MLP (``teacher_magic``) are local so the shipped weights load verbatim.

Configs (matching the shipped weights):
  - [10,64,32,2]    bern deg3  alpha=0.5 T=2   -> KD students        (TABLE V "KD" rows)
  - [10,64,32,16,2] bern deg3  alpha=0.0 T=1   -> no-KD (pure CE)    (TABLE V "no-KD" rows)
  - [10,64,32,2]    relu       alpha=0.5 T=2   -> matched ReLU       (TABLE X ReLU column)

Note: training from scratch will not bit-reproduce the shipped weights (different seeds
of float nondeterminism give different learned neuron shapes, hence different extracted
rule counts); it reproduces the TABLE V metrics within +/- std. For the exact published
numbers, use the shipped weights + `make_table5.py` / `make_table_x.py`.

Run:
    python run_kd_experiments.py                       # full: 3 configs x 5 seeds, 100 ep
    python run_kd_experiments.py --seeds 42 --epochs 5 # quick smoke test (one seed)
"""
import argparse
import os
import sys

import torch
import torch.optim as optim

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(SCRIPT_DIR, "..")))  # shared modules

from bern2edge.data import magic_dataloaders                                    # noqa: E402
from bern2edge.kdtrain import train_knowledge_distillation, kd_train_perlayer_bound, _safe  # noqa: E402
from bern2edge.models import FCModel                                            # noqa: E402  (shared: trains with kdtrain)
from teacher_magic import load_teacher                                # noqa: E402  (local teacher MLP)

# Training uses the shared ``models.FCModel`` (co-designed with ``kdtrain``'s per-layer
# Bernstein calibration). The saved state_dict is layout-compatible with the local
# ``bern_net.FCModel`` used for extraction/certification, so from-scratch students load
# there verbatim.

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Fixed hyper-parameters (match the shipped weights) ───────────────────────
TEACHER_CKPT  = os.path.join(SCRIPT_DIR, "teacher_magic.pt")
LEARNING_RATE = 3e-3
WEIGHT_DECAY  = 1e-4
RANGE_PENALTY = 0.0        # MAGIC students trained without the range penalty
BATCH_SIZE    = 256
SEEDS         = [0, 1, 2, 3, 42]

# ── Experiment table:  (layer_sizes, act, degree, alpha, T) ──────────────────
RUNS = [
    ([10, 64, 32, 2],     "bern", 3,    0.5, 2.0),   # KD students   (TABLE V "KD")
    ([10, 64, 32, 16, 2], "bern", 3,    0.0, 1.0),   # no-KD pure CE (TABLE V "no-KD")
    ([10, 64, 32, 2],     "relu", None, 0.5, 2.0),   # matched ReLU  (TABLE X ReLU col)
]


def save_checkpoint(out_dir, student, layer_sizes, act, degree, alpha, T, seed, results):
    """Write one student checkpoint using the published ``kd_fc_*.pth`` naming."""
    os.makedirs(out_dir, exist_ok=True)
    arch_str = "x".join(map(str, layer_sizes))
    deg_str  = f"deg{degree}" if act == "bern" else "degNA"
    name = _safe(
        f"kd_fc_{arch_str}_{act}_{deg_str}_alpha{alpha:g}_T{T:g}_"
        f"lr{LEARNING_RATE:g}_wd{WEIGHT_DECAY:g}_seed{seed}.pth"
    )
    path = os.path.join(out_dir, name)
    torch.save(
        {
            "state_dict":    student.state_dict(),
            "arch":          layer_sizes,
            "activation":    act,
            "degree":        degree if act == "bern" else None,
            "alpha":         alpha,
            "T":             T,
            "last_bern":     False,
            "learning_rate": LEARNING_RATE,
            "weight_decay":  WEIGHT_DECAY,
            "seed":          seed,
            "val_acc":       float(results["val_acc"]),
            "test_acc":      float(results["test_acc"]),
        },
        path,
    )
    print(f"  saved {path}")


def main():
    ap = argparse.ArgumentParser(description="MAGIC KD training driver (TABLE V students).")
    ap.add_argument("--out-dir", default=os.path.join(SCRIPT_DIR, "student_model_weights"))
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    args = ap.parse_args()

    teacher, _ = load_teacher(TEACHER_CKPT, device=device)
    teacher.to(device).eval()

    for layer_sizes, act, degree, alpha, T in RUNS:
        for seed in args.seeds:
            train_loader, val_loader, test_loader, d_in, n_classes, _qt = magic_dataloaders(
                batch_size=BATCH_SIZE, seed=seed
            )
            print("=" * 80)
            print(f"arch={layer_sizes}  act={act}  degree={degree}  "
                  f"alpha={alpha}  T={T}  seed={seed}")

            torch.manual_seed(seed)
            student = FCModel(layer_sizes=layer_sizes, degree=degree if degree else 3,
                              act=act, last_bern=False).to(device)
            optimizer = optim.AdamW(student.parameters(),
                                    lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

            # Bernstein students: progressive per-layer input-bound calibration first.
            if act == "bern":
                print("-> per-layer Bernstein calibration")
                kd_train_perlayer_bound(
                    student=student, teacher=teacher,
                    train_loader=train_loader, val_loader=val_loader,
                    optimizer=optimizer, label_smoothing=0.0,
                    T=T, alpha=alpha, device=device,
                    warmup_epochs=2, stage_epochs=5, max_calib_batches=1000,
                    range_penalty_weight=RANGE_PENALTY,
                )
                for group in optimizer.param_groups:          # reset optimizer state
                    for p in group["params"]:
                        optimizer.state.pop(p, None)
                    group["lr"] = LEARNING_RATE

            print("-> KD training")
            results = train_knowledge_distillation(
                teacher=teacher, student=student,
                train_loader=train_loader, val_loader=val_loader,
                test_loader=test_loader, epochs=args.epochs,
                T=T, alpha=alpha, device=device, optimizer=optimizer,
                label_smoothing=0.0, patience=30, act=act,
                range_penalty_weight=RANGE_PENALTY,
            )
            save_checkpoint(args.out_dir, student, layer_sizes, act,
                            degree, alpha, T, seed, results)


if __name__ == "__main__":
    main()
