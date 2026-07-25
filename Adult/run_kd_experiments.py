"""
Adult (ordinal) — knowledge-distillation training driver (Bern2Edge).

Reproduces the published student weights in ``student_model_weights/`` that back
the Bernstein-vs-ReLU comparison table. For every (architecture, activation)
config in ``RUNS`` it trains a compact ``FCModel`` student against the frozen
``AdultTeacherMLP`` teacher, across the 5 cross-validation folds (seed = 6 + fold),
and saves one checkpoint per (config, fold).

Pipeline per run (mirrors orindinal_exp.ipynb):
  1. build the student and an AdamW optimizer,
  2. (Bernstein only) progressive per-layer bound calibration,
  3. full KD training (cosine LR, early stopping on val accuracy),
  4. save weights + metadata using the published ``kd_fc_*.pth`` naming.

Run:
    python run_kd_experiments.py                       # full reproduction (k=5, 200 ep)
    python run_kd_experiments.py --folds 1 --epochs 5  # quick smoke test
"""

import argparse
import os
import sys

import torch
import torch.optim as optim

# The shared library modules live one directory up (Bern2Edge/).
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from bern2edge.models import AdultTeacherMLP, FCModel                   # noqa: E402
from bern2edge.data import adult_ordinal_dataloaders_kfold             # noqa: E402
from bern2edge.kdtrain import train_knowledge_distillation, kd_train_perlayer_bound, _safe  # noqa: E402

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Fixed hyper-parameters (shared by every run) ─────────────────────────────
TEACHER_CKPT  = "adult_teacher_ordinal.pt"
LEARNING_RATE = 6e-3
WEIGHT_DECAY  = 1e-4
RANGE_PENALTY = 1e-2     # Bernstein out-of-bounds penalty weight
BATCH_SIZE    = 256
EVAL_BS       = 8192
DATA_SEED     = 6        # fold seed; per-fold student seed = DATA_SEED + fold_id

# ── Experiment table ─────────────────────────────────────────────────────────
# One Bernstein and one ReLU example. To reproduce another architecture, copy a
# bern/relu pair and edit the fields:
#   layer_sizes : full width list, input ... output (input=14, output=2)
#   degree      : Bernstein polynomial degree (3 or 5); use None for ReLU
#   alpha       : KD mix; 0.0 = pure hard CE, higher = more teacher soft targets
#   T           : distillation temperature (any value is fine when alpha == 0)
RUNS = [
    # (layer_sizes,     act,    degree, alpha, T)
    ([14, 16, 2],       "bern", 3,      0.85,  2.0),    # Bernstein student
    ([14, 16, 2],       "relu", None,   0.5,   2.0),    # ReLU baseline
    # Published table also includes:
    # ([14, 32, 16, 2], "bern", 5,      0.5,   2.0),
    # ([14, 32, 16, 2], "relu", None,   0.85,  2.0),
    # ([14, 128, 2],    "bern", 3,      0.0,   1.0),
    # ([14, 128, 2],    "relu", None,   0.85,  2.0),
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
    parser = argparse.ArgumentParser(description="Adult ordinal KD training driver.")
    parser.add_argument("--out-dir", default="student_model_weights",
                        help="Directory to write student checkpoints into.")
    parser.add_argument("--epochs", type=int, default=200,
                        help="Max training epochs per run (early stopping may halt sooner).")
    parser.add_argument("--folds", type=int, default=5,
                        help="Number of cross-validation folds to train (<= 5).")
    args = parser.parse_args()

    # ── Build the K-fold loaders and load the frozen teacher ──────────────────
    folds, d_in, n_classes = adult_ordinal_dataloaders_kfold(
        teacher_ckpt=TEACHER_CKPT, batch_size=BATCH_SIZE, eval_bs=EVAL_BS,
        k=5, seed=DATA_SEED,
    )
    ckpt = torch.load(TEACHER_CKPT, map_location=device, weights_only=False)
    teacher = AdultTeacherMLP(input_dim=ckpt["input_dim"], d_layers=ckpt["d_layers"],
                              dropout=ckpt["dropout"], n_classes=n_classes).to(device)
    teacher.load_state_dict(ckpt["model_state"])
    teacher.eval()

    # ── Train every (run, fold) combination ──────────────────────────────────
    for layer_sizes, act, degree, alpha, T in RUNS:
        for fold_id in range(args.folds):
            seed = DATA_SEED + fold_id
            train_loader, val_loader, test_loader = folds[fold_id]

            print("=" * 80)
            print(f"arch={layer_sizes}  act={act}  degree={degree}  "
                  f"alpha={alpha}  T={T}  fold={fold_id}  seed={seed}")

            torch.manual_seed(seed)
            student = FCModel(layer_sizes=layer_sizes, degree=degree,
                              act=act, last_bern=False).to(device)
            optimizer = optim.AdamW(student.parameters(),
                                    lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

            # Bernstein students: calibrate per-layer input bounds first.
            if act == "bern":
                print("→ per-layer Bernstein calibration")
                kd_train_perlayer_bound(
                    student=student, teacher=teacher,
                    train_loader=train_loader, val_loader=val_loader,
                    optimizer=optimizer, label_smoothing=0.0,
                    T=T, alpha=alpha, device=device,
                    warmup_epochs=2, stage_epochs=5, max_calib_batches=1000,
                    range_penalty_weight=RANGE_PENALTY,
                )
                # Reset optimizer state so calibration momentum doesn't leak in.
                for group in optimizer.param_groups:
                    for p in group["params"]:
                        optimizer.state.pop(p, None)
                    group["lr"] = LEARNING_RATE

            # Full KD training (handles its own final test evaluation).
            print("→ KD training")
            results = train_knowledge_distillation(
                teacher=teacher, student=student,
                train_loader=train_loader, val_loader=val_loader,
                test_loader=test_loader, epochs=args.epochs,
                T=T, alpha=alpha, device=device, optimizer=optimizer,
                label_smoothing=0.0, patience=20, act=act,
                range_penalty_weight=RANGE_PENALTY,
            )

            save_checkpoint(args.out_dir, student, layer_sizes, act,
                            degree, alpha, T, seed, results)


if __name__ == "__main__":
    main()
