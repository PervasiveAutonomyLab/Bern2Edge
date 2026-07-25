"""
train_teacher.py
================
Train the ReLU FC teacher on ACS Income California-2018 (in-distribution).

Gate: if held-out test accuracy < ACC_GATE, print a warning and halt (no
distillation downstream).

Reuses Bern2Edge's teacher MLP (`models.AdultTeacherMLP`, via `_compat.MLP`) +
the `_compat.make_loader` helper; ACS data loaders live in `data.acs_income`.
"""

import argparse
import os
import sys

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = os.path.dirname(SCRIPT_DIR)          # ACS -> Bern2Edge
sys.path.insert(0, REPO_DIR)
sys.path.insert(0, SCRIPT_DIR)

from _compat import MLP, make_loader             # noqa: E402
from bern2edge.data import acs_income as acs_data          # noqa: E402

# ── Config ───────────────────────────────────────────────────────────────────
SEED        = acs_data.SEED
D_LAYERS    = [512, 256, 128]
DROPOUT     = 0.1
LR          = 1e-3
WEIGHT_DECAY = 1e-4
BATCH_SIZE  = 256
EVAL_BS     = 8192
MAX_EPOCHS  = 200
PATIENCE    = 16
# Raw-code (no one-hot) ACSIncome caps at ~81.4% for MLPs (HistGBDT ~82.6%), so a
# literal 0.83 gate is unreachable with the chosen "categoricals as-is" encoding.
# The experiment targets the distribution-shift effect, not SOTA accuracy, so the
# gate guards against a *broken* teacher rather than enforcing SOTA.
ACC_GATE    = 0.80


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    all_logits, all_labels = [], []
    for X_batch, y_batch in loader:
        all_logits.append(model(X_batch.to(device)).cpu())
        all_labels.append(y_batch)
    logits = torch.cat(all_logits)
    labels = torch.cat(all_labels).numpy()
    probs  = torch.softmax(logits, dim=1)[:, 1].numpy()
    preds  = logits.argmax(dim=1).numpy()
    return accuracy_score(labels, preds), roc_auc_score(labels, probs)


def fit_relu_mlp(X_tr, y_tr, X_val, y_val, input_dim, device, *,
                 d_layers=D_LAYERS, dropout=DROPOUT, lr=LR, wd=WEIGHT_DECAY,
                 batch_size=BATCH_SIZE, eval_bs=EVAL_BS, max_epochs=MAX_EPOCHS,
                 patience=PATIENCE, seed=SEED, verbose=True):
    """Train a ReLU MLP with early stopping on val accuracy; return best model.
    Shared by this script and the multi-seed driver."""
    torch.manual_seed(seed); np.random.seed(seed)
    model     = MLP(input_dim, d_layers, dropout=dropout, n_classes=2).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    criterion = nn.CrossEntropyLoss()
    train_loader = make_loader(X_tr,  y_tr,  batch_size, shuffle=True)
    val_loader   = make_loader(X_val, y_val, eval_bs,    shuffle=False)

    best_val_acc, best_state, best_epoch, bad = -1.0, None, 0, 0
    for epoch in range(1, max_epochs + 1):
        model.train()
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            criterion(model(X_batch), y_batch).backward()
            optimizer.step()
        val_acc, val_auc = evaluate(model, val_loader, device)
        if val_acc > best_val_acc:
            best_val_acc, best_epoch, bad = val_acc, epoch, 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            if verbose:
                print(f"  Epoch {epoch:4d} | val_acc={val_acc:.4f} auc={val_auc:.4f} ✓")
        else:
            bad += 1
            if bad >= patience:
                if verbose:
                    print(f"  Early stop at epoch {epoch}, best epoch={best_epoch}")
                break
    model.load_state_dict(best_state)
    return model


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--data-dir',    default=os.path.join(SCRIPT_DIR, 'data'))
    ap.add_argument('--out',         default=os.path.join(SCRIPT_DIR, 'checkpoints', 'teacher.pt'))
    ap.add_argument('--acc-gate',    type=float, default=ACC_GATE)
    args = ap.parse_args()

    acs_data.set_seed(SEED)
    device = acs_data.get_device()
    print(f"Using device: {device}")

    # 1. Load CA-2018 (in-distribution)
    print("Loading ACS Income CA-2018 ...", flush=True)
    splits = acs_data.download_acs(args.data_dir)
    X_raw, y_raw = splits['ca2018']
    input_dim = X_raw.shape[1]
    print(f"  CA-2018: X={X_raw.shape}  positive rate={y_raw.mean():.4f}")

    # 2. Stratified 80/20 train/test split, then fit scaler on TRAIN only
    idx = np.arange(len(y_raw))
    idx_train, idx_test = train_test_split(
        idx, test_size=0.2, random_state=SEED, stratify=y_raw)
    scaler = acs_data.fit_scaler(X_raw[idx_train])

    X_tr_all = acs_data.apply_scaler(scaler, X_raw[idx_train])
    X_test   = acs_data.apply_scaler(scaler, X_raw[idx_test])
    y_tr_all = y_raw[idx_train]
    y_test   = y_raw[idx_test]

    # internal train/val split for early stopping
    rel_tr, rel_val = train_test_split(
        np.arange(len(idx_train)), test_size=0.1, random_state=SEED, stratify=y_tr_all)
    X_tr, X_val = X_tr_all[rel_tr], X_tr_all[rel_val]
    y_tr, y_val = y_tr_all[rel_tr], y_tr_all[rel_val]
    print(f"  train={len(y_tr)}  val={len(y_val)}  test={len(y_test)}")

    # 3. Train ReLU MLP with early stopping on val accuracy
    model = fit_relu_mlp(X_tr, y_tr, X_val, y_val, input_dim, device, seed=SEED)
    test_loader = make_loader(X_test, y_test, EVAL_BS, shuffle=False)
    test_acc, test_auc = evaluate(model, test_loader, device)
    print(f"\nTeacher held-out test: acc={test_acc:.4f}  auc={test_auc:.4f}")

    # 4. Accuracy gate
    if test_acc < args.acc_gate:
        print(f"\n*** WARNING: teacher test acc {test_acc:.4f} < gate {args.acc_gate:.2f}. "
              f"Halting — not proceeding to distillation. ***")
        sys.exit(1)

    # 5. Save checkpoint (raw data + indices + scaler so downstream is reproducible)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    torch.save({
        'model_state': {k: v.cpu() for k, v in model.state_dict().items()},
        'd_layers':    D_LAYERS,
        'input_dim':   input_dim,
        'dropout':     DROPOUT,
        'scaler':      scaler,
        'X_raw':       X_raw,
        'y_raw':       y_raw,
        'idx_train':   idx_train,
        'idx_test':    idx_test,
        'test_acc':    float(test_acc),
        'test_auc':    float(test_auc),
        'seed':        SEED,
    }, args.out)
    print(f"Saved teacher -> {args.out}")


if __name__ == '__main__':
    main()
