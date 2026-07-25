"""
teacher_magic.py  (MAGIC teacher MLP — local)
=============================================
The frozen ReLU-MLP teacher distilled into the Bernstein/ReLU students. The shipped
``teacher_magic.pt`` is loaded by the KD driver (``run_kd_experiments.py``) and by the
certification scripts (it also carries the fixed split + fitted preprocessor + raw
arrays the rule extractor / margin analysis read).

Kept local (a small ReLU ``MLP`` matching the class that produced ``teacher_magic.pt``)
because the repo-wide ``models.TeacherMLP`` uses different state-dict keys and would not
load the shipped checkpoint. Run this module directly to re-train the teacher from
scratch and re-write ``teacher_magic.pt``.

Usage:
    python teacher_magic.py          # re-train + overwrite teacher_magic.pt
"""
import os
import sys

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import QuantileTransformer
from sklearn.metrics import accuracy_score, roc_auc_score

SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(SCRIPT_DIR, "..")))  # shared data.py
from bern2edge.data import _load_magic, MAGIC_FEATURES   # noqa: E402

TEACHER_CKPT = os.path.join(SCRIPT_DIR, "teacher_magic.pt")
SEED         = 42
DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Teacher hyper-parameters (match the shipped checkpoint).
D_LAYERS     = [64, 32, 16]
INPUT_DIM    = 10
N_CLASSES    = 2
LR           = 1e-3
WEIGHT_DECAY = 1e-4
BATCH_SIZE   = 32
EVAL_BS      = 8192
EPOCHS       = 200
PATIENCE     = 16


class MLP(nn.Module):
    """(in) -> [Linear -> ReLU (-> Dropout)] x L -> Linear -> logits."""

    def __init__(self, input_dim, d_layers, dropout=0.0, n_classes=2):
        super().__init__()
        layers, in_dim = [], input_dim
        for out_dim in d_layers:
            layers.append(nn.Linear(in_dim, out_dim))
            layers.append(nn.ReLU())
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
            in_dim = out_dim
        layers.append(nn.Linear(in_dim, n_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def make_loader(X, y, batch_size, shuffle=True):
    ds = TensorDataset(torch.tensor(X, dtype=torch.float32),
                       torch.tensor(y, dtype=torch.long))
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)


def load_teacher(ckpt_path=TEACHER_CKPT, device="cpu"):
    """Rebuild the frozen teacher MLP from a checkpoint. Returns (model.eval(), ckpt_dict)."""
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = MLP(ck["input_dim"], ck["d_layers"], dropout=0.0, n_classes=N_CLASSES).to(device)
    model.load_state_dict(ck["model_state"])
    model.eval()
    return model, ck


@torch.no_grad()
def _evaluate(model, loader):
    model.eval()
    logits, labels = [], []
    for Xb, yb in loader:
        logits.append(model(Xb.to(DEVICE)).cpu())
        labels.append(yb)
    logits = torch.cat(logits); labels = torch.cat(labels).numpy()
    probs = torch.softmax(logits, dim=1)[:, 1].numpy()
    preds = logits.argmax(dim=1).numpy()
    return accuracy_score(labels, preds), roc_auc_score(labels, probs)


def main():
    torch.manual_seed(SEED); np.random.seed(SEED)
    print(f"Using device: {DEVICE}")

    # Fixed split + preprocessing (identical to data.magic_dataloaders).
    X, y = _load_magic()
    idx = np.arange(len(y))
    idx_train, idx_test = train_test_split(idx, test_size=0.2, random_state=42, stratify=y)
    qt = QuantileTransformer(output_distribution='normal', n_quantiles=1000,
                             random_state=42, subsample=int(1e9))
    X_train_all = qt.fit_transform(X[idx_train]).astype(np.float32)
    X_test      = qt.transform(X[idx_test]).astype(np.float32)
    y_train_all, y_test = y[idx_train], y[idx_test]

    idx_tr, idx_val = train_test_split(np.arange(len(idx_train)), test_size=0.1,
                                       random_state=SEED, stratify=y_train_all)

    train_loader = make_loader(X_train_all[idx_tr],  y_train_all[idx_tr],  BATCH_SIZE, True)
    val_loader   = make_loader(X_train_all[idx_val], y_train_all[idx_val], EVAL_BS,    False)
    test_loader  = make_loader(X_test, y_test, EVAL_BS, False)

    model     = MLP(INPUT_DIM, D_LAYERS, dropout=0.0, n_classes=N_CLASSES).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    criterion = nn.CrossEntropyLoss()

    best_acc, best_state, best_epoch, patience = -1.0, None, 0, 0
    for epoch in range(1, EPOCHS + 1):
        model.train()
        for Xb, yb in train_loader:
            Xb, yb = Xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad(); criterion(model(Xb), yb).backward(); optimizer.step()
        val_acc, _ = _evaluate(model, val_loader)
        if val_acc > best_acc:
            best_acc, best_epoch, patience = val_acc, epoch, 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience += 1
            if patience >= PATIENCE:
                print(f"  Early stop at epoch {epoch} (best {best_epoch})"); break

    model.load_state_dict(best_state)
    test_acc, test_auc = _evaluate(model, test_loader)
    print(f"Teacher test_acc={test_acc:.4f}  test_auc={test_auc:.4f}")

    torch.save({
        'model_state':   model.state_dict(),
        'preprocessor':  qt,
        'input_dim':     INPUT_DIM,
        'd_layers':      D_LAYERS,
        'test_acc':      test_acc,
        'test_auc':      test_auc,
        'idx_train':     idx_train,
        'idx_test':      idx_test,
        'feature_names': MAGIC_FEATURES,
        'X_raw':         X,
        'y_raw':         y,
    }, TEACHER_CKPT)
    print(f"Checkpoint saved -> {TEACHER_CKPT}")


if __name__ == "__main__":
    main()
