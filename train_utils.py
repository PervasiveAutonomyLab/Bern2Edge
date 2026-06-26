"""
Plain (non-distillation) training utilities for the Cover Type students.

  * train               : full training loop with cosine LR + early stopping.
  * train_perlayer_bound: progressive per-layer Bernstein bound calibration.
  * test                : single-pass test-set evaluation (loss + accuracy).

For the knowledge-distillation versions of these, see `kdtrain.py`.
"""

import copy
import torch
import torch.nn as nn


def train(
    model,
    train_loader,
    val_loader,
    test_loader=None,          # optional: evaluate once at the end
    epochs=200,
    device="cuda",
    label_smoothing=0.0,
    optimizer=None,
    patience=20,               # early-stopping patience (epochs without improvement)
    act="relu",
):
    """
    Train `model` with CrossEntropy (+ Bernstein range penalty when act=="bern"),
    keeping the best-validation-accuracy checkpoint. Returns
    (best_val_acc, train_ce_at_best, val_loss_at_best).
    """
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    model.to(device)

    best_state = None
    best_val_acc = -1.0
    bad_epochs = 0
    acc_val = loss_train = v_loss = None

    for epoch in range(epochs):
        # ---- Train ----
        model.train()
        running_loss = 0.0
        ce_running_loss = 0.0
        for data, targets in train_loader:
            data, targets = data.to(device), targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(data)
            ce_loss = criterion(logits, targets)
            # Bernstein activations add a soft penalty for inputs outside [0, 1].
            loss = ce_loss + (1e-2 * model.range_penalty()) if act == "bern" else ce_loss
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * data.size(0)
            ce_running_loss += ce_loss.item() * data.size(0)
        epoch_ce_loss = ce_running_loss / len(train_loader.dataset)
        scheduler.step()

        # ---- Validate ----
        model.eval()
        correct, total = 0, 0
        val_running_loss = 0.0
        with torch.no_grad():
            for data, targets in val_loader:
                data, targets = data.to(device), targets.to(device)
                logits = model(data)
                val_running_loss += criterion(logits, targets).item() * data.size(0)
                correct += (logits.argmax(dim=1) == targets).sum().item()
                total += targets.size(0)
        val_loss = val_running_loss / len(val_loader.dataset)
        val_acc = 100.0 * correct / total

        print(
            f"Epoch {epoch + 1}/{epochs} | CE train loss: {epoch_ce_loss:.4f} | "
            f"val loss: {val_loss:.4f} | val acc: {val_acc:.2f}%"
        )

        # ---- Early stopping on best validation accuracy ----
        if val_acc > best_val_acc + 0.01:
            best_val_acc = val_acc
            acc_val = best_val_acc
            v_loss = val_loss
            loss_train = epoch_ce_loss
            best_state = copy.deepcopy(model.state_dict())
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                print(
                    f"Early stopping at epoch {epoch + 1}. "
                    f"Best val acc: {best_val_acc:.2f}%"
                )
                break

    # Restore the best checkpoint.
    if best_state is not None:
        model.load_state_dict(best_state)

    # ---- Optional final test evaluation ----
    if test_loader is not None:
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for data, targets in test_loader:
                data, targets = data.to(device), targets.to(device)
                correct += (model(data).argmax(dim=1) == targets).sum().item()
                total += targets.size(0)
        print(f"Final test acc: {100.0 * correct / total:.2f}%")

    return acc_val, loss_train, v_loss


def train_one_epoch(model, loader, optimizer, criterion, device):
    """Single plain training epoch; returns (mean CE loss, accuracy fraction)."""
    model.train()
    correct, total = 0, 0
    ce_running_loss = 0.0
    for data, targets in loader:
        data, targets = data.to(device), targets.to(device)
        optimizer.zero_grad(set_to_none=True)
        output = model(data)
        ce_loss = criterion(output, targets)
        loss = ce_loss + (1e-2 * model.range_penalty())
        loss.backward()
        optimizer.step()
        ce_running_loss += ce_loss.item() * data.size(0)
        correct += (output.argmax(dim=1) == targets).sum().item()
        total += data.size(0)
    return ce_running_loss / total, correct / total


@torch.no_grad()
def eval_one_epoch(model, loader, criterion, device):
    """Single evaluation pass; returns (mean loss, accuracy fraction)."""
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        total_loss += criterion(logits, y).item() * x.size(0)
        correct += (logits.argmax(dim=1) == y).sum().item()
        total += x.size(0)
    return total_loss / total, correct / total


def train_perlayer_bound(
    model,
    train_loader,
    val_loader,
    optimizer,
    label_smoothing,
    device,
    warmup_epochs=2,
    stage_epochs=5,
    max_calib_batches=200,
):
    """
    Calibrate Bernstein input bounds layer by layer:
      1. warm up with wide initial bounds [-3, 3],
      2. for each Bernstein layer: estimate bounds from data, then train a few
         epochs before moving on to the next layer.
    """
    model.to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    bern_layers = model.get_bern_layers()

    # ---- Warmup with wide initial bounds ----
    print("initial bounds: -3, 3")
    for bl in bern_layers:
        bl.input_bounds[..., 0].fill_(-3.0)
        bl.input_bounds[..., 1].fill_(3.0)
        bl.use_bounds = True

    for e in range(warmup_epochs):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
        te_loss, te_acc = eval_one_epoch(model, val_loader, criterion, device)
        print(
            f"Warmup {e + 1}/{warmup_epochs} | loss={tr_loss:.4f} "
            f"train acc={tr_acc * 100:.2f}% | val acc={te_acc * 100:.2f}%"
        )

    # ---- Progressive per-layer calibration ----
    for k in range(len(bern_layers)):
        model.calibrate_one_bern_layer_from_data(
            loader=train_loader,
            device=device,
            layer_idx=k,
            max_batches=max_calib_batches,
            p_lo=0.001,
            p_hi=0.999,
            min_width=0.2,
        )
        for e in range(stage_epochs):
            tr_loss, tr_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
            te_loss, te_acc = eval_one_epoch(model, val_loader, criterion, device)
            print(
                f"  Layer {k} stage {e + 1}/{stage_epochs} | loss={tr_loss:.4f} "
                f"train acc={tr_acc * 100:.2f}% | val acc={te_acc * 100:.2f}%"
            )


def test(model, test_loader, device):
    """Evaluate `model` on `test_loader`; returns (mean CE loss, accuracy %)."""
    model.to(device)
    model.eval()
    criterion = nn.CrossEntropyLoss(label_smoothing=0)

    correct, total, test_loss = 0, 0, 0.0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item() * data.size(0)
            correct += output.argmax(dim=1).eq(target).sum().item()
            total += target.size(0)

    test_loss /= total
    accuracy = 100.0 * correct / total
    print(f"Test set: average loss: {test_loss:.4f}, accuracy: {correct}/{total} ({accuracy:.2f}%)")
    return test_loss, accuracy
