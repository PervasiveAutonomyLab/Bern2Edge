import copy
import numpy as np
import torch
import torch.nn as nn
from models import FCModel
import torch.optim as optim
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import pandas as pd
import os, re

def _safe(s: str) -> str:
    # filesystem-safe slug
    s = str(s)
    s = s.replace(" ", "")
    s = re.sub(r"[^a-zA-Z0-9_\-\.]", "", s)
    return s

# ─────────────────────────────────────────────────────────────────────────────
# Per-class F1 report
# ─────────────────────────────────────────────────────────────────────────────

def classification_report_simple(preds, targets, class_names, indent="  "):
    n  = len(class_names)
    tp = np.zeros(n)
    fp = np.zeros(n)
    fn = np.zeros(n)

    for p, t in zip(preds, targets):
        if p == t:
            tp[t] += 1
        else:
            fp[p] += 1
            fn[t] += 1

    print(f"\n{indent}{'Class':<22} {'Prec':>7} {'Rec':>7} {'F1':>7} {'Support':>9}")
    print(f"{indent}{'-'*55}")
    f1_scores = []
    for i, name in enumerate(class_names):
        prec = tp[i] / (tp[i] + fp[i] + 1e-8)
        rec  = tp[i] / (tp[i] + fn[i] + 1e-8)
        f1   = 2 * prec * rec / (prec + rec + 1e-8)
        sup  = int(tp[i] + fn[i])
        f1_scores.append(f1)
        print(f"{indent}{name:<22} {prec:>7.4f} {rec:>7.4f} {f1:>7.4f} {sup:>9}")

    macro_f1 = np.mean(f1_scores)
    print(f"{indent}{'-'*55}")
    print(f"{indent}{'Macro F1':<22} {macro_f1:>7.4f}")
    return macro_f1


# ─────────────────────────────────────────────────────────────────────────────
# Main KD training loop
# ─────────────────────────────────────────────────────────────────────────────

def train_knowledge_distillation(
    teacher,
    student,
    train_loader,
    val_loader,
    test_loader,
    epochs,
    T,
    alpha,
    device,
    optimizer,
    #class_names,                   # list of class name strings for F1 report
    label_smoothing=0.0,
    patience=20,
    act='relu',
    range_penalty_weight=0.0,      # set > 0 to enable range penalty (bern only)
):
    """
    Knowledge distillation training loop.

    Returns
    -------
    dict with keys:
        val_acc       : best validation accuracy (%)
        test_acc      : final test accuracy (%)
        train_hard_loss : CE loss at best checkpoint epoch
        train_soft_loss : KL soft loss at best checkpoint epoch
        val_loss      : val loss at best checkpoint epoch
        macro_f1      : macro F1 on test set
        history       : dict of per-epoch lists (train_loss, soft_loss,
                        hard_loss, val_loss, val_acc)
    """
    teacher = teacher.to(device)
    student = student.to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    kl_loss   = nn.KLDivLoss(reduction="batchmean")
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    teacher.eval()

    history = {
        'train_loss': [],
        'soft_loss':  [],
        'hard_loss':  [],
        'val_loss':   [],
        'val_acc':    [],
    }

    best_state        = None
    best_val_acc      = -1.0
    best_soft_loss    = None
    best_hard_loss    = None
    best_val_loss     = None
    bad_epochs        = 0

    for epoch in range(epochs):

        # ── DEBUG reset ───────────────────────────────────────────────────────
        if act == "bern":
            for bl in student.get_bern_layers():
                bl._clamp_total = 0
                bl._clamp_count = 0
                bl.reset_xnorm_stats()

        # ── Train ─────────────────────────────────────────────────────────────
        student.train()
        running_loss = 0.0
        running_soft = 0.0
        running_hard = 0.0

        for data, targets in train_loader:
            data, targets = data.to(device), targets.to(device)
            optimizer.zero_grad(set_to_none=True)

            with torch.no_grad():
                teacher_logits = teacher(data)

            student_logits = student(data)

            soft_loss = kl_loss(
                nn.functional.log_softmax(student_logits / T, dim=1),
                nn.functional.softmax(teacher_logits / T, dim=1)
            ) * (T * T)

            hard_loss = criterion(student_logits, targets)
            loss      = alpha * soft_loss + (1 - alpha) * hard_loss

            if act == "bern" and range_penalty_weight > 0.0:
                loss = loss + range_penalty_weight * student.range_penalty()
                """ for bl in student.get_bern_layers():
                    loss = loss + 1e-4 * bl.pre_act_penalty """

            loss.backward()
            optimizer.step()

            running_loss += loss.item() * data.size(0)
            running_soft += soft_loss.item() * data.size(0)
            running_hard += hard_loss.item() * data.size(0)

        scheduler.step()

        # ── DEBUG print ───────────────────────────────────────────────────────
        if act == "bern":
            with torch.no_grad():
                for i, bl in enumerate(student.get_bern_layers()):
                    ratio = bl._clamp_total / bl._clamp_count if bl._clamp_count > 0 else 0.0
                    xmin, xmean, xmax = bl.get_xnorm_stats()
                    print(
                        f"[Epoch stats] Bern layer {i}: "
                        f"clamp={ratio:.4f}, "
                        f"x_norm min={xmin:.3f}, mean={xmean:.3f}, max={xmax:.3f}"
                    )

        n_train      = len(train_loader.dataset)
        epoch_loss   = running_loss / n_train
        epoch_soft   = running_soft / n_train
        epoch_hard   = running_hard / n_train

        history['train_loss'].append(epoch_loss)
        history['soft_loss'].append(epoch_soft)
        history['hard_loss'].append(epoch_hard)

        # ── Validate ──────────────────────────────────────────────────────────
        student.eval()
        correct, total   = 0, 0
        val_running_loss = 0.0
        with torch.no_grad():
            for data, targets in val_loader:
                data, targets = data.to(device), targets.to(device)
                outputs       = student(data)
                val_running_loss += criterion(outputs, targets).item() * data.size(0)
                correct += (outputs.argmax(dim=1) == targets).sum().item()
                total   += targets.size(0)

        val_loss = val_running_loss / len(val_loader.dataset)
        val_acc  = 100.0 * correct / total

        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        print(
            f"Epoch {epoch+1:>3}/{epochs} | "
            f"loss={epoch_loss:.4f} (soft={epoch_soft:.4f}, ce={epoch_hard:.4f}) | "
            f"val loss={val_loss:.4f} val acc={val_acc:.2f}%"
        )

        # ── Early stopping / checkpoint ───────────────────────────────────────
        if val_acc > best_val_acc + 0.01:
            best_val_acc   = val_acc
            best_soft_loss = epoch_soft
            best_hard_loss = epoch_hard
            best_val_loss  = val_loss
            best_state     = copy.deepcopy(student.state_dict())
            bad_epochs     = 0
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                print(f"Early stopping at epoch {epoch+1}. Best val acc: {best_val_acc:.2f}%")
                break

    # ── Restore best checkpoint ───────────────────────────────────────────────
    if best_state is not None:
        student.load_state_dict(best_state)

    # ── Final test evaluation with per-class F1 ───────────────────────────────
    test_acc  = None
    #macro_f1  = None

    if test_loader is not None:
        student.eval()
        all_preds, all_targets = [], []
        correct, total = 0, 0
        with torch.no_grad():
            for data, targets in test_loader:
                data, targets = data.to(device), targets.to(device)
                logits  = student(data)
                preds   = logits.argmax(dim=1)
                correct += (preds == targets).sum().item()
                total   += targets.size(0)
                all_preds.append(preds.cpu().numpy())
                all_targets.append(targets.cpu().numpy())

        test_acc    = 100.0 * correct / total
        all_preds   = np.concatenate(all_preds)
        all_targets = np.concatenate(all_targets)

        print(f"\n[KD Student] Test accuracy : {test_acc:.2f}%")
        #macro_f1 = classification_report_simple(all_preds, all_targets, class_names)
        print(f"\n[KD Student] Best val acc  : {best_val_acc:.2f}%")
        print(f"[KD Student] Test acc      : {test_acc:.2f}%")
        #print(f"[KD Student] Macro F1      : {macro_f1:.4f}")
        print(f"[KD Student] Best soft loss: {best_soft_loss:.4f}")
        print(f"[KD Student] Best hard loss: {best_hard_loss:.4f}")

    return {
        'val_acc':        best_val_acc,
        'test_acc':       test_acc,
        'train_soft_loss': best_soft_loss,
        'train_hard_loss': best_hard_loss,
        'val_loss':       best_val_loss,
        #'macro_f1':       macro_f1,
        'history':        history,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Single epoch KD train (used by kd_train_perlayer_bound)
# ─────────────────────────────────────────────────────────────────────────────

def kd_train_one_epoch(
    teacher,
    student,
    train_loader,
    T,
    alpha,
    criterion,
    optimizer,
    device,
    act='bern',
    range_penalty_weight=0.0,
):
    student.train()
    correct, total  = 0.0, 0
    running_loss    = 0.0
    ce_running_loss = 0.0
    kl_loss         = nn.KLDivLoss(reduction="batchmean")

    # ── DEBUG reset ───────────────────────────────────────────────────────────
    if act == "bern":
        for bl in student.get_bern_layers():
            bl._clamp_total = 0
            bl._clamp_count = 0
            bl.reset_xnorm_stats()

    for data, targets in train_loader:
        data, targets = data.to(device), targets.to(device)
        optimizer.zero_grad(set_to_none=True)

        with torch.no_grad():
            teacher_logits = teacher(data)

        student_logits = student(data)

        soft_loss = kl_loss(
            nn.functional.log_softmax(student_logits / T, dim=1),
            nn.functional.softmax(teacher_logits / T, dim=1)
        ) * (T * T)

        hard_loss = criterion(student_logits, targets)
        loss      = alpha * soft_loss + (1 - alpha) * hard_loss

        if act == "bern" and range_penalty_weight > 0.0:
            loss = loss + range_penalty_weight * student.range_penalty()
        """ for bl in student.get_bern_layers():
                loss = loss + 1e-4 * bl.pre_act_penalty """

        loss.backward()
        optimizer.step()

        ce_running_loss += hard_loss.item() * data.size(0)
        running_loss    += loss.item() * data.size(0)
        pred             = student_logits.argmax(dim=1)
        correct         += (pred == targets).sum().item()
        total           += data.size(0)

    # ── DEBUG print ───────────────────────────────────────────────────────────
    if act == "bern":
        for i, bl in enumerate(student.get_bern_layers()):
            ratio = bl._clamp_total / bl._clamp_count if bl._clamp_count > 0 else 0.0
            xmin, xmean, xmax = bl.get_xnorm_stats()
            print(
                f"[Epoch stats] Bern layer {i}: "
                f"clamp={ratio:.4f}, "
                f"x_norm min={xmin:.3f}, mean={xmean:.3f}, max={xmax:.3f}"
            )

    return ce_running_loss / total, correct / total

#evaluate one epoch
@torch.no_grad()
def eval_one_epoch(model, loader, criterion, device):
    
    model.eval()
    total_loss, correct, total = 0.0, 0, 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = criterion(logits, y)
        total_loss += loss.item() * x.size(0)
        pred = logits.argmax(dim=1)
        correct += (pred == y).sum().item()
        total += x.size(0)

    return total_loss / total, correct / total

# ─────────────────────────────────────────────────────────────────────────────
# Per-layer bound calibration with KD + scheduler
# ─────────────────────────────────────────────────────────────────────────────

def kd_train_perlayer_bound(
    student,
    teacher,
    train_loader,
    val_loader,
    optimizer,
    label_smoothing,
    T,
    alpha,
    device,
    warmup_epochs=1,
    stage_epochs=5,
    max_calib_batches=200,
    range_penalty_weight=0.0,
):
    student = student.to(device)
    teacher = teacher.to(device)
    teacher.eval()

    criterion    = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    bern_layers  = student.get_bern_layers()
    # ── Warmup ────────────────────────────────────────────────────────────────
    print("initial bounds: -7, 7")
    for bl in bern_layers:
        bl.input_bounds[..., 0].fill_(-7.0)
        bl.input_bounds[..., 1].fill_(7.0)
        bl.use_bounds = True

    for e in range(warmup_epochs):
        tr_loss, tr_acc = kd_train_one_epoch(
            teacher, student, train_loader, T, alpha,
            criterion, optimizer, device,
            act='bern', range_penalty_weight=range_penalty_weight,
        )
        
        te_loss, te_acc = eval_one_epoch(student, val_loader, criterion, device)
        print(
            f"Warmup {e+1}/{warmup_epochs} | "
            f"loss={tr_loss:.4f} train acc={tr_acc*100:.2f}% | "
            f"val acc={te_acc*100:.2f}%"
        )

    # ── Progressive calibration ───────────────────────────────────────────────
    for k in range(len(bern_layers)):
        student.calibrate_one_bern_layer_from_data(
            loader=train_loader,
            device=device,
            layer_idx=k,
            max_batches=max_calib_batches,
            p_lo=0.01,
            p_hi=0.99,
            min_width=0.5,
        )

        for e in range(stage_epochs):
            tr_loss, tr_acc = kd_train_one_epoch(
                teacher, student, train_loader, T, alpha,
                criterion, optimizer, device,
                act='bern', range_penalty_weight=range_penalty_weight,
            )
            te_loss, te_acc = eval_one_epoch(student, val_loader, criterion, device)
            print(
                f"  Layer {k} stage {e+1}/{stage_epochs} | "
                f"loss={tr_loss:.4f} train acc={tr_acc*100:.2f}% | "
                f"val acc={te_acc*100:.2f}%"
            )

              
def kd_train_models(
    architectures,
    activations,
    alphas,
    temps,
    degree,
    last_bern,
    learning_rate,
    weight_decay,
    train_loader,
    val_loader,
    test_loader,
    #class_names,                   # list of class name strings for F1 report
    epochs,
    seed,
    teacher,                       # teacher model (must be trained)
    file_name=None,
    range_penalty_weight=0.0,      # set > 0 to enable range penalty for bern
):
    all_results = []

    for layer_sizes in architectures:
        print("=" * 80)
        for alpha in alphas:
            Ts = [1.0] if alpha == 0.0 else temps  # T irrelevant if no KD
            for T in Ts:
                for act in activations:
                    print(f"\nArchitecture : {layer_sizes}")
                    print(f"act={act}  alpha={alpha}  T={T}")

                    torch.manual_seed(seed)

                    student = FCModel(
                        layer_sizes=layer_sizes,
                        degree=degree,
                        act=act,
                        last_bern=last_bern,
                    ).to(device)

                   
                    optimizer = optim.AdamW(student.parameters(), lr=learning_rate, weight_decay=weight_decay)

                    # ── Bernstein: staged calibration ─────────────────────────
                    if act == "bern":
                        print("→ Running per-layer Bernstein calibration")
                        kd_train_perlayer_bound(
                            student=student,
                            teacher=teacher,
                            train_loader=train_loader,
                            val_loader=val_loader,
                            optimizer=optimizer,
                            label_smoothing=0.0,
                            T=T,
                            alpha=alpha,
                            device=device,
                            warmup_epochs=2,
                            stage_epochs=5,
                            max_calib_batches=1000,
                            range_penalty_weight=range_penalty_weight,
                        )
                        # Reset optimizer state after calibration phase
                        for group in optimizer.param_groups:
                            for p in group['params']:
                                if p in optimizer.state:
                                    optimizer.state[p] = {}
                            group['lr'] = learning_rate

                    # ── Full KD training ──────────────────────────────────────
                    print("→ Training model")
                    results = train_knowledge_distillation(
                        teacher=teacher,
                        student=student,
                        train_loader=train_loader,
                        val_loader=val_loader,
                        test_loader=test_loader,
                        epochs=epochs,
                        T=T,
                        alpha=alpha,
                        device=device,
                        optimizer=optimizer,
                        label_smoothing=0.0,
                        patience=20,
                        act=act,
                        range_penalty_weight=range_penalty_weight,
                    )

                    # Unpack results dict
                    val_acc        = results['val_acc']
                    test_acc       = results['test_acc']
                    val_loss       = results['val_loss']
                    train_hard     = results['train_hard_loss']
                    train_soft     = results['train_soft_loss']

                    # ── Save student weights ───────────────────────────────────
                    save_dir = "saved_models_kd"
                    os.makedirs(save_dir, exist_ok=True)

                    arch_str  = "x".join(map(str, layer_sizes))
                    deg_str   = f"deg{degree}" if act == "bern" else "degNA"
                    alpha_str = f"alpha{alpha:g}"
                    T_str     = f"T{T:g}"
                    lr_str    = f"lr{learning_rate:g}"
                    wd_str    = f"wd{weight_decay:g}"
                    seed_str  = f"seed{seed}"

                    pth_name = _safe(
                        f"kd_fc_{arch_str}_{act}_{deg_str}_{alpha_str}_{T_str}_"
                        f"{lr_str}_{wd_str}_{seed_str}.pth"
                    )
                    pth_path = os.path.join(save_dir, pth_name)

                    torch.save(
                        {
                            "state_dict":    student.state_dict(),
                            "arch":          layer_sizes,
                            "activation":    act,
                            "degree":        degree if act == "bern" else None,
                            "alpha":         alpha,
                            "T":             T,
                            "last_bern":     last_bern,
                            "learning_rate": learning_rate,
                            "weight_decay":  weight_decay,
                            "seed":          seed,
                            "val_acc":       float(val_acc),
                            "test_acc":      float(test_acc),
                            "ce_train_loss":   float(train_hard) if train_hard else None,
                            "soft_train_loss": float(train_soft) if train_soft else None,
                            "val_loss":        float(val_loss)   if val_loss   else None,
                        },
                        pth_path,
                    )

                    # ── Log results ────────────────────────────────────────────
                    hidden_layer_sizes = layer_sizes[1:-1]

                    row = {
                        "activation":      act,
                        "layer_sizes":     str(hidden_layer_sizes),
                        "degree":          degree if act == "bern" else None,
                        "alpha":           alpha,
                        "T":               T,
                        "learning_rate":   learning_rate,
                        "weight_decay":    weight_decay,
                        "ce_train_loss":   float(train_hard) if train_hard else None,
                        "soft_train_loss": float(train_soft) if train_soft else None,
                        "val_accuracy_%":  float(val_acc),
                        "val_loss":        float(val_loss) if val_loss else None,
                        "test_accuracy_%": float(test_acc) if test_acc else None,
                        #"macro_f1":        float(macro_f1) if macro_f1 else None,
                        "fold_idx":        seed - 1000,
                        "model_path":      pth_path,
                    }

                    # Append row to CSV
                    if file_name is not None:
                        df_row = pd.DataFrame([row])
                        write_header = not pd.io.common.file_exists(file_name)
                        df_row.to_csv(file_name, index=False, mode='a', header=write_header)

                    all_results.append(row)

    all_df = pd.DataFrame(all_results)
    print("\n===== Results =====")
    print(all_df.to_string(index=False))
    return all_df