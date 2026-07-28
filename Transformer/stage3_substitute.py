"""
Stage 3 — Cold substitution (+ optional KD fine-tune).

Start from the SST-2-fine-tuned teacher itself, cold-substitute the Stage-2
Bernstein FFNs into the chosen encoder layers, then either:
  * evaluate directly (--no-kd) — the "swap-and-ship" LUT deployment (Test A), or
  * KD fine-tune for a few epochs (--kd, default) — recovers/exceeds teacher
    accuracy (Test B).

No recalibration anywhere: Stage-2 bounds are already calibrated on the
fine-tuned distribution; recalibrating at substitution time hurts (exp12b).

Consolidation of exp12b_test_a / exp12c_test_b / exp13c / exp13d_v2.

Examples:
  # 4-layer cold swap, no fine-tuning (Test A):
  python Transformer/stage3_substitute.py --no-kd \
      --layers 0 1 2 3 \
      --layer-ckpts Transformer/models/stage2/layer0_d15_h600_best.pt \
                    Transformer/models/stage2/layer1_d15_h600_best.pt \
                    Transformer/models/stage2/layer2_d15_h600_best.pt \
                    Transformer/models/stage2/layer3_d15_h600_best.pt

  # 4-layer + KD, exp13d_v2 recipe (alpha=0.7, L3 bern_lr 10x, warmup 5%):
  python Transformer/stage3_substitute.py --kd --layers 0 1 2 3 \
      --layer-ckpts ... --alpha 0.7 --bern-lr-l3 1.2e-3 --warmup-frac 0.05 \
      --out Transformer/models/bern_4layer_best.pt
"""

import argparse
import json
import math
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW

import shared_utils as U

_kl = nn.KLDivLoss(reduction="batchmean")
_ce = nn.CrossEntropyLoss()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Stage 3: cold substitution + optional KD")
    p.add_argument("--task", type=str, default="sst2",
                   help="GLUE task the teacher was fine-tuned on")
    p.add_argument("--layers", type=int, nargs="+", required=True,
                   help="encoder layer indices to replace, e.g. 0 1 2 3")
    p.add_argument("--layer-ckpts", type=str, nargs="+", required=True,
                   help="Stage-2 checkpoint per --layers entry (same order)")
    p.add_argument("--teacher-ckpt", type=str,
                   default=U.models_path("teacher_gelu_9037.pt"))
    p.add_argument("--hidden", type=int, default=600)
    p.add_argument("--degree", type=int, default=15)
    p.add_argument("--act", choices=["bern", "relu", "gelu"], default="bern")

    kd = p.add_mutually_exclusive_group()
    kd.add_argument("--kd", dest="kd", action="store_true",
                    help="KD fine-tune after substitution (default)")
    kd.add_argument("--no-kd", dest="kd", action="store_false",
                    help="cold-swap only: evaluate, do not fine-tune")
    p.set_defaults(kd=True)

    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--max-length", type=int, default=64)
    p.add_argument("--temperature", type=float, default=4.0)
    p.add_argument("--alpha", type=float, default=0.5,
                   help="KD soft-loss weight (exp13d_v2 used 0.7 for 4-layer)")
    p.add_argument("--bert-lr", type=float, default=2e-5)
    p.add_argument("--bern-lr", type=float, default=1.2e-4)
    p.add_argument("--bern-lr-l3", type=float, default=None,
                   help="override bern_lr for layer 3 (exp13d_v2 used 1.2e-3)")
    p.add_argument("--warmup-frac", type=float, default=0.1)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", type=str, default=None)
    p.add_argument("--log", type=str, default=None)
    p.add_argument("--results-json", type=str, default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    assert len(args.layers) == len(args.layer_ckpts), \
        "--layers and --layer-ckpts must have the same length"

    mode = "kd" if args.kd else "cold"
    tag = f"{args.act}_{len(args.layers)}layer_{mode}_h{args.hidden}"
    out_ckpt = args.out or U.models_path(f"{tag}_best.pt")
    log_path = args.log or U.results_path(f"stage3_{tag}.log")
    json_path = args.results_json or U.results_path(f"stage3_{tag}.json")
    os.makedirs(os.path.dirname(out_ckpt) or ".", exist_ok=True)
    os.makedirs("results", exist_ok=True)

    import sys
    sys.stdout = U.Tee(log_path)

    U.set_seed(args.seed)
    device = U.get_device()
    print("=" * 64)
    print(f"Stage 3 — cold substitution ({'+ KD' if args.kd else 'no fine-tune'})")
    print(f"  task={args.task}  layers={args.layers}  hidden={args.hidden}  degree={args.degree}")
    for li, ck in zip(args.layers, args.layer_ckpts):
        assert os.path.exists(ck), f"missing {ck}"
        print(f"    layer {li}: {ck}")
    U.require_gpu("Stage 3 training")

    num_labels = U.GLUE_TASKS[args.task]["num_labels"]
    tokenizer = U.build_tokenizer()
    train_loader, val_loader = U.make_glue_loaders(
        tokenizer, args.task, args.batch_size, args.max_length, with_labels=True)

    teacher = U.load_finetuned_teacher(args.teacher_ckpt, device, freeze=True,
                                       num_labels=num_labels)
    teacher_acc = U.evaluate_classifier(teacher, val_loader, device)
    print(f"  teacher val_acc={teacher_acc:.4f}")

    # Student starts from the teacher itself, then FFNs are cold-substituted.
    student = U.load_finetuned_teacher(args.teacher_ckpt, device, freeze=False,
                                       num_labels=num_labels)
    fcs = {li: U.load_fc(ck, args.hidden, args.degree, device, act=args.act)
           for li, ck in zip(args.layers, args.layer_ckpts)}
    for li in args.layers:
        U.patch_ffn(student, li, fcs[li])
    cold_acc = U.evaluate_classifier(student, val_loader, device)
    print(f"  cold-substituted val_acc={cold_acc:.4f} "
          f"({(cold_acc - teacher_acc) * 100:+.2f}pp vs teacher)")

    summary = {"teacher_acc": teacher_acc, "cold_acc": cold_acc,
               "config": vars(args)}

    if not args.kd:
        # Cold deployment — save and stop.
        torch.save({"bert": student.state_dict(),
                    **{f"fc{li}": fc.state_dict() for li, fc in fcs.items()}},
                   out_ckpt)
        summary["best_val_acc"] = cold_acc
        summary["checkpoint"] = out_ckpt
        with open(json_path, "w") as f:
            json.dump(summary, f, indent=2)
        print("=" * 64)
        print(f"Cold-swap (no fine-tune). val_acc={cold_acc:.4f}")
        print(f"Checkpoint: {out_ckpt}")
        return

    # ── KD fine-tune ────────────────────────────────────────────────────────
    from transformers import get_cosine_schedule_with_warmup

    no_decay = ["bias", "LayerNorm.weight"]
    bert_decay = [p for n, p in student.named_parameters()
                  if not any(nd in n for nd in no_decay)]
    bert_nodecay = [p for n, p in student.named_parameters()
                    if any(nd in n for nd in no_decay)]

    # Optionally split layer 3's bern_coeffs into its own (higher-LR) group.
    l3_special = args.bern_lr_l3 is not None and 3 in fcs
    bern_std, bern_l3, fc_linear = [], [], []
    for li, fc in fcs.items():
        coeffs = [p for n, p in fc.named_parameters() if "bern_coeffs" in n]
        fc_linear += [p for n, p in fc.named_parameters() if "bern_coeffs" not in n]
        if l3_special and li == 3:
            bern_l3 += coeffs
        else:
            bern_std += coeffs

    groups = [
        {"params": bert_decay + fc_linear, "lr": args.bert_lr, "weight_decay": 0.01},
        {"params": bert_nodecay, "lr": args.bert_lr, "weight_decay": 0.0},
        {"params": bern_std, "lr": args.bern_lr, "weight_decay": 0.0},
    ]
    if l3_special:
        groups.append({"params": bern_l3, "lr": args.bern_lr_l3, "weight_decay": 0.0})
    # GeLU/ReLU FFNs have no bern_coeffs -> drop empty groups (AdamW rejects them).
    groups = [g for g in groups if g["params"]]
    optimizer = AdamW(groups)
    print(f"  optimizer: bert_lr={args.bert_lr} bern_lr={args.bern_lr}"
          + (f" bern_lr_l3={args.bern_lr_l3}" if l3_special else ""))

    total_steps = len(train_loader) * args.epochs
    warmup_steps = math.floor(total_steps * args.warmup_frac)
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    T, alpha = args.temperature, args.alpha
    print(f"  KD: epochs={args.epochs} T={T} alpha={alpha} warmup_frac={args.warmup_frac}")

    best_val = 0.0
    history = {"train_loss": [], "val_acc": [], "clamp": [], "bgnorm": []}
    all_params = list(student.parameters()) + \
        [p for fc in fcs.values() for p in fc.parameters()]

    for epoch in range(args.epochs):
        student.train()
        for fc in fcs.values():
            fc.train()
            U.reset_bern_stats(fc)

        epoch_loss = 0.0
        gn = {li: 0.0 for li in args.layers}
        for batch in train_loader:
            inputs = U.batch_inputs(batch, device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad(set_to_none=True)
            with torch.no_grad():
                t_logits = teacher(**inputs).logits
            s_logits = student(**inputs).logits

            hard = _ce(s_logits, labels)
            soft = _kl(F.log_softmax(s_logits / T, dim=1),
                       F.softmax(t_logits / T, dim=1)) * (T * T)
            loss = alpha * soft + (1 - alpha) * hard
            loss.backward()
            for li in args.layers:
                gn[li] += U.bern_grad_norm(fcs[li])
            torch.nn.utils.clip_grad_norm_(all_params, max_norm=args.grad_clip)
            optimizer.step()
            scheduler.step()
            epoch_loss += loss.item()

        n = len(train_loader)
        epoch_loss /= n
        for li in gn:
            gn[li] /= n

        val_acc = U.evaluate_classifier(student, val_loader, device)
        clamps = {li: U.clamp_ratio(fcs[li]) for li in args.layers}
        if val_acc > best_val:
            best_val = val_acc
            torch.save({"bert": student.state_dict(),
                        **{f"fc{li}": fc.state_dict() for li, fc in fcs.items()}},
                       out_ckpt)

        history["train_loss"].append(epoch_loss)
        history["val_acc"].append(val_acc)
        history["clamp"].append(clamps)
        history["bgnorm"].append(gn)
        marker = " *" if val_acc == best_val else ""
        cl = ",".join(f"{clamps[li]:.3f}" for li in args.layers)
        bg = ",".join(f"{gn[li]:.4f}" for li in args.layers)
        print(f"  epoch {epoch + 1}/{args.epochs}  loss={epoch_loss:.4f}  "
              f"val_acc={val_acc:.4f}  lr={optimizer.param_groups[0]['lr']:.2e}  "
              f"clamp=[{cl}]  bgnorm=[{bg}]{marker}")

    summary.update({"best_val_acc": best_val, "history": history,
                    "checkpoint": out_ckpt})
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)

    print("=" * 64)
    print(f"Teacher       : {teacher_acc:.4f}")
    print(f"Cold (no FT)  : {cold_acc:.4f}")
    print(f"Best KD val   : {best_val:.4f} ({(best_val - teacher_acc) * 100:+.2f}pp vs teacher)")
    print(f"Checkpoint    : {out_ckpt}")


if __name__ == "__main__":
    main()
