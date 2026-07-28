"""
Stage 2 — Fine-tuned function matching.

Warm-start from a Stage-1 checkpoint and continue isolation training, but now
against the SST-2-FINE-TUNED teacher's FFN distribution (not the general model).
Bounds are recalibrated on the fine-tuned distribution first. This is the
configurable consolidation of exp12a/exp13b.

Why a separate stage: fine-tuning shifts the FFN function substantially
(layer 0 by ~468x MSE if cold-evaluated). Training against the deployment-time
distribution is what makes the later cold-substitution (Stage 3) work without
any task re-training.

Recipe: 30 epochs, exp7c param groups (linear lr=1e-3 wd=0.01, bern lr=3e-3
wd=0.0), CosineAnnealingLR(T_max=epochs), recalib_every=5, freeze_epoch=15.

Example:
  python Transformer/stage2_finetuned_match.py --layer 3 --hidden 600 --degree 15 \
      --stage1-ckpt Transformer/models/stage1/layer3_bern_d15_h600_best.pt \
      --teacher-ckpt Transformer/models/teacher_gelu_9037.pt \
      --out Transformer/models/stage2/layer3_d15_h600.pt
"""

import argparse
import json
import os

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

import shared_utils as U


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Stage 2: fine-tuned FFN function match")
    p.add_argument("--task", type=str, default="sst2",
                   help="GLUE task the teacher was fine-tuned on")
    p.add_argument("--train-subset", type=int, default=None,
                   help="cap train examples (bounds --cache memory on large tasks)")
    p.add_argument("--layer", type=int, required=True)
    p.add_argument("--hidden", type=int, default=600)
    p.add_argument("--degree", type=int, default=15)
    p.add_argument("--act", choices=["bern", "relu", "gelu"], default="bern")
    p.add_argument("--stage1-ckpt", type=str, required=True,
                   help="Stage-1 checkpoint to warm-start from")
    p.add_argument("--teacher-ckpt", type=str,
                   default=U.models_path("teacher_gelu_9037.pt"),
                   help="SST-2-fine-tuned teacher (state under 'bert' key)")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--max-length", type=int, default=64)
    p.add_argument("--recalib-every", type=int, default=5)
    p.add_argument("--freeze-epoch", type=int, default=15)
    p.add_argument("--lr-linear", type=float, default=1e-3)
    p.add_argument("--lr-bern", type=float, default=3e-3)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--min-width", type=float, default=0.5)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--cal-batches", type=int, default=100)
    p.add_argument("--cache", action="store_true",
                   help="capture FFN I/O once on-device and train on it "
                        "(faster, equivalent; default off keeps the online path)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", type=str, default=None)
    p.add_argument("--log", type=str, default=None)
    p.add_argument("--results-json", type=str, default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()

    tag = f"layer{args.layer}_{args.act}_d{args.degree}_h{args.hidden}"
    out_ckpt = args.out or U.models_path("stage2", f"{tag}_best.pt")
    log_path = args.log or U.results_path(f"stage2_{tag}.log")
    json_path = args.results_json or U.results_path(f"stage2_{tag}.json")
    os.makedirs(os.path.dirname(out_ckpt) or ".", exist_ok=True)
    os.makedirs("results", exist_ok=True)

    import sys
    sys.stdout = U.Tee(log_path)

    U.set_seed(args.seed)
    device = U.get_device()
    print("=" * 64)
    print(f"Stage 2 — fine-tuned FFN match | task={args.task} | {tag}")
    print(f"  warm-start: {args.stage1_ckpt}")
    print(f"  teacher:    {args.teacher_ckpt}")
    print(f"  out={out_ckpt}")
    U.require_gpu("Stage 2 training")
    assert os.path.exists(args.stage1_ckpt), f"missing {args.stage1_ckpt}"
    assert os.path.exists(args.teacher_ckpt), f"missing {args.teacher_ckpt}"

    num_labels = U.GLUE_TASKS[args.task]["num_labels"]
    tokenizer = U.build_tokenizer()
    train_loader, val_loader = U.make_glue_loaders(
        tokenizer, args.task, args.batch_size, args.max_length,
        with_labels=False, train_subset=args.train_subset)

    teacher = U.load_finetuned_teacher(args.teacher_ckpt, device, freeze=True,
                                       num_labels=num_labels)
    layer = U.encoder_layers(teacher)[args.layer]

    use_bern = args.act == "bern"
    fc = U.build_fc(args.hidden, args.degree, device, act=args.act)
    fc.load_state_dict(torch.load(args.stage1_ckpt, map_location=device))
    n_params = sum(p.numel() for p in fc.parameters())
    print(f"  FCModel params={n_params:,}  act={args.act}")

    criterion = nn.MSELoss()

    # Pre-Stage-2 MSE on the fine-tuned distribution (== cold-swap MSE).
    fc.eval()
    pre_mse = 0.0
    with torch.no_grad():
        for batch in val_loader:
            fin, ftgt = U.capture_ffn_io(teacher, layer, batch, device)
            pre_mse += criterion(fc(fin), ftgt).item()
    pre_mse /= len(val_loader)
    print(f"  pre-Stage-2 (cold) val_mse = {pre_mse:.6f}")

    cal_loader = None
    if use_bern:
        cal_loader, cal_flat = U.build_cal_loader(
            teacher, layer, train_loader, device, n_batches=args.cal_batches)
        print(f"  cal vectors={len(cal_flat):,}  mean={cal_flat.mean():.4f} "
              f"std={cal_flat.std():.4f}")
        U.calibrate_all_bern_layers(fc, cal_loader, device, min_width=args.min_width)

    bern_params = [p for n, p in fc.named_parameters() if "bern_coeffs" in n]
    other_params = [p for n, p in fc.named_parameters() if "bern_coeffs" not in n]
    groups = [{"params": other_params, "lr": args.lr_linear,
               "weight_decay": args.weight_decay}]
    if bern_params:
        groups.append({"params": bern_params, "lr": args.lr_bern, "weight_decay": 0.0})
    optimizer = AdamW(groups)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    # Optional one-time FFN I/O cache (frozen teacher + fixed data).
    tok_bs = args.batch_size * args.max_length
    cache = None
    if args.cache:
        print("  [cache] capturing FFN I/O once on-device ...")
        Xtr, Ytr = U.cache_ffn_io(teacher, layer, train_loader, device)
        Xval, Yval = U.cache_ffn_io(teacher, layer, val_loader, device)
        n_train = (Xtr.size(0) + tok_bs - 1) // tok_bs
        n_val = (Xval.size(0) + tok_bs - 1) // tok_bs
        cache = (Xtr, Ytr, Xval, Yval, n_train, n_val)
        print(f"  [cache] train_vectors={Xtr.size(0):,} val_vectors={Xval.size(0):,} "
              f"steps/epoch={n_train}")

    results = {"train_mse": [], "val_mse": [], "pre_stage2_val_mse": pre_mse,
               "config": vars(args)}
    best_val = float("inf")

    for epoch in range(args.epochs):
        if (use_bern and epoch > 0 and epoch % args.recalib_every == 0
                and epoch <= args.freeze_epoch):
            print(f"  [recalibrate @ epoch {epoch + 1}]")
            U.calibrate_all_bern_layers(fc, cal_loader, device,
                                        min_width=args.min_width)
            if epoch == args.freeze_epoch:
                print(f"  [bounds frozen from epoch {epoch + 1}]")

        if use_bern:
            U.reset_bern_stats(fc)
        fc.train()
        train_loss = 0.0
        if cache is not None:
            Xtr, Ytr, _, _, n_train, _ = cache
            for x, y in U.iter_cached(Xtr, Ytr, tok_bs, True, device):
                optimizer.zero_grad(set_to_none=True)
                loss = criterion(fc(x), y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(fc.parameters(), max_norm=args.grad_clip)
                optimizer.step()
                train_loss += loss.item()
            train_loss /= n_train
        else:
            for batch in train_loader:
                fin, ftgt = U.capture_ffn_io(teacher, layer, batch, device)
                optimizer.zero_grad(set_to_none=True)
                pred = fc(fin)
                loss = criterion(pred, ftgt)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(fc.parameters(), max_norm=args.grad_clip)
                optimizer.step()
                train_loss += loss.item()
            train_loss /= len(train_loader)

        fc.eval()
        val_loss = 0.0
        with torch.no_grad():
            if cache is not None:
                _, _, Xval, Yval, _, n_val = cache
                for x, y in U.iter_cached(Xval, Yval, tok_bs, False, device):
                    val_loss += criterion(fc(x), y).item()
                val_loss /= n_val
            else:
                for batch in val_loader:
                    fin, ftgt = U.capture_ffn_io(teacher, layer, batch, device)
                    val_loss += criterion(fc(fin), ftgt).item()
                val_loss /= len(val_loader)

        if val_loss < best_val:
            best_val = val_loss
            torch.save(fc.state_dict(), out_ckpt)

        scheduler.step()
        results["train_mse"].append(train_loss)
        results["val_mse"].append(val_loss)

        if epoch % 3 == 0 or epoch == args.epochs - 1:
            marker = " *" if val_loss == best_val else ""
            print(f"  epoch {epoch + 1:02d}/{args.epochs}  train={train_loss:.6f}  "
                  f"val={val_loss:.6f}  lr={optimizer.param_groups[0]['lr']:.2e}  "
                  f"clamp={U.clamp_ratio(fc):.3f}{marker}")

    results.update({"best_val_mse": best_val, "n_params": n_params,
                    "checkpoint": out_ckpt})
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)

    print("=" * 64)
    print(f"Pre-Stage-2 (cold) val_mse : {pre_mse:.6f}")
    print(f"Best Stage-2 val_mse       : {best_val:.6f}")
    print(f"Checkpoint                 : {out_ckpt}")


if __name__ == "__main__":
    main()
