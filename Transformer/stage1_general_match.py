"""
Stage 1 — General function matching.

Pre-train a Bernstein (or ReLU) FFN to approximate one TinyBERT4 FFN layer's
input->output mapping in isolation, against the GENERAL pretrained model's
activation distribution (BertModel, no task head). MSE loss, no end-to-end
training. This is the configurable consolidation of exp7b/exp10a/exp11a/exp13a.

Recipe (proven — see tasks/lessons.md):
  - param groups: linear lr=1e-3 wd=0.01, bern_coeffs lr=3e-3 wd=0.0
  - CosineAnnealingLR(T_max=epochs, eta_min=1e-6); never T_max > epochs (L4)
  - range_penalty_weight = 0 (L2)
  - recalibrate bounds every `--recalib-every`, freeze at `--freeze-epoch`
  - always save the best-val checkpoint (L5)

Example:
  python Transformer/stage1_general_match.py --layer 3 --hidden 600 --degree 15 \
      --epochs 100 --out Transformer/models/stage1/layer3_d15_h600.pt
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
    p = argparse.ArgumentParser(description="Stage 1: general FFN function match")
    p.add_argument("--layer", type=int, required=True,
                   help="TinyBERT encoder layer index to approximate (0-3)")
    p.add_argument("--hidden", type=int, default=600,
                   help="FFN intermediate width (312->hidden->312)")
    p.add_argument("--degree", type=int, default=15, help="Bernstein degree")
    p.add_argument("--act", choices=["bern", "relu", "gelu"], default="bern")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--max-length", type=int, default=64)
    p.add_argument("--recalib-every", type=int, default=10)
    p.add_argument("--freeze-epoch", type=int, default=30,
                   help="must be divisible by --recalib-every (L6)")
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
    p.add_argument("--out", type=str, default=None,
                   help="best-val checkpoint path (default: auto under Transformer/models/stage1/)")
    p.add_argument("--log", type=str, default=None)
    p.add_argument("--results-json", type=str, default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()

    tag = f"layer{args.layer}_{args.act}_d{args.degree}_h{args.hidden}"
    out_ckpt = args.out or U.models_path("stage1", f"{tag}_best.pt")
    log_path = args.log or U.results_path(f"stage1_{tag}.log")
    json_path = args.results_json or U.results_path(f"stage1_{tag}.json")
    os.makedirs(os.path.dirname(out_ckpt) or ".", exist_ok=True)
    os.makedirs("results", exist_ok=True)

    import sys
    sys.stdout = U.Tee(log_path)

    U.set_seed(args.seed)
    device = U.get_device()
    print("=" * 64)
    print(f"Stage 1 — general FFN match | {tag}")
    print(f"  device={device}  epochs={args.epochs}  batch={args.batch_size}")
    print(f"  recalib_every={args.recalib_every}  freeze_epoch={args.freeze_epoch}")
    print(f"  out={out_ckpt}")
    U.require_gpu("Stage 1 training")
    if args.act == "bern" and args.freeze_epoch % args.recalib_every != 0:
        print("WARNING: freeze_epoch not divisible by recalib_every (see lesson L6)")

    tokenizer = U.build_tokenizer()
    train_loader, val_loader = U.make_sst2_loaders(
        tokenizer, args.batch_size, args.max_length, with_labels=False)
    print(f"  train batches={len(train_loader)}  val batches={len(val_loader)}")

    tinybert = U.load_general_tinybert(device)
    layer = U.encoder_layers(tinybert)[args.layer]

    fc = U.build_fc(args.hidden, args.degree, device, act=args.act)
    n_params = sum(p.numel() for p in fc.parameters())
    print(f"  FCModel params={n_params:,}")

    use_bern = args.act == "bern"
    cal_loader = None
    if use_bern:
        cal_loader, cal_flat = U.build_cal_loader(
            tinybert, layer, train_loader, device, n_batches=args.cal_batches)
        print(f"  cal vectors={len(cal_flat):,}  mean={cal_flat.mean():.4f} "
              f"std={cal_flat.std():.4f}")
        U.init_bounds(fc)
        U.calibrate_all_bern_layers(fc, cal_loader, device, min_width=args.min_width)

    bern_params = [p for n, p in fc.named_parameters() if "bern_coeffs" in n]
    other_params = [p for n, p in fc.named_parameters() if "bern_coeffs" not in n]
    groups = [{"params": other_params, "lr": args.lr_linear,
               "weight_decay": args.weight_decay}]
    if bern_params:
        groups.append({"params": bern_params, "lr": args.lr_bern,
                       "weight_decay": 0.0})
    optimizer = AdamW(groups)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    criterion = nn.MSELoss()

    # Optional one-time FFN I/O cache (frozen teacher + fixed data => identical
    # targets every epoch). Same per-step token count as the online path.
    tok_bs = args.batch_size * args.max_length
    cache = None
    if args.cache:
        print("  [cache] capturing FFN I/O once on-device ...")
        Xtr, Ytr = U.cache_ffn_io(tinybert, layer, train_loader, device)
        Xval, Yval = U.cache_ffn_io(tinybert, layer, val_loader, device)
        n_train = (Xtr.size(0) + tok_bs - 1) // tok_bs
        n_val = (Xval.size(0) + tok_bs - 1) // tok_bs
        cache = (Xtr, Ytr, Xval, Yval, n_train, n_val)
        print(f"  [cache] train_vectors={Xtr.size(0):,} val_vectors={Xval.size(0):,} "
              f"steps/epoch={n_train}")

    results = {"train_mse": [], "val_mse": [], "config": vars(args)}
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
                ffn_in, ffn_tgt = U.capture_ffn_io(tinybert, layer, batch, device)
                optimizer.zero_grad(set_to_none=True)
                pred = fc(ffn_in)
                loss = criterion(pred, ffn_tgt)
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
                    ffn_in, ffn_tgt = U.capture_ffn_io(tinybert, layer, batch, device)
                    val_loss += criterion(fc(ffn_in), ffn_tgt).item()
                val_loss /= len(val_loader)

        if val_loss < best_val:
            best_val = val_loss
            torch.save(fc.state_dict(), out_ckpt)

        scheduler.step()
        results["train_mse"].append(train_loss)
        results["val_mse"].append(val_loss)

        if epoch % 5 == 0 or epoch == args.epochs - 1:
            marker = " *" if val_loss == best_val else ""
            extra = ""
            if use_bern:
                extra = f"  clamp={U.clamp_ratio(fc):.3f}"
            print(f"  epoch {epoch + 1:03d}/{args.epochs}  train={train_loss:.6f}  "
                  f"val={val_loss:.6f}  lr={optimizer.param_groups[0]['lr']:.2e}"
                  f"{extra}{marker}")

    overfit = results["val_mse"][-1] / max(results["train_mse"][-1], 1e-9)
    results.update({"best_val_mse": best_val, "n_params": n_params,
                    "overfit_ratio": overfit, "checkpoint": out_ckpt})
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)

    print("=" * 64)
    print(f"Best val MSE : {best_val:.6f}")
    print(f"Overfit ratio: {overfit:.3f}")
    print(f"Checkpoint   : {out_ckpt}")
    print(f"Results JSON : {json_path}")


if __name__ == "__main__":
    main()
