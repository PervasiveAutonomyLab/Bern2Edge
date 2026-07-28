"""
Task-generalized teacher fine-tune.

Fine-tunes a fresh TinyBERT4 BertForSequenceClassification on a GLUE task to
produce the teacher that Stage 2 (function-match) and Stage 3 (substitute + KD)
consume. This is a faithful reuse of the SST-2 recipe in experiments/exp8d_run.py
(5 epochs, AdamW lr=2e-5 with no-decay on bias/LayerNorm, cosine schedule + 10%
warmup, CE loss), generalized over GLUE_TASKS so the same recipe trains MRPC,
RTE and QNLI teachers.

The best-val checkpoint is saved as {"bert": state_dict()} — the format
load_finetuned_teacher expects.

Example:
  python Transformer/finetune_teacher.py --task mrpc --max-length 128
  # -> Transformer/models/teacher_mrpc.pt
"""

import argparse
import json
import math
import os
import sys

import torch
from torch.optim import AdamW

import shared_utils as U


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fine-tune a TinyBERT4 teacher on a GLUE task")
    p.add_argument("--task", type=str, required=True, choices=sorted(U.GLUE_TASKS))
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--max-length", type=int, default=128)
    p.add_argument("--bert-lr", type=float, default=2e-5)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--warmup-frac", type=float, default=0.1)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", type=str, default=None)
    p.add_argument("--log", type=str, default=None)
    p.add_argument("--results-json", type=str, default=None)
    return p.parse_args()


def build_optimizer(model, bert_lr: float, weight_decay: float) -> AdamW:
    """Standard BERT fine-tune param groups: no weight decay on bias/LayerNorm."""
    no_decay = ["bias", "LayerNorm.weight"]
    decay = [p for n, p in model.named_parameters()
             if not any(nd in n for nd in no_decay)]
    nodecay = [p for n, p in model.named_parameters()
               if any(nd in n for nd in no_decay)]
    return AdamW([
        {"params": decay, "lr": bert_lr, "weight_decay": weight_decay},
        {"params": nodecay, "lr": bert_lr, "weight_decay": 0.0},
    ])


def main() -> None:
    args = parse_args()

    out_ckpt = args.out or U.models_path(f"teacher_{args.task}.pt")
    log_path = args.log or U.results_path(f"finetune_teacher_{args.task}.log")
    json_path = args.results_json or U.results_path(f"finetune_teacher_{args.task}.json")
    os.makedirs(os.path.dirname(out_ckpt) or ".", exist_ok=True)
    os.makedirs("results", exist_ok=True)

    sys.stdout = U.Tee(log_path)

    U.set_seed(args.seed)
    device = U.get_device()
    num_labels = U.GLUE_TASKS[args.task]["num_labels"]
    print("=" * 64)
    print(f"Fine-tune teacher | task={args.task}  num_labels={num_labels}")
    print(f"  epochs={args.epochs} bert_lr={args.bert_lr} max_length={args.max_length}")
    print(f"  out={out_ckpt}")
    U.require_gpu("Teacher fine-tuning")

    from transformers import get_cosine_schedule_with_warmup

    tokenizer = U.build_tokenizer()
    train_loader, val_loader = U.make_glue_loaders(
        tokenizer, args.task, args.batch_size, args.max_length, with_labels=True)
    print(f"  train batches={len(train_loader)}  val batches={len(val_loader)}")

    model = U.load_fresh_teacher(device, num_labels=num_labels)

    optimizer = build_optimizer(model, args.bert_lr, args.weight_decay)
    total_steps = len(train_loader) * args.epochs
    warmup_steps = math.floor(total_steps * args.warmup_frac)
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    best_val = 0.0
    history = {"train_loss": [], "val_acc": []}

    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0.0
        for batch in train_loader:
            inputs = U.batch_inputs(batch, device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad(set_to_none=True)
            loss = model(**inputs, labels=labels).loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.grad_clip)
            optimizer.step()
            scheduler.step()
            epoch_loss += loss.item()
        epoch_loss /= len(train_loader)

        val_acc = U.evaluate_classifier(model, val_loader, device)
        if val_acc > best_val:
            best_val = val_acc
            torch.save({"bert": model.state_dict()}, out_ckpt)

        history["train_loss"].append(epoch_loss)
        history["val_acc"].append(val_acc)
        marker = " *" if val_acc == best_val else ""
        print(f"  epoch {epoch + 1}/{args.epochs}  loss={epoch_loss:.4f}  "
              f"val_acc={val_acc:.4f}  lr={optimizer.param_groups[0]['lr']:.2e}{marker}")

    summary = {"task": args.task, "num_labels": num_labels,
               "best_val_acc": best_val, "history": history,
               "checkpoint": out_ckpt, "config": vars(args)}
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)

    print("=" * 64)
    print(f"Best teacher val_acc : {best_val:.4f}")
    print(f"Checkpoint           : {out_ckpt}")


if __name__ == "__main__":
    main()
