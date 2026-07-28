"""
Golden-reference runner. Loads one of the four substituted-FFN TinyBERT4 models
(Bernstein or GeLU, width 312 or 600) and classifies a sentence. Useful as a
quick "the weights load and the network runs" check, and to bit-check a hardware
implementation of the full network.

Requires torch + transformers; the stock BERT parts load from
huawei-noah/TinyBERT_General_4L_312D (downloaded once).

  python Transformer/load_and_run.py bern_h312 "this movie was a delight"
  python Transformer/load_and_run.py gelu_h600 "a boring waste of time"

Variants: bern_h312, bern_h600, gelu_h312, gelu_h600  (default: bern_h312).
"""

import json
import os
import sys

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from bernbert import BernBertForSequenceClassification  # noqa: E402

VARIANTS = ["bern_h312", "bern_h600", "gelu_h312", "gelu_h600"]
MODEL_NAME = "huawei-noah/TinyBERT_General_4L_312D"


def main() -> None:
    which = sys.argv[1] if len(sys.argv) > 1 else "bern_h312"
    if which not in VARIANTS:
        raise SystemExit(f"unknown variant {which!r}; choose from {VARIANTS}")
    sentence = sys.argv[2] if len(sys.argv) > 2 else "this movie was a delight"

    base = os.path.join(_HERE, "models", f"release_{which}")
    with open(base + ".meta.json") as f:
        meta = json.load(f)
    print(f"Model: {which} | act:", meta["act"], "| replaced:", meta["replaced_layers"],
          "| hidden:", meta["hidden"], "| degree:", meta["degree"])
    print("Verified SST-2 val acc:", meta["verified_val_acc"])

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = BernBertForSequenceClassification(
        meta["replaced_layers"], hidden=meta["hidden"],
        degree=meta["degree"] or 0, act=meta["act"])
    model.load_state_dict(torch.load(base + ".pt", map_location="cpu")["state_dict"])
    model.eval().to(device)

    from transformers.models.bert.tokenization_bert import BertTokenizer
    tok = BertTokenizer.from_pretrained(MODEL_NAME)
    enc = tok([sentence], max_length=64, padding="max_length",
              truncation=True, return_tensors="pt").to(device)
    with torch.no_grad():
        logits = model(enc["input_ids"], enc["attention_mask"])
    prob = torch.softmax(logits, -1)[0]
    label = "positive" if prob[1] > prob[0] else "negative"
    print(f"\nsentence : {sentence!r}")
    print(f"logits   : {logits[0].tolist()}")
    print(f"prob     : neg={prob[0]:.4f}  pos={prob[1]:.4f}  -> {label}")


if __name__ == "__main__":
    main()
