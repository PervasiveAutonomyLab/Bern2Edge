#!/usr/bin/env python3
"""Evaluate ready Adult rule artifacts without rerunning extraction.

Examples (run from the repository root):

    # One artifact
    python Adult/evaluate_rule_artifacts.py path/to/rules.json

    # Every JSON in one or more directories
    python Adult/evaluate_rule_artifacts.py path/to/artifacts/ other/rules.json \
        --output metrics.csv

Decision-tree artifacts use a sidecar beside the JSON; both research names
``<json_stem>_tree.npz`` and current extractor names ``fallback_tree_float.npz``
are recognized. Self-contained ``lr_x_space`` JSON artifacts are also supported.
The CLI writes one CSV row per artifact and returns the same rows from
``evaluate_paths()`` when imported. Metrics are measured against Adult ground
truth on the fixed seed-6/fold-0 test split:

``n_rules``, ``test_covered_pct``, ``test_covered_rule_acc``,
``test_rule_acc`` (rules plus fallback), ``n_conflicts``, and
``avg_conditions``. Use ``--conflict-strategy max_purity`` for artifacts whose
highest-purity matching rule wins; the default evaluates rules by descending
training coverage.

This evaluator reads existing artifacts; it does not extract or train rules.
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(REPO))

from bern2edge.data import adult_ordinal_dataloaders_kfold  # noqa: E402
from bern2edge.models import FCModel  # noqa: E402


CSV_FIELDS = (
    "arch",
    "same_cov_alpha",
    "conflict_alpha",
    "n_rules",
    "test_covered_pct",
    "test_covered_rule_acc",
    "test_rule_acc",
    "test_fidelity",
    "test_uncovered_pct",
    "fallback_acc_uncovered",
    "fallback_fidelity_uncovered",
    "reevaluated_test_covered_pct",
    "reevaluated_test_rule_acc",
    "n_conflicts",
    "avg_conditions",
    "rule_json",
    "fallback_artifact",
    "cart_npz",
)


def load_adult_test_split():
    """Return ``(X_test, y_test)`` for the fixed Adult artifact split."""
    teacher = HERE / "adult_teacher_ordinal.pt"
    folds, _, _ = adult_ordinal_dataloaders_kfold(str(teacher), k=5, seed=6)
    test_loader = folds[0][2]
    X_test, y_test = (tensor.numpy() for tensor in test_loader.dataset.tensors)
    return X_test.astype(np.float32), y_test.astype(np.int64)


def _rule_mask(rule, X):
    mask = np.ones(X.shape[0], dtype=bool)
    for condition in rule["conditions"]:
        score = X @ np.asarray(condition["weight_vector"], dtype=np.float64)
        if condition["band_lo"] is not None:
            mask &= score >= float(condition["band_lo"])
        if condition["band_hi"] is not None:
            mask &= score < float(condition["band_hi"])
    return mask


def _cart_predict(npz_path, X):
    with np.load(npz_path) as tree:
        left = tree["children_left"]
        right = tree["children_right"]
        feature = tree["feature"]
        threshold = tree["threshold"]
        value = tree["value"]

    pred = np.empty(X.shape[0], dtype=np.int64)
    for row_index, row in enumerate(X):
        node = 0
        while left[node] != right[node]:
            node = left[node] if row[feature[node]] <= threshold[node] else right[node]
        pred[row_index] = int(np.argmax(value[node]))
    return pred


def _find_cart(json_path):
    candidates = (
        json_path.with_name(json_path.stem + "_tree.npz"),
        json_path.with_name("fallback_tree_float.npz"),
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate
    names = ", ".join(path.name for path in candidates)
    raise FileNotFoundError(f"{json_path.name}: missing CART sidecar ({names})")


def _network_predict(checkpoint_path, X):
    raw = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = FCModel(
        raw["arch"],
        raw["degree"],
        act=raw.get("activation", raw.get("act", "bern")),
        last_bern=raw.get("last_bern", False),
    )
    model.load_state_dict(raw["state_dict"])
    model.eval()
    with torch.no_grad():
        pred = model(torch.from_numpy(X.astype(np.float32))).argmax(1).numpy()
    return pred.astype(np.int64)


def _find_network(json_path, fallback_type):
    names = (
        ("fallback_network.pth",)
        if fallback_type == "neural_network"
        else ("fallback_net.pt", "fallback_net_float.pt")
    )
    for name in names:
        path = json_path.with_name(name)
        if path.exists():
            return path
    raise FileNotFoundError(f"{json_path.name}: missing {' or '.join(names)}")


def _fallback_predict(artifact, json_path, X):
    fallback = artifact.get("fallback", {})
    fallback_type = fallback.get("type")
    if fallback_type == "decision_tree":
        path = _find_cart(json_path)
        return _cart_predict(path, X), path
    if fallback_type == "lr_x_space":
        weights = np.asarray(fallback["w_eff"], dtype=np.float64)
        bias = float(fallback["b_eff"])
        return ((X @ weights + bias) > 0).astype(np.int64), None
    if fallback_type in ("neural_network", "small_bern_nn"):
        path = _find_network(json_path, fallback_type)
        return _network_predict(path, X), path
    raise ValueError(f"{json_path.name}: unsupported fallback {fallback_type!r}")


def evaluate_rule_artifact(
    json_path,
    X_test,
    y_test,
    stored_tolerance=0.02,
    conflict_strategy="coverage",
    validate_stored=True,
    y_model=None,
):
    """Evaluate one ready artifact and return a flat metrics dictionary.

    JSON weights are serialized to six decimals, which can move a boundary
    sample. Generation-time metrics embedded in the JSON are retained only
    after the live result agrees within ``stored_tolerance``. The live coverage
    and total accuracy are also returned in ``reevaluated_*`` audit columns.

    ``conflict_strategy`` is ``coverage`` (descending training coverage) or
    ``max_purity`` (highest-purity matching rule).
    """
    json_path = Path(json_path).resolve()
    with json_path.open() as stream:
        artifact = json.load(stream)

    rules = artifact["rules"]
    masks = [_rule_mask(rule, X_test) for rule in rules]
    covered_correct = np.zeros(len(y_test), dtype=bool)
    covered_wrong = np.zeros(len(y_test), dtype=bool)
    for rule, mask in zip(rules, masks):
        label = int(rule["label"])
        covered_correct |= mask & (y_test == label)
        covered_wrong |= mask & (y_test != label)
    covered = covered_correct | covered_wrong

    pred = np.full(len(y_test), -1, dtype=np.int64)
    if conflict_strategy == "coverage":
        remaining = np.ones(len(y_test), dtype=bool)
        order = sorted(range(len(rules)), key=lambda i: rules[i].get("coverage", 0),
                       reverse=True)
        for rule_index in order:
            fire = masks[rule_index] & remaining
            pred[fire] = int(rules[rule_index]["label"])
            remaining[fire] = False
    elif conflict_strategy == "max_purity":
        best_purity = np.full(len(y_test), -1.0)
        for rule, mask in zip(rules, masks):
            upgrade = mask & (float(rule["purity"]) > best_purity)
            pred[upgrade] = int(rule["label"])
            best_purity[upgrade] = float(rule["purity"])
        remaining = pred == -1
    else:
        raise ValueError(f"unknown conflict strategy: {conflict_strategy}")
    fallback, fallback_path = _fallback_predict(artifact, json_path, X_test)
    pred[remaining] = fallback[remaining]

    live = {
        "n_rules": len(rules),
        "test_covered_pct": round(100.0 * float(covered.mean()), 2),
        "test_covered_rule_acc": round(
            100.0 * float(covered_correct[covered].mean()) if covered.any() else 0.0, 2
        ),
        "test_rule_acc": round(100.0 * float((pred == y_test).mean()), 2),
        "test_fidelity": (
            round(100.0 * float((pred == y_model).mean()), 2)
            if y_model is not None else ""
        ),
        "test_uncovered_pct": round(100.0 * float(remaining.mean()), 2),
        "fallback_acc_uncovered": round(
            100.0 * float((fallback[remaining] == y_test[remaining]).mean())
            if remaining.any() else 0.0,
            2,
        ),
        "fallback_fidelity_uncovered": (
            round(
                100.0 * float((fallback[remaining] == y_model[remaining]).mean())
                if remaining.any() else 0.0,
                2,
            )
            if y_model is not None else ""
        ),
        "n_conflicts": int((covered_correct & covered_wrong).sum()),
        "avg_conditions": round(
            float(np.mean([len(rule["conditions"]) for rule in rules])) if rules else 0.0,
            2,
        ),
    }

    stored = artifact.get("metrics", {})
    if validate_stored:
        for key in ("n_rules", "test_covered_pct", "test_rule_acc",
                    "n_conflicts", "avg_conditions"):
            if key not in stored:
                continue
            difference = abs(float(live[key]) - float(stored[key]))
            if difference > stored_tolerance:
                raise RuntimeError(
                    f"{json_path.name}: live {key}={live[key]}, stored={stored[key]}"
                )

    metrics = dict(live)
    if validate_stored:
        for key in ("n_rules", "test_covered_pct", "test_rule_acc",
                    "n_conflicts", "avg_conditions"):
            if key in stored:
                metrics[key] = stored[key]
    metrics["reevaluated_test_covered_pct"] = live["test_covered_pct"]
    metrics["reevaluated_test_rule_acc"] = live["test_rule_acc"]

    config = artifact.get("config", {})
    return {
        "arch": "x".join(str(v) for v in config.get("arch", [])),
        "same_cov_alpha": config.get("same_cov_alpha", ""),
        "conflict_alpha": config.get("conflict_alpha", ""),
        **metrics,
        "rule_json": _display_path(json_path),
        "fallback_artifact": _display_path(fallback_path) if fallback_path else "embedded",
        "cart_npz": _display_path(fallback_path) if fallback_path else "",
    }


def _display_path(path):
    try:
        return str(path.relative_to(REPO))
    except ValueError:
        return str(path)


def resolve_json_paths(inputs, pattern="*.json", recursive=False):
    """Resolve files/directories into a sorted, duplicate-free JSON list."""
    paths = []
    for item in inputs:
        item = Path(item)
        if item.is_file():
            paths.append(item.resolve())
        elif item.is_dir():
            iterator = item.rglob(pattern) if recursive else item.glob(pattern)
            paths.extend(path.resolve() for path in iterator if path.is_file())
        else:
            raise FileNotFoundError(item)
    unique = sorted(set(paths))
    if not unique:
        raise FileNotFoundError("no rule JSON artifacts matched")
    return unique


def evaluate_paths(
    paths,
    output=None,
    X_test=None,
    y_test=None,
    conflict_strategy="coverage",
    validate_stored=True,
    y_model=None,
):
    """Evaluate paths and return rows; optionally write them to ``output``."""
    if X_test is None or y_test is None:
        X_test, y_test = load_adult_test_split()
    rows = [
        evaluate_rule_artifact(
            path,
            X_test,
            y_test,
            conflict_strategy=conflict_strategy,
            validate_stored=validate_stored,
            y_model=y_model,
        )
        for path in paths
    ]
    if output is not None:
        output = Path(output)
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=CSV_FIELDS)
            writer.writeheader()
            writer.writerows(rows)
    return rows


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("inputs", nargs="+", help="rule JSON files or directories")
    parser.add_argument("--glob", default="*.json", help="directory pattern")
    parser.add_argument("--recursive", action="store_true", help="search directories recursively")
    parser.add_argument("--output", default="rule_artifact_metrics.csv", help="output CSV")
    parser.add_argument(
        "--conflict-strategy",
        choices=("coverage", "max_purity"),
        default="coverage",
        help="winner when multiple rules match (default: coverage)",
    )
    parser.add_argument(
        "--reference-network",
        help="student checkpoint used for fidelity metrics",
    )
    parser.add_argument(
        "--no-stored-validation",
        action="store_true",
        help="return live metrics without validating generation-time metrics",
    )
    args = parser.parse_args()

    paths = resolve_json_paths(args.inputs, args.glob, args.recursive)
    y_model = None
    if args.reference_network:
        X_test, y_test = load_adult_test_split()
        y_model = _network_predict(Path(args.reference_network), X_test)
    else:
        X_test = y_test = None
    rows = evaluate_paths(
        paths,
        args.output,
        X_test=X_test,
        y_test=y_test,
        conflict_strategy=args.conflict_strategy,
        validate_stored=not args.no_stored_validation,
        y_model=y_model,
    )
    print(f"Evaluated {len(rows)} artifact(s) -> {args.output}")
    for row in rows:
        print(
            f"{Path(row['rule_json']).name}: rules={row['n_rules']} "
            f"coverage={row['test_covered_pct']:.2f}% "
            f"covered_acc={row['test_covered_rule_acc']:.2f}% "
            f"total_acc={row['test_rule_acc']:.2f}%"
        )


if __name__ == "__main__":
    main()
