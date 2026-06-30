from __future__ import annotations

import argparse
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple


TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")
LABELS = ["SUPPORTS", "REFUTES", "NOT ENOUGH INFO"]
NEGATION_TERMS = {"not", "never", "no", "false", "disassociated", "refused", "without"}


def tokenize(text: str) -> List[str]:
    return [token.lower() for token in TOKEN_RE.findall(text)]


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def group_records(records: Sequence[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[str(record.get("rank_group", ""))].append(record)
    return {key: value for key, value in grouped.items() if key}


def lexical_overlap(claim: str, evidence: str) -> float:
    claim_tokens = set(tokenize(claim))
    evidence_tokens = set(tokenize(evidence))
    if not claim_tokens:
        return 0.0
    return len(claim_tokens & evidence_tokens) / len(claim_tokens)


def features(record: Dict[str, Any]) -> List[float]:
    claim = str(record.get("claim_text", record.get("subject", {}).get("label", "")))
    evidence = str(record.get("evidence", {}).get("snippet", ""))
    source_type = str(record.get("evidence", {}).get("source_type", ""))
    label = str(record.get("candidate_label", ""))
    claim_tokens = tokenize(claim)
    evidence_tokens = tokenize(evidence)
    has_evidence = float(source_type == "wikipedia_sentence")
    has_pointer = float(source_type == "wikipedia_sentence_pointer")
    has_claim_only = float(source_type == "claim")
    has_negation = float(any(token in NEGATION_TERMS for token in claim_tokens))
    label_support = float(label == "SUPPORTS")
    label_refutes = float(label == "REFUTES")
    label_nei = float(label == "NOT ENOUGH INFO")
    overlap = lexical_overlap(claim, evidence)
    length_ratio = min(len(evidence_tokens) / max(len(claim_tokens), 1), 10.0) / 10.0
    return [
        1.0,
        has_evidence,
        has_pointer,
        has_claim_only,
        overlap,
        length_ratio,
        has_negation,
        label_support,
        label_refutes,
        label_nei,
        has_evidence * label_support,
        has_evidence * label_refutes,
        has_claim_only * label_nei,
        has_negation * label_refutes,
    ]


def sigmoid(value: float) -> float:
    if value >= 0:
        z = math.exp(-value)
        return 1.0 / (1.0 + z)
    z = math.exp(value)
    return z / (1.0 + z)


def dot(left: Sequence[float], right: Sequence[float]) -> float:
    return sum(a * b for a, b in zip(left, right))


def train_logreg(rows: Sequence[Dict[str, Any]], epochs: int, lr: float, l2: float) -> List[float]:
    weights = [0.0] * len(features(rows[0]))
    for _ in range(epochs):
        for row in rows:
            x = features(row)
            y = 1.0 if row.get("is_gold") else 0.0
            pred = sigmoid(dot(weights, x))
            error = pred - y
            for idx, value in enumerate(x):
                weights[idx] -= lr * (error * value + l2 * weights[idx])
    return weights


def score_record(record: Dict[str, Any], weights: Sequence[float]) -> float:
    return sigmoid(dot(weights, features(record)))


def evaluate_groups(records: Sequence[Dict[str, Any]], weights: Sequence[float]) -> Dict[str, float]:
    evaluated = 0
    correct = 0
    for _, candidates in group_records(records).items():
        gold = next((row for row in candidates if row.get("is_gold")), None)
        if gold is None:
            continue
        top = max(candidates, key=lambda row: float(row.get("confidence", 0.0)) * score_record(row, weights))
        evaluated += 1
        correct += int(top.get("is_gold") is True)
    return {"evaluated": evaluated, "top1_accuracy": correct / evaluated if evaluated else 0.0}


def apply_scores(records: Sequence[Dict[str, Any]], weights: Sequence[float]) -> List[Dict[str, Any]]:
    scored: List[Dict[str, Any]] = []
    for record in records:
        copied = json.loads(json.dumps(record))
        evidence = dict(copied.get("evidence", {}))
        evidence["weight"] = round(score_record(copied, weights), 6)
        copied["evidence"] = evidence
        copied["evidence_scorer"] = "learned_fever_logreg"
        scored.append(copied)
    return scored


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a lightweight learned FEVER evidence scorer.")
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--train-groups", type=int, default=6000)
    parser.add_argument("--valid-groups", type=int, default=2000)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--l2", type=float, default=1e-4)
    args = parser.parse_args()

    records = load_jsonl(args.input)
    grouped = group_records(records)
    group_ids = sorted(grouped)
    train_ids = set(group_ids[: args.train_groups])
    valid_ids = set(group_ids[args.train_groups : args.train_groups + args.valid_groups])
    train_rows = [row for group_id in train_ids for row in grouped[group_id]]
    valid_rows = [row for group_id in valid_ids for row in grouped[group_id]]
    weights = train_logreg(train_rows, args.epochs, args.lr, args.l2)
    train_metrics = evaluate_groups(train_rows, weights)
    valid_metrics = evaluate_groups(valid_rows, weights)
    all_scored = apply_scores(records, weights)

    args.output.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output / "assertions_scored.jsonl", all_scored)
    payload = {
        "scorer": "fever_logistic_regression",
        "weights": weights,
        "feature_names": [
            "bias",
            "has_evidence",
            "has_pointer",
            "has_claim_only",
            "overlap",
            "length_ratio",
            "has_negation",
            "label_support",
            "label_refutes",
            "label_nei",
            "has_evidence_x_support",
            "has_evidence_x_refutes",
            "claim_only_x_nei",
            "negation_x_refutes",
        ],
        "train": train_metrics,
        "valid": valid_metrics,
        "train_groups": len(train_ids),
        "valid_groups": len(valid_ids),
        "output_assertions": str(args.output / "assertions_scored.jsonl"),
    }
    (args.output / "scorer.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (args.output / "summary.md").write_text(
        "\n".join(
            [
                "# FEVER Learned Evidence Scorer",
                "",
                f"- Train groups: {len(train_ids)}",
                f"- Valid groups: {len(valid_ids)}",
                f"- Train top1: {train_metrics['top1_accuracy']:.4f}",
                f"- Valid top1: {valid_metrics['top1_accuracy']:.4f}",
                f"- Scored assertions: `{args.output / 'assertions_scored.jsonl'}`",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    print((args.output / "summary.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
