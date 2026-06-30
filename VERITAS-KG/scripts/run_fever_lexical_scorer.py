from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Sequence


TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")
LABELS = ["SUPPORTS", "REFUTES", "NOT ENOUGH INFO"]


def tokenize(text: str) -> List[str]:
    return [token.lower() for token in TOKEN_RE.findall(text)]


def load_groups(path: Path, groups_limit: int) -> List[List[Dict[str, Any]]]:
    groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            group = record.get("rank_group")
            if not group:
                continue
            group_id = str(group)
            if group_id not in groups and len(groups) >= groups_limit:
                continue
            groups[group_id].append(record)
    return [groups[key] for key in sorted(groups)]


def macro_f1(rows: Sequence[Dict[str, Any]]) -> float:
    scores: List[float] = []
    for label in LABELS:
        tp = sum(1 for row in rows if row["gold_label"] == label and row["predicted_label"] == label)
        fp = sum(1 for row in rows if row["gold_label"] != label and row["predicted_label"] == label)
        fn = sum(1 for row in rows if row["gold_label"] == label and row["predicted_label"] != label)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        scores.append(2 * precision * recall / (precision + recall) if precision + recall else 0.0)
    return sum(scores) / len(scores)


def classify(claim: str, evidence: str) -> str:
    claim_tokens = set(tokenize(claim))
    evidence_tokens = set(tokenize(evidence))
    overlap = len(claim_tokens & evidence_tokens)
    if "not enough info" in claim.lower():
        return "NOT ENOUGH INFO"
    if overlap >= 3:
        return "SUPPORTS"
    if any(token in claim_tokens for token in {"not", "never", "no", "false", "fake"}):
        return "REFUTES"
    if overlap <= 1:
        return "NOT ENOUGH INFO"
    return "SUPPORTS" if overlap > 2 else "REFUTES"


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a lexical FEVER scorer baseline.")
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--groups-limit", type=int, default=1000)
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    rows: List[Dict[str, Any]] = []
    for candidates in load_groups(args.input, args.groups_limit):
        gold = next((item for item in candidates if item.get("is_gold")), None)
        if gold is None:
            continue
        claim = str(gold.get("claim_text", gold["subject"]["label"]))
        evidence = str(gold.get("evidence", {}).get("snippet", ""))
        predicted = classify(claim, evidence)
        rows.append(
            {
                "rank_group": gold["rank_group"],
                "claim": claim,
                "gold_label": gold["gold_label"],
                "predicted_label": predicted,
                "correct": predicted == gold["gold_label"],
            }
        )

    accuracy = sum(1 for row in rows if row["correct"]) / len(rows) if rows else 0.0
    payload = {
        "baseline": "fever_lexical_overlap",
        "evaluated": len(rows),
        "accuracy": accuracy,
        "macro_f1": macro_f1(rows),
        "details": rows[:100],
    }
    (args.output / "baseline.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    with (args.output / "details.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["rank_group", "claim", "gold_label", "predicted_label", "correct"])
        writer.writeheader()
        for row in rows[:200]:
            writer.writerow(row)
    (args.output / "summary.md").write_text(
        "\n".join(
            [
                "# FEVER Lexical Baseline",
                "",
                f"- Evaluated: {len(rows)}",
                f"- Accuracy: {accuracy:.4f}",
                f"- Macro-F1: {payload['macro_f1']:.4f}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    print((args.output / "summary.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
