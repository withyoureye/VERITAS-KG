from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Sequence


LABELS = ["SUPPORTS", "REFUTES", "NOT ENOUGH INFO"]


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


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


def load_qwen(path: Path) -> Dict[str, Dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {str(row["rank_group"]): row for row in payload.get("details", [])}


def load_kg_rankings(path: Path) -> Dict[str, Dict[str, Any]]:
    rankings: Dict[str, Dict[str, Any]] = {}
    for row in load_jsonl(path):
        rankings[str(row["rank_group"])] = row
    return rankings


def evaluate(
    kg_rankings: Dict[str, Dict[str, Any]],
    qwen_rows: Dict[str, Dict[str, Any]],
    alpha: float,
    qwen_boost: float,
    details_limit: int,
) -> Dict[str, Any]:
    rows: List[Dict[str, Any]] = []
    for group_id, qwen in sorted(qwen_rows.items()):
        kg = kg_rankings.get(group_id)
        if kg is None:
            continue
        candidates = kg.get("candidates", [])
        qwen_label = qwen.get("predicted_label")
        best = None
        for candidate in candidates:
            kg_score = float(candidate.get("score", 0.0))
            llm_score = qwen_boost if candidate.get("label") == qwen_label else 0.0
            fused = alpha * kg_score + (1.0 - alpha) * llm_score
            item = dict(candidate)
            item["fused_score"] = fused
            if best is None or fused > best["fused_score"]:
                best = item
        if best is None:
            continue
        rows.append(
            {
                "rank_group": group_id,
                "gold_label": kg.get("gold_label"),
                "kg_label": kg.get("top_label"),
                "qwen_label": qwen_label,
                "predicted_label": best.get("label"),
                "correct": best.get("label") == kg.get("gold_label"),
            }
        )
    accuracy = sum(1 for row in rows if row["correct"]) / len(rows) if rows else 0.0
    return {
        "alpha": alpha,
        "qwen_boost": qwen_boost,
        "evaluated": len(rows),
        "accuracy": accuracy,
        "macro_f1": macro_f1(rows),
        "details": rows if details_limit < 0 else rows[:details_limit],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Fuse FEVER KG ranking with Qwen evidence-reader labels.")
    parser.add_argument("--kg-rankings", required=True, type=Path)
    parser.add_argument("--qwen", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--alpha", action="append", type=float, default=[])
    parser.add_argument("--qwen-boost", type=float, default=1.0)
    parser.add_argument(
        "--details-limit",
        type=int,
        default=-1,
        help="Number of per-claim details to keep; use -1 to keep all.",
    )
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    kg_rankings = load_kg_rankings(args.kg_rankings)
    qwen_rows = load_qwen(args.qwen)
    alphas = args.alpha or [0.0, 0.25, 0.5, 0.75, 1.0]
    results = [
        evaluate(kg_rankings, qwen_rows, alpha, args.qwen_boost, args.details_limit)
        for alpha in alphas
    ]
    best = max(results, key=lambda row: (row["accuracy"], row["macro_f1"])) if results else {}
    payload = {"baseline": "fever_kg_qwen_fusion", "results": results, "best": best}
    (args.output / "fusion.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# FEVER KG/Qwen Fusion",
        "",
        "| alpha | evaluated | accuracy | macro_f1 |",
        "|---:|---:|---:|---:|",
    ]
    for row in results:
        lines.append(f"| {row['alpha']:.2f} | {row['evaluated']} | {row['accuracy']:.4f} | {row['macro_f1']:.4f} |")
    lines.extend(
        [
            "",
            f"Best alpha: `{best.get('alpha', '')}`",
            f"Best accuracy: `{float(best.get('accuracy', 0.0)):.4f}`",
            f"Best macro-F1: `{float(best.get('macro_f1', 0.0)):.4f}`",
        ]
    )
    (args.output / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print((args.output / "summary.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
