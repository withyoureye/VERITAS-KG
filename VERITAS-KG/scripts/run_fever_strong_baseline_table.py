from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence


def load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def macro_f1(rows: Sequence[Dict[str, Any]]) -> float:
    labels = ["SUPPORTS", "REFUTES", "NOT ENOUGH INFO"]
    scores: List[float] = []
    for label in labels:
        tp = sum(1 for row in rows if row["gold_label"] == label and row["predicted_label"] == label)
        fp = sum(1 for row in rows if row["gold_label"] != label and row["predicted_label"] == label)
        fn = sum(1 for row in rows if row["gold_label"] == label and row["predicted_label"] != label)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        scores.append(2 * precision * recall / (precision + recall) if precision + recall else 0.0)
    return sum(scores) / len(scores)


def kg_only_from_rankings(path: Path) -> Dict[str, Any]:
    rows: List[Dict[str, Any]] = []
    for row in load_jsonl(path):
        rows.append(
            {
                "rank_group": row["rank_group"],
                "gold_label": row["gold_label"],
                "predicted_label": row["top_label"],
                "correct": row["gold_label"] == row["top_label"],
            }
        )
    accuracy = sum(row["correct"] for row in rows) / len(rows) if rows else 0.0
    return {
        "baseline": "kg_scorer_only",
        "evaluated": len(rows),
        "accuracy": accuracy,
        "macro_f1": macro_f1(rows),
        "details": rows,
    }


def bm25_nli_from_reader(path: Path, name: str = "bm25_lightweight_nli_reader") -> Dict[str, Any]:
    payload = load_json(path)
    return {
        "baseline": name,
        "evaluated": payload.get("evaluated", 0),
        "accuracy": payload.get("accuracy", 0.0),
        "macro_f1": payload.get("macro_f1", 0.0),
        "note": "BM25 retrieval + lexical NLI-style reader; neural NLI weights not available locally.",
    }


def qwen_reader(path: Path, name: str) -> Dict[str, Any]:
    payload = load_json(path)
    return {
        "baseline": name,
        "evaluated": payload.get("evaluated", 0),
        "accuracy": payload.get("accuracy", 0.0),
        "macro_f1": payload.get("macro_f1", 0.0),
        "top_k": payload.get("top_k"),
    }


def fusion(path: Path) -> Dict[str, Any]:
    payload = load_json(path)
    best = payload.get("best") or payload.get("evaluation", {})
    return {
        "baseline": "kg_qwen_fusion_calibrated" if "evaluation" in payload else "kg_qwen_fusion",
        "evaluated": best.get("evaluated", 0),
        "accuracy": best.get("accuracy", 0.0),
        "macro_f1": best.get("macro_f1", 0.0),
        "alpha": payload.get("selected_alpha", best.get("alpha")),
        "note": "alpha selected on calibration split and frozen for held-out evaluation" if "evaluation" in payload else "",
    }


def maybe_dense_qwen(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {
            "baseline": "dense_retrieval_qwen_reader",
            "evaluated": 0,
            "accuracy": None,
            "macro_f1": None,
            "note": "blocked: no complete local DPR/Contriever retriever weights; network unavailable",
        }
    return qwen_reader(path, "dense_retrieval_qwen_reader")


def write_outputs(output: Path, rows: Sequence[Dict[str, Any]]) -> None:
    output.mkdir(parents=True, exist_ok=True)
    (output / "baselines.json").write_text(json.dumps({"rows": rows}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# FEVER Strong Baseline Table",
        "",
        "| Method | Evaluated | Accuracy | Macro-F1 | Note |",
        "|---|---:|---:|---:|---|",
    ]
    for row in rows:
        acc = row.get("accuracy")
        f1 = row.get("macro_f1")
        lines.append(
            "| {method} | {evaluated} | {acc} | {f1} | {note} |".format(
                method=row.get("baseline", ""),
                evaluated=row.get("evaluated", ""),
                acc="" if acc is None else f"{float(acc):.4f}",
                f1="" if f1 is None else f"{float(f1):.4f}",
                note=row.get("note", f"alpha={row.get('alpha')}" if row.get("alpha") is not None else ""),
            )
        )
    (output / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print((output / "summary.md").read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Write a unified FEVER strong baseline table.")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--kg-rankings", required=True, type=Path)
    parser.add_argument("--bm25-reader", required=True, type=Path)
    parser.add_argument("--bm25-qwen", required=True, type=Path)
    parser.add_argument("--dense-qwen", type=Path)
    parser.add_argument("--qwen-only", required=True, type=Path)
    parser.add_argument("--fusion", required=True, type=Path)
    args = parser.parse_args()

    rows = [
        bm25_nli_from_reader(args.bm25_reader),
        qwen_reader(args.bm25_qwen, "bm25_qwen_reader"),
        maybe_dense_qwen(args.dense_qwen) if args.dense_qwen else {"baseline": "dpr_contriever_qwen_reader", "evaluated": 0, "accuracy": None, "macro_f1": None, "note": "blocked: no complete local DPR/Contriever retriever weights; network unavailable"},
        kg_only_from_rankings(args.kg_rankings),
        qwen_reader(args.qwen_only, "qwen_reader_only"),
        fusion(args.fusion),
    ]
    write_outputs(args.output, rows)


if __name__ == "__main__":
    main()
