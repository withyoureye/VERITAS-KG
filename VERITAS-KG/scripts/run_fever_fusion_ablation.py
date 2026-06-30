from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Set

LABELS = ["SUPPORTS", "REFUTES", "NOT ENOUGH INFO"]


def load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


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


def stable_score(value: str, seed: int) -> float:
    digest = hashlib.sha256(f"{seed}:{value}".encode("utf-8")).hexdigest()
    return int(digest[:16], 16) / float(16**16 - 1)


def load_rankings(path: Path) -> Dict[str, Dict[str, Any]]:
    return {str(row["rank_group"]): row for row in load_jsonl(path)}


def load_qwen(path: Path) -> Dict[str, Dict[str, Any]]:
    payload = load_json(path)
    return {str(row["rank_group"]): row for row in payload.get("details", [])}


def split_ids(group_ids: Sequence[str], seed: int, calibration_ratio: float) -> tuple[Set[str], Set[str]]:
    calibration = {gid for gid in group_ids if stable_score(gid, seed) < calibration_ratio}
    evaluation = set(group_ids) - calibration
    return calibration, evaluation


def candidate_score(candidate: Dict[str, Any], mode: str, rng: random.Random) -> float:
    confidence = float(candidate.get("confidence", 0.0))
    evidence = float(candidate.get("evidence_weight", 1.0))
    if mode == "full":
        return float(candidate.get("score", confidence * evidence))
    if mode == "no_evidence_reliability":
        return confidence
    if mode == "random_evidence_weight":
        return confidence * rng.random()
    raise ValueError(f"Unknown KG mode: {mode}")


def evaluate_variant(
    rankings: Dict[str, Dict[str, Any]],
    qwen: Dict[str, Dict[str, Any]],
    group_ids: Set[str],
    alpha: float,
    variant: str,
    seed: int,
) -> Dict[str, Any]:
    rng = random.Random(seed)
    rows: List[Dict[str, Any]] = []
    for group_id in sorted(group_ids):
        ranking = rankings.get(group_id)
        qwen_row = qwen.get(group_id)
        if ranking is None or qwen_row is None:
            continue
        qwen_label = qwen_row.get("predicted_label")
        best = None
        for candidate in ranking.get("candidates", []):
            label = candidate.get("label")
            if variant == "no_kg_score":
                fused = 1.0 if label == qwen_label else 0.0
            elif variant == "no_llm_score":
                fused = candidate_score(candidate, "full", rng)
            elif variant == "no_evidence_reliability":
                kg_score = candidate_score(candidate, "no_evidence_reliability", rng)
                fused = alpha * kg_score + (1.0 - alpha) * (1.0 if label == qwen_label else 0.0)
            elif variant == "random_evidence_weight":
                kg_score = candidate_score(candidate, "random_evidence_weight", rng)
                fused = alpha * kg_score + (1.0 - alpha) * (1.0 if label == qwen_label else 0.0)
            elif variant == "full":
                kg_score = candidate_score(candidate, "full", rng)
                fused = alpha * kg_score + (1.0 - alpha) * (1.0 if label == qwen_label else 0.0)
            else:
                raise ValueError(f"Unknown variant: {variant}")
            item = dict(candidate)
            item["fused_score"] = fused
            if best is None or fused > best["fused_score"]:
                best = item
        if best is None:
            continue
        rows.append(
            {
                "rank_group": group_id,
                "gold_label": ranking.get("gold_label"),
                "predicted_label": best.get("label"),
                "qwen_label": qwen_label,
                "correct": best.get("label") == ranking.get("gold_label"),
            }
        )
    accuracy = sum(row["correct"] for row in rows) / len(rows) if rows else 0.0
    return {
        "variant": variant,
        "evaluated": len(rows),
        "accuracy": accuracy,
        "macro_f1": macro_f1(rows),
        "details": rows,
    }


def external_result(path: Path, name: str) -> Dict[str, Any]:
    payload = load_json(path)
    return {
        "variant": name,
        "evaluated": payload.get("evaluated", 0),
        "accuracy": payload.get("accuracy", 0.0),
        "macro_f1": payload.get("macro_f1", 0.0),
    }


def topk_rows(analysis_path: Path) -> List[Dict[str, Any]]:
    payload = load_json(analysis_path)
    rows = []
    by_k: Dict[int, Dict[str, Any]] = {}
    for row in payload.get("calibration", []):
        k = int(row.get("top_k", -1))
        current = by_k.get(k)
        if current is None or (row.get("accuracy", 0.0), row.get("macro_f1", 0.0)) > (
            current.get("accuracy", 0.0),
            current.get("macro_f1", 0.0),
        ):
            by_k[k] = row
    for k in sorted(by_k):
        row = by_k[k]
        rows.append(
            {
                "top_k": k,
                "accuracy": row.get("accuracy", 0.0),
                "macro_f1": row.get("macro_f1", 0.0),
                "threshold": row.get("threshold"),
            }
        )
    return rows


def write_summary(output: Path, payload: Dict[str, Any]) -> None:
    lines = [
        "# FEVER Fine-Grained Fusion Ablation",
        "",
        f"- Split seed: `{payload['split_seed']}`",
        f"- Calibration ratio: `{payload['calibration_ratio']}`",
        f"- Evaluation groups: `{payload['evaluation_groups']}`",
        f"- Frozen alpha: `{payload['alpha']}`",
        "",
        "## Fusion Ablations on Held-Out Evaluation",
        "",
        "| Variant | Evaluated | Accuracy | Macro-F1 |",
        "|---|---:|---:|---:|",
    ]
    for row in payload["fusion_ablation"]:
        lines.append(f"| {row['variant']} | {row['evaluated']} | {row['accuracy']:.4f} | {row['macro_f1']:.4f} |")
    lines.extend(
        [
            "",
            "## Gold vs Retrieved Evidence Readers",
            "",
            "| Variant | Evaluated | Accuracy | Macro-F1 |",
            "|---|---:|---:|---:|",
        ]
    )
    for row in payload["evidence_source"]:
        lines.append(f"| {row['variant']} | {row['evaluated']} | {row['accuracy']:.4f} | {row['macro_f1']:.4f} |")
    lines.extend(
        [
            "",
            "## Top-k Retrieval Sensitivity",
            "",
            "| top-k | best threshold | Accuracy | Macro-F1 |",
            "|---:|---:|---:|---:|",
        ]
    )
    for row in payload["topk_sensitivity"]:
        lines.append(f"| {row['top_k']} | {row['threshold']} | {row['accuracy']:.4f} | {row['macro_f1']:.4f} |")
    output.mkdir(parents=True, exist_ok=True)
    (output / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run FEVER fine-grained fusion ablations.")
    parser.add_argument("--kg-rankings", required=True, type=Path)
    parser.add_argument("--qwen", required=True, type=Path)
    parser.add_argument("--gold-reader", required=True, type=Path)
    parser.add_argument("--retrieved-reader", required=True, type=Path)
    parser.add_argument("--retrieval-analysis", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--alpha", type=float, default=0.9)
    parser.add_argument("--calibration-ratio", type=float, default=0.2)
    parser.add_argument("--split-seed", type=int, default=13)
    parser.add_argument("--random-seed", type=int, default=0)
    args = parser.parse_args()

    rankings = load_rankings(args.kg_rankings)
    qwen = load_qwen(args.qwen)
    group_ids = sorted(set(rankings) & set(qwen))
    _, evaluation_ids = split_ids(group_ids, args.split_seed, args.calibration_ratio)
    variants = [
        "no_kg_score",
        "no_llm_score",
        "no_evidence_reliability",
        "random_evidence_weight",
        "full",
    ]
    fusion_rows = [
        evaluate_variant(rankings, qwen, evaluation_ids, args.alpha, variant, args.random_seed)
        for variant in variants
    ]
    payload = {
        "split_seed": args.split_seed,
        "calibration_ratio": args.calibration_ratio,
        "evaluation_groups": len(evaluation_ids),
        "alpha": args.alpha,
        "fusion_ablation": [
            {key: value for key, value in row.items() if key != "details"}
            for row in fusion_rows
        ],
        "evidence_source": [
            external_result(args.gold_reader, "gold_evidence_reader"),
            external_result(args.retrieved_reader, "retrieved_bm25_reader"),
        ],
        "topk_sensitivity": topk_rows(args.retrieval_analysis),
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "ablation.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_summary(args.output, payload)
    print((args.output / "summary.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
