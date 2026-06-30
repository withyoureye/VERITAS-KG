from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


def load_jsonl(path: Path, limit: Optional[int] = None) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for idx, line in enumerate(handle):
            if limit is not None and idx >= limit:
                break
            if line.strip():
                rows.append(json.loads(line))
    return rows


def normalize_label(label: str) -> str:
    return str(label).strip().upper()


def fever_llm_only(records: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    groups: Dict[str, List[Dict[str, Any]]] = {}
    for record in records:
        groups.setdefault(str(record["rank_group"]), []).append(record)
    evaluated = 0
    correct = 0
    details: List[Dict[str, Any]] = []
    for group_id, candidates in groups.items():
        gold = next((item for item in candidates if item.get("is_gold")), None)
        if gold is None:
            continue
        evaluated += 1
        claim = str(gold.get("claim_text", gold["subject"]["label"]))
        prompt = claim.lower()
        if " not " in prompt or " no " in prompt or "refuse" in prompt or "not enough" in prompt:
            predicted = "REFUTES"
        elif "who" in prompt or "what" in prompt or "where" in prompt:
            predicted = "NOT ENOUGH INFO"
        else:
            predicted = "SUPPORTS"
        top = next((item for item in candidates if item["candidate_label"] == predicted), candidates[0])
        is_correct = normalize_label(top["candidate_label"]) == normalize_label(gold["gold_label"])
        correct += int(is_correct)
        details.append(
            {
                "rank_group": group_id,
                "predicted_label": predicted,
                "gold_label": gold["gold_label"],
                "correct": is_correct,
                "source": "heuristic_llm_only",
            }
        )
    accuracy = correct / evaluated if evaluated else 0.0
    return {
        "baseline": "fever_llm_only",
        "evaluated": evaluated,
        "accuracy": accuracy,
        "sampled_details": details[:50],
    }


def icews_temporal_baseline(records: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    groups: Dict[str, List[Dict[str, Any]]] = {}
    for record in records:
        if record.get("task") != "temporal_assertion_ranking":
            continue
        groups.setdefault(str(record["rank_group"]), []).append(record)
    evaluated = 0
    correct = 0
    for _, candidates in groups.items():
        gold = next((item for item in candidates if item.get("is_gold")), None)
        if gold is None:
            continue
        evaluated += 1
        top = max(candidates, key=lambda item: (float(item["confidence"]), float(item["evidence"]["weight"])))
        correct += int(bool(top.get("is_gold")))
    return {
        "baseline": "icews_temporal_ranker",
        "evaluated": evaluated,
        "accuracy": correct / evaluated if evaluated else 0.0,
    }


def yago_rule_baseline(records: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    evaluated = 0
    invalid = 0
    for record in records:
        if record.get("task") != "ontology_validation":
            continue
        if not record.get("rank_group"):
            continue
        evaluated += 1
        subj_type = record["subject"]["type"]
        obj_type = record["object"]["type"]
        relation = record["relation"]
        if relation == "worksAt" and not (subj_type == "Person" and obj_type == "Organization"):
            invalid += 1
        elif relation == "isLocatedIn" and obj_type != "Location":
            invalid += 1
        elif relation == "wasBornIn" and not (subj_type == "Person" and obj_type == "Location"):
            invalid += 1
    return {
        "baseline": "yago_rule_filter",
        "evaluated": evaluated,
        "invalid_rate": invalid / evaluated if evaluated else 0.0,
    }


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run lightweight external baselines.")
    parser.add_argument("--dataset", choices=["fever", "icews14", "yago"], required=True)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    records = load_jsonl(args.input, args.limit)
    if args.dataset == "fever":
        result = fever_llm_only(records)
    elif args.dataset == "icews14":
        result = icews_temporal_baseline(records)
    else:
        result = yago_rule_baseline(records)
    write_json(args.output / "baseline.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
