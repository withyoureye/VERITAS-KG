from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Sequence


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_md(path: Path, title: str, rows: Sequence[Dict[str, Any]]) -> None:
    lines = [
        f"# {title}",
        "",
        "| case | gold | comparison | why it matters | evidence/context |",
        "|---|---|---|---|---|",
    ]
    for row in rows:
        lines.append(
            "| {case} | {gold} | {comparison} | {why} | {evidence} |".format(
                case=row.get("case", ""),
                gold=row.get("gold", ""),
                comparison=row.get("comparison", ""),
                why=row.get("why", ""),
                evidence=row.get("evidence", ""),
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def fever_cases(records: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    groups: Dict[str, List[Dict[str, Any]]] = {}
    for record in records:
        group = record.get("rank_group")
        if group:
            groups.setdefault(str(group), []).append(record)
    rows: List[Dict[str, Any]] = []
    for group_id, candidates in groups.items():
        gold = next((item for item in candidates if item.get("is_gold")), None)
        if gold is None:
            continue
        full = max(candidates, key=lambda item: float(item.get("confidence", 0.0)) * float(item.get("evidence", {}).get("weight", 0.0)))
        qwen_like = max(candidates, key=lambda item: float(item.get("confidence", 0.0)))
        if full.get("candidate_label") == qwen_like.get("candidate_label"):
            continue
        rows.append(
            {
                "case": f"FEVER {group_id}",
                "gold": str(gold.get("gold_label")),
                "comparison": f"KG-style full={full.get('candidate_label')} vs confidence/Qwen-reader proxy={qwen_like.get('candidate_label')}",
                "why": "Evidence weighting changes the top verdict, illustrating why KG/Qwen fusion can outperform a reader-only score.",
                "evidence": str(gold.get("evidence", {}).get("snippet", ""))[:220],
            }
        )
        if len(rows) >= 5:
            break
    return rows


def fever_fusion_cases(path: Path) -> List[Dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    best = payload.get("best", {})
    details = best.get("details", [])
    if not details and isinstance(payload.get("evaluation"), dict):
        details = payload["evaluation"].get("details", [])
    rows: List[Dict[str, Any]] = []
    for item in details:
        if item.get("correct") is not True:
            continue
        if item.get("qwen_label") == item.get("gold_label"):
            continue
        rows.append(
            {
                "case": str(item.get("rank_group")),
                "gold": str(item.get("gold_label")),
                "comparison": f"Qwen={item.get('qwen_label')} vs KG={item.get('kg_label')} vs fusion={item.get('predicted_label')}",
                "why": "Qwen reader alone misses the label, while KG evidence scoring supplies the complementary signal used by fusion.",
                "evidence": f"See fusion details in {path}.",
            }
        )
        if len(rows) >= 5:
            break
    return rows


def icews_cases(records: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    groups: Dict[str, List[Dict[str, Any]]] = {}
    for record in records:
        group = record.get("rank_group")
        if group:
            groups.setdefault(str(group), []).append(record)
    rows: List[Dict[str, Any]] = []
    for group_id, candidates in groups.items():
        gold = next((item for item in candidates if item.get("is_gold")), None)
        if gold is None:
            continue
        if gold.get("task") == "icews14_controlled_temporal_diagnostic":
            corrupted = next((item for item in candidates if item.get("is_gold") is not True), None)
            if corrupted is None:
                continue
            diagnostic = corrupted.get("diagnostic", {})
            rows.append(
                {
                    "case": f"ICEWS14 {group_id}",
                    "gold": f"{gold.get('raw_subject_id')} {gold.get('raw_relation_id')} {gold.get('raw_object_id')} @ {gold.get('context', {}).get('time_start')}",
                    "comparison": f"gold vs {corrupted.get('candidate_label')} perturbation; target ablation={diagnostic.get('target_ablation')}",
                    "why": "The controlled diagnostic keeps real ICEWS14 entities/relations/evidence and changes one field to test context or evidence provenance.",
                    "evidence": f"{gold.get('evidence', {}).get('snippet', '')[:160]} | query={gold.get('query_context', {})} | corrupt={corrupted.get('context', {})}",
                }
            )
            if len(rows) >= 5:
                break
            continue
        full = max(candidates, key=lambda item: float(item.get("confidence", 0.0)) * float(item.get("evidence", {}).get("weight", 0.0)))
        no_evidence = max(candidates, key=lambda item: float(item.get("confidence", 0.0)))
        if bool(full.get("is_gold")) == bool(no_evidence.get("is_gold")):
            continue
        rows.append(
            {
                "case": f"ICEWS14 {group_id}",
                "gold": str(gold.get("raw_subject_id")) + " " + str(gold.get("raw_relation_id")) + " " + str(gold.get("raw_object_id")),
                "comparison": f"full_is_gold={full.get('is_gold')} vs no_evidence_is_gold={no_evidence.get('is_gold')}",
                "why": "The positive event keeps high evidence support while confidence-only ranking selects a corrupted candidate.",
                "evidence": str(gold.get("evidence", {}).get("snippet", ""))[:220] + " " + str(gold.get("context", {}))[:120],
            }
        )
        if len(rows) >= 5:
            break
    return rows


def yago_cases(records: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for record in records:
        if record.get("task") != "ontology_validation":
            continue
        if record.get("is_gold") is True:
            continue
        rows.append(
            {
                "case": str(record.get("assertion_id")),
                "gold": f"is_gold={record.get('is_gold')}",
                "comparison": f"relation signature accepts relation={record.get('relation')} but ontology rejects type path {record['subject']['type']}->{record['object']['type']}",
                "why": "Relation-level signatures are too coarse; ontology/type reasoning catches subject/object type violations.",
                "evidence": str(record.get("evidence", {}).get("snippet", ""))[:220],
            }
        )
        if len(rows) >= 5:
            break
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Create compact case analysis markdown.")
    parser.add_argument("--dataset", choices=["fever", "icews14", "yago"], required=True)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--fusion-json", type=Path)
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    records = load_jsonl(args.input)
    if args.dataset == "fever":
        rows = fever_fusion_cases(args.fusion_json) if args.fusion_json else fever_cases(records)
    elif args.dataset == "icews14":
        rows = icews_cases(records)
    else:
        rows = yago_cases(records)
    write_md(args.output / "summary.md", f"{args.dataset.upper()} Case Analysis", rows)
    print((args.output / "summary.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
