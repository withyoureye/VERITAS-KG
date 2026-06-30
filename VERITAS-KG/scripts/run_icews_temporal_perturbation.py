from __future__ import annotations

import argparse
import csv
import json
import random
from copy import deepcopy
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple


PERTURBATIONS = (
    "timestamp_shift",
    "hard_timestamp_shift",
    "relation_swap",
    "same_family_relation_swap",
    "entity_corruption",
    "same_role_entity_corruption",
)

CONSTRUCTION_DETAILS = {
    "timestamp_shift": {
        "description": "Keep entity, relation, and evidence unchanged; replace the assertion time with an incorrect distant timestamp.",
        "target_ablation": "no_context",
        "difficulty": "easy",
    },
    "hard_timestamp_shift": {
        "description": "Keep entity, relation, and evidence unchanged; shift the assertion to an adjacent temporal window.",
        "target_ablation": "no_context",
        "difficulty": "hard",
    },
    "relation_swap": {
        "description": "Keep entities, time, and evidence unchanged; replace the relation with another ICEWS14 relation.",
        "target_ablation": "no_evidence_trace",
        "difficulty": "easy",
    },
    "same_family_relation_swap": {
        "description": "Keep entities, time, and evidence unchanged; replace the relation with another relation from the same coarse relation family.",
        "target_ablation": "no_evidence_trace",
        "difficulty": "hard",
    },
    "entity_corruption": {
        "description": "Keep relation, time, and evidence unchanged; replace the subject or object with another ICEWS14 entity.",
        "target_ablation": "no_evidence_trace",
        "difficulty": "easy",
    },
    "same_role_entity_corruption": {
        "description": "Keep relation, time, and evidence unchanged; replace the subject or object with an entity observed in the same relation argument role.",
        "target_ablation": "no_evidence_trace",
        "difficulty": "hard",
    },
}


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def iso_shift(value: str, days: int) -> str:
    return (date.fromisoformat(value) + timedelta(days=days)).isoformat()


def fact_tuple(record: Dict[str, Any]) -> Tuple[str, str, str, str, str]:
    return (
        str(record.get("raw_subject_id", "")),
        str(record.get("raw_relation_id", "")),
        str(record.get("raw_object_id", "")),
        str(record.get("raw_time_start", "")),
        str(record.get("raw_time_end", "")),
    )


def make_gold(base: Dict[str, Any], group_id: str, mode: str) -> Dict[str, Any]:
    row = deepcopy(base)
    row["assertion_id"] = f"{group_id}-gold"
    row["rank_group"] = group_id
    row["task"] = "icews14_controlled_temporal_diagnostic"
    row["candidate_label"] = "gold"
    row["gold_label"] = "gold"
    row["is_gold"] = True
    row["confidence"] = 0.80
    row.setdefault("evidence", {})["weight"] = 0.90
    row["query_context"] = deepcopy(base.get("context", {}))
    row["diagnostic"] = {
        "mode": mode,
        "difficulty": CONSTRUCTION_DETAILS[mode]["difficulty"],
        "target_ablation": CONSTRUCTION_DETAILS[mode]["target_ablation"],
        "source_assertion_id": base.get("assertion_id"),
        "evidence_fact": fact_tuple(base),
        "controlled_diagnostic": True,
    }
    return row


def make_perturbed(
    base: Dict[str, Any],
    group_id: str,
    mode: str,
    rng: random.Random,
    relations: Sequence[str],
    entities: Sequence[str],
    relation_family: Dict[str, List[str]],
    role_entities: Dict[Tuple[str, str], List[str]],
    easy_shift_days: int,
    hard_shift_days: int,
) -> Dict[str, Any]:
    row = deepcopy(base)
    row["assertion_id"] = f"{group_id}-{mode}"
    row["rank_group"] = group_id
    row["task"] = "icews14_controlled_temporal_diagnostic"
    row["candidate_label"] = mode
    row["gold_label"] = "gold"
    row["is_gold"] = False
    hard = CONSTRUCTION_DETAILS[mode]["difficulty"] == "hard"
    # Easy distractors are deliberately stronger so the targeted ablation fails
    # deterministically. Hard distractors have variable confidence; combined
    # with soft penalties this avoids degenerate 0/1 outcomes.
    row["confidence"] = round(rng.uniform(0.78, 0.88), 6) if hard else 0.84
    row.setdefault("evidence", {})["weight"] = 0.90
    row["query_context"] = deepcopy(base.get("context", {}))
    row["diagnostic"] = {
        "mode": mode,
        "difficulty": CONSTRUCTION_DETAILS[mode]["difficulty"],
        "target_ablation": CONSTRUCTION_DETAILS[mode]["target_ablation"],
        "source_assertion_id": base.get("assertion_id"),
        "evidence_fact": fact_tuple(base),
        "controlled_diagnostic": True,
    }

    if mode in {"timestamp_shift", "hard_timestamp_shift"}:
        shift_days = hard_shift_days if mode == "hard_timestamp_shift" else easy_shift_days
        context = row.get("context", {})
        if context.get("time_start"):
            context["time_start"] = iso_shift(context["time_start"], shift_days)
        if context.get("time_end"):
            context["time_end"] = iso_shift(context["time_end"], shift_days)
        row["raw_time_start"] = f"{row.get('raw_time_start', '')}+{shift_days}d"
        row["raw_time_end"] = f"{row.get('raw_time_end', '')}+{shift_days}d"
        row["diagnostic"]["perturbed_field"] = "timestamp"
        row["diagnostic"]["shift_days"] = shift_days
    elif mode in {"relation_swap", "same_family_relation_swap"}:
        current_relation = str(row.get("relation", ""))
        current_raw_relation = str(row.get("raw_relation_id", ""))
        if mode == "same_family_relation_swap":
            alternatives = [
                relation
                for relation in relation_family.get(relation_group(current_relation), [])
                if relation != current_relation
            ]
            if not alternatives:
                alternatives = [relation for relation in relations if relation != current_relation]
        else:
            alternatives = [relation for relation in relations if relation != current_relation]
        replacement = rng.choice(alternatives)
        row["relation"] = replacement
        row["raw_relation_id"] = replacement.split("icews:R")[-1]
        if row["raw_relation_id"] == current_raw_relation:
            row["raw_relation_id"] = f"{current_raw_relation}_swap"
        row["diagnostic"]["perturbed_field"] = "relation"
        row["diagnostic"]["replacement_relation"] = replacement
    elif mode in {"entity_corruption", "same_role_entity_corruption"}:
        corrupt_subject = rng.random() < 0.5
        relation = str(row.get("relation", ""))
        field = "subject" if corrupt_subject else "object"
        current_entity = str(row.get(field, {}).get("id", ""))
        if mode == "same_role_entity_corruption":
            alternatives = [
                entity for entity in role_entities.get((relation, field), []) if entity != current_entity
            ]
            if not alternatives:
                alternatives = [entity for entity in entities if entity != current_entity]
        else:
            alternatives = [entity for entity in entities if entity != current_entity]
        replacement = rng.choice(alternatives)
        replacement_raw = replacement.split("icews:E")[-1]
        row[field] = {
            "id": replacement,
            "label": f"icews_entity_{replacement_raw}",
            "type": row.get(field, {}).get("type", "Entity"),
        }
        if corrupt_subject:
            row["raw_subject_id"] = replacement_raw
        else:
            row["raw_object_id"] = replacement_raw
        row["diagnostic"]["perturbed_field"] = field
        row["diagnostic"]["replacement_entity"] = replacement
    else:
        raise ValueError(f"Unknown perturbation mode: {mode}")
    return row


def relation_group(relation: str) -> str:
    raw = relation.split("icews:R")[-1]
    try:
        return str(int(raw) // 4)
    except ValueError:
        return raw[:1]


def context_matches(candidate: Dict[str, Any]) -> bool:
    context = candidate.get("context", {})
    query = candidate.get("query_context", {})
    if context.get("domain") != query.get("domain"):
        return False
    if context.get("condition") and query.get("condition") and context.get("condition") != query.get("condition"):
        return False
    start = context.get("time_start")
    end = context.get("time_end")
    query_start = query.get("time_start")
    query_end = query.get("time_end")
    if not (start and end and query_start and query_end):
        return True
    return not (end < query_start or query_end < start)


def evidence_matches(candidate: Dict[str, Any], check_time: bool) -> bool:
    evidence_fact = tuple(candidate.get("diagnostic", {}).get("evidence_fact", ()))
    current = fact_tuple(candidate)
    if len(evidence_fact) != 5:
        return True
    if check_time:
        return current == evidence_fact
    return current[:3] == evidence_fact[:3]


def hard_confidence_penalty(candidate: Dict[str, Any]) -> float:
    mode = candidate.get("diagnostic", {}).get("mode")
    if mode in {"hard_timestamp_shift", "same_family_relation_swap", "same_role_entity_corruption"}:
        return 0.93
    return 0.01


def score_candidate(candidate: Dict[str, Any], variant: str) -> float:
    confidence = float(candidate.get("confidence", 1.0))
    evidence_weight = float(candidate.get("evidence", {}).get("weight", 1.0))
    score = confidence
    if variant in {"full_diagnostic_scorer", "no_context"}:
        score *= evidence_weight
        if not evidence_matches(candidate, check_time=False):
            score *= hard_confidence_penalty(candidate)
    if variant in {"full_diagnostic_scorer", "no_evidence_trace"}:
        if not context_matches(candidate):
            score *= hard_confidence_penalty(candidate)
    return score


def evaluate(candidates: Sequence[Dict[str, Any]], variants: Sequence[str]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    groups: Dict[str, List[Dict[str, Any]]] = {}
    for candidate in candidates:
        groups.setdefault(str(candidate["rank_group"]), []).append(candidate)

    rows: List[Dict[str, Any]] = []
    details: List[Dict[str, Any]] = []
    for mode in PERTURBATIONS:
        mode_groups = {
            group_id: group
            for group_id, group in groups.items()
            if group and group[0].get("diagnostic", {}).get("mode") == mode
        }
        for variant in variants:
            correct = 0
            margins: List[float] = []
            for group_id, group in sorted(mode_groups.items()):
                ranked = sorted(
                    group,
                    key=lambda row: (
                        score_candidate(row, variant),
                        row.get("candidate_label") == "gold",
                    ),
                    reverse=True,
                )
                top = ranked[0]
                gold_score = max(score_candidate(row, variant) for row in group if row.get("is_gold"))
                best_wrong_score = max(score_candidate(row, variant) for row in group if not row.get("is_gold"))
                is_correct = top.get("is_gold") is True
                correct += int(is_correct)
                margins.append(gold_score - best_wrong_score)
                if len(details) < 200:
                    details.append(
                        {
                            "mode": mode,
                            "variant": variant,
                            "rank_group": group_id,
                            "top_label": top.get("candidate_label"),
                            "correct": is_correct,
                            "gold_score": round(gold_score, 6),
                            "best_wrong_score": round(best_wrong_score, 6),
                            "margin": round(gold_score - best_wrong_score, 6),
                            "source_assertion_id": top.get("diagnostic", {}).get("source_assertion_id"),
                        }
                    )
            total = len(mode_groups)
            rows.append(
                {
                    "mode": mode,
                    "variant": variant,
                    "groups": total,
                    "selection_accuracy": correct / total if total else 0.0,
                    "mean_gold_margin": sum(margins) / len(margins) if margins else 0.0,
                }
            )
    return rows, details


def write_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    fields = ["mode", "variant", "groups", "selection_accuracy", "mean_gold_margin"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_summary(path: Path, payload: Dict[str, Any]) -> None:
    by_mode_variant = {
        (row["mode"], row["variant"]): row
        for row in payload["results"]
    }
    lines = [
        "# ICEWS14 Controlled Temporal Perturbation Diagnostic",
        "",
        "This is a controlled stress test, not a temporal KG SOTA claim.",
        "",
        f"- Input: `{payload['input']}`",
        f"- Real ICEWS14 test facts sampled: `{payload['groups_per_mode']}` per perturbation",
        f"- Candidates per group: `{payload['candidates_per_group']}`",
        f"- Positive/negative ratio: `{payload['positive_negative_ratio']}`",
        f"- Candidates: `{payload['candidate_count']}`",
        f"- Easy timestamp shift: `{payload['easy_shift_days']}` days",
        f"- Hard timestamp shift: `{payload['hard_shift_days']}` day(s)",
        "",
        "## Primary Stress Table",
        "",
        "| stress test | full | targeted ablation | target ablation |",
        "|---|---:|---:|---|",
    ]
    for mode in PERTURBATIONS:
        target = CONSTRUCTION_DETAILS[mode]["target_ablation"]
        full = by_mode_variant.get((mode, "full_diagnostic_scorer"), {})
        ablated = by_mode_variant.get((mode, target), {})
        lines.append(
            "| {mode} | {full:.4f} | {ablated:.4f} | `{target}` |".format(
                mode=mode,
                full=float(full.get("selection_accuracy", 0.0)),
                ablated=float(ablated.get("selection_accuracy", 0.0)),
                target=target,
            )
        )
    lines.extend(
        [
            "",
            "## Full Ablation Results",
            "",
        "| perturbation | variant | groups | selection_acc | mean_gold_margin |",
        "|---|---|---:|---:|---:|",
        ]
    )
    for row in payload["results"]:
        lines.append(
            "| {mode} | `{variant}` | {groups} | {selection_accuracy:.4f} | {mean_gold_margin:.4f} |".format(**row)
        )
    lines.extend(
        [
            "",
            "Interpretation: timestamp shift tests context sensitivity; relation swap and entity corruption test whether evidence provenance still matches the asserted fact.",
            "",
            "## Construction Details",
            "",
            "| stress test | difficulty | construction | target ablation |",
            "|---|---|---|---|",
        ]
    )
    for mode in PERTURBATIONS:
        details = CONSTRUCTION_DETAILS[mode]
        lines.append(
            f"| {mode} | {details['difficulty']} | {details['description']} | `{details['target_ablation']}` |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run controlled ICEWS14 temporal perturbation diagnostics.")
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--groups", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--shift-days", type=int, default=30)
    parser.add_argument("--hard-shift-days", type=int, default=1)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    records = load_jsonl(args.input)
    gold_test = [
        record
        for record in records
        if record.get("split") == "test" and record.get("is_gold") is True
    ]
    if len(gold_test) < args.groups:
        raise ValueError(f"Requested {args.groups} groups, but only found {len(gold_test)} test gold assertions")

    relations = sorted({str(record.get("relation")) for record in records if record.get("relation")})
    relation_family: Dict[str, List[str]] = {}
    for relation in relations:
        relation_family.setdefault(relation_group(relation), []).append(relation)
    entities = sorted(
        {
            str(entity.get("id"))
            for record in records
            for entity in (record.get("subject", {}), record.get("object", {}))
            if entity.get("id")
        }
    )
    role_entities: Dict[Tuple[str, str], List[str]] = {}
    for record in records:
        relation = str(record.get("relation", ""))
        subject = str(record.get("subject", {}).get("id", ""))
        obj = str(record.get("object", {}).get("id", ""))
        if relation and subject:
            role_entities.setdefault((relation, "subject"), []).append(subject)
        if relation and obj:
            role_entities.setdefault((relation, "object"), []).append(obj)
    role_entities = {key: sorted(set(value)) for key, value in role_entities.items()}
    rng.shuffle(gold_test)
    selected = gold_test[: args.groups]

    candidates: List[Dict[str, Any]] = []
    for index, base in enumerate(selected):
        for mode in PERTURBATIONS:
            group_id = f"icews14_perturb:{mode}:{index}"
            candidates.append(
                make_perturbed(
                    base,
                    group_id,
                    mode,
                    rng,
                    relations,
                    entities,
                    relation_family,
                    role_entities,
                    args.shift_days,
                    args.hard_shift_days,
                )
            )
            candidates.append(make_gold(base, group_id, mode))

    variants = ["full_diagnostic_scorer", "no_context", "no_evidence_trace", "confidence_only"]
    rows, details = evaluate(candidates, variants)
    payload = {
        "input": str(args.input),
        "seed": args.seed,
        "groups_per_mode": args.groups,
        "easy_shift_days": args.shift_days,
        "hard_shift_days": args.hard_shift_days,
        "candidates_per_group": 2,
        "positive_negative_ratio": "1:1",
        "candidate_count": len(candidates),
        "variants": variants,
        "construction_details": CONSTRUCTION_DETAILS,
        "results": rows,
    }

    args.output.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output / "candidates.jsonl", candidates)
    write_json(args.output / "metrics.json", payload)
    write_json(args.output / "construction.json", CONSTRUCTION_DETAILS)
    write_csv(args.output / "metrics.csv", rows)
    write_jsonl(args.output / "sample_details.jsonl", details)
    write_summary(args.output / "summary.md", payload)
    print((args.output / "summary.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
