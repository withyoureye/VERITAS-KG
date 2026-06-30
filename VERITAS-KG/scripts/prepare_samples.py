from __future__ import annotations

import argparse
import json
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


YAGO_SIGNATURES = {
    "wasBornIn": ("Person", "Location"),
    "diedIn": ("Person", "Location"),
    "worksAt": ("Person", "Organization"),
    "playsFor": ("Person", "Organization"),
    "hasWonPrize": ("Person", "Award"),
    "isMarriedTo": ("Person", "Person"),
    "owns": ("Person", "Entity"),
    "graduatedFrom": ("Person", "Organization"),
    "isAffiliatedTo": ("Person", "Organization"),
    "created": ("Person", "Artifact"),
}

FEVER_LABELS = ["SUPPORTS", "REFUTES", "NOT ENOUGH INFO"]


def clean_name(value: str) -> str:
    return value.strip().strip("<>").replace("_", " ")


def read_yago_entities(path: Path) -> Dict[str, str]:
    entities: Dict[str, str] = {}
    if not path.exists():
        return entities
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.split("\t")
        if len(parts) >= 2:
            entities[parts[1]] = clean_name(parts[0])
    return entities


def read_yago_relations(path: Path) -> Dict[str, str]:
    relations: Dict[str, str] = {}
    if not path.exists():
        return relations
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.split("\t")
        if len(parts) >= 2:
            relations[parts[1]] = clean_name(parts[0])
    return relations


def entity_record(entity_id: str, label: str, entity_type: str) -> Dict[str, str]:
    return {"id": entity_id, "label": label, "type": entity_type}


def context_record(
    domain: str,
    time_start: Optional[date] = None,
    time_end: Optional[date] = None,
    location: Optional[str] = None,
    condition: Optional[str] = None,
) -> Dict[str, Optional[str]]:
    return {
        "domain": domain,
        "time_start": time_start.isoformat() if time_start else None,
        "time_end": time_end.isoformat() if time_end else None,
        "location": location,
        "condition": condition,
    }


def evidence_record(
    source_id: str,
    source_type: str,
    snippet: str,
    weight: float,
) -> Dict[str, Any]:
    return {
        "source_id": source_id,
        "source_type": source_type,
        "snippet": snippet,
        "weight": round(weight, 4),
    }


def icews_date(hour_value: str) -> date:
    hours = int(float(hour_value))
    return date(2014, 1, 1) + timedelta(days=max(hours, 0) // 24)


def load_icews(data_root: Path, limit: int, inject_conflicts: int) -> List[Dict[str, Any]]:
    path = data_root / "ICEWS14" / "train.txt"
    records: List[Dict[str, Any]] = []
    if not path.exists():
        return records

    with path.open("r", encoding="utf-8") as handle:
        for idx, line in enumerate(handle):
            if idx >= limit:
                break
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            subj, rel, obj, start_raw, end_raw = parts[:5]
            start = icews_date(start_raw)
            end = icews_date(end_raw) if int(float(end_raw)) > int(float(start_raw)) else start
            confidence = 0.62 + (idx % 17) * 0.015
            weight = 0.68 + (idx % 13) * 0.012
            records.append(
                {
                    "assertion_id": f"icews14-train-{idx}",
                    "dataset": "ICEWS14",
                    "split": "train",
                    "task": "conflict_detection",
                    "subject": entity_record(
                        f"icews:E{subj}", f"icews_entity_{subj}", "Entity"
                    ),
                    "relation": f"icews:R{rel}",
                    "object": entity_record(
                        f"icews:E{obj}", f"icews_entity_{obj}", "Entity"
                    ),
                    "context": context_record(
                        "icews_event", start, end, condition="observed"
                    ),
                    "evidence": evidence_record(
                        f"ICEWS14/train:{idx}",
                        "event_quad",
                        f"ICEWS14 row {idx}: {subj} {rel} {obj} at {start.isoformat()}",
                        weight,
                    ),
                    "confidence": round(confidence, 4),
                    "polarity": True,
                    "is_gold": True,
                }
            )

    injected = 0
    for base in list(records[:inject_conflicts]):
        copied = dict(base)
        copied["assertion_id"] = f"{base['assertion_id']}-negative-{injected}"
        copied["polarity"] = False
        copied["confidence"] = 0.58 if injected % 2 == 0 else 0.52
        copied["is_gold"] = False
        copied["evidence"] = evidence_record(
            f"{base['evidence']['source_id']}:negative:{injected}",
            "simulated_extraction",
            f"Injected contradictory candidate for {base['assertion_id']}",
            0.35 if injected % 2 == 0 else 0.42,
        )
        if injected % 2 == 1:
            old_context = copied["context"]
            shifted = date.fromisoformat(old_context["time_start"]) + timedelta(days=45)
            copied["context"] = context_record(
                old_context["domain"],
                shifted,
                shifted,
                old_context["location"],
                "observed_later",
            )
            copied["non_overlapping_negative"] = True
        else:
            copied["overlapping_negative"] = True
        records.append(copied)
        injected += 1
    return records


def flatten_fever_evidence(raw_evidence: Any) -> Tuple[str, str]:
    pages: List[str] = []
    if isinstance(raw_evidence, list):
        for group in raw_evidence:
            if not isinstance(group, list):
                continue
            for item in group:
                if isinstance(item, list) and len(item) >= 4 and item[2] is not None:
                    pages.append(f"{item[2]}#{item[3]}")
    if not pages:
        return "claim_only", "No page-level evidence supplied in this FEVER subset."
    return pages[0], "; ".join(pages[:4])


def load_fever(data_root: Path, limit: int) -> List[Dict[str, Any]]:
    path = data_root / "fever" / "paper_dev.jsonl"
    records: List[Dict[str, Any]] = []
    if not path.exists():
        return records

    with path.open("r", encoding="utf-8") as handle:
        for idx, line in enumerate(handle):
            if idx >= limit:
                break
            item = json.loads(line)
            claim_id = str(item["id"])
            gold_label = str(item["label"])
            evidence_id, evidence_text = flatten_fever_evidence(item.get("evidence"))
            for label_idx, label in enumerate(FEVER_LABELS):
                is_gold = label == gold_label
                weight = 0.95 if is_gold and gold_label != "NOT ENOUGH INFO" else 0.75
                if not is_gold:
                    weight = 0.20 + 0.05 * label_idx
                confidence = 0.62 if is_gold else (0.90 if label_idx == 0 else 0.56)
                records.append(
                    {
                        "assertion_id": f"fever-dev-{claim_id}-{label}",
                        "dataset": "FEVER",
                        "split": "paper_dev",
                        "task": "fact_verification",
                        "rank_group": f"fever:{claim_id}",
                        "is_gold": is_gold,
                        "gold_label": gold_label,
                        "candidate_label": label,
                        "subject": entity_record(
                            f"fever:claim:{claim_id}", item["claim"], "Claim"
                        ),
                        "relation": "has_verdict",
                        "object": entity_record(
                            f"fever:verdict:{label}", label, "Verdict"
                        ),
                        "context": context_record(
                            "fever_claim",
                            condition=f"candidate_label={label}",
                        ),
                        "evidence": evidence_record(
                            f"FEVER/{evidence_id}",
                            "wikipedia_sentence" if evidence_id != "claim_only" else "claim",
                            f"{item['claim']} Evidence: {evidence_text}",
                            weight,
                        ),
                        "confidence": confidence,
                        "polarity": True,
                    }
                )
    return records


def load_yago(data_root: Path, limit: int, inject_invalid: int) -> List[Dict[str, Any]]:
    root = data_root / "YAGO"
    labels = read_yago_entities(root / "entity2id.txt")
    relations = read_yago_relations(root / "relation2id.txt")
    path = root / "train.txt"
    records: List[Dict[str, Any]] = []
    entity_types: Dict[str, str] = {}
    if not path.exists():
        return records

    with path.open("r", encoding="utf-8") as handle:
        for idx, line in enumerate(handle):
            if idx >= limit:
                break
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            subj, rel_id, obj, start_raw, end_raw = parts[:5]
            relation = relations.get(rel_id, f"yago_relation_{rel_id}")
            subj_type, obj_type = YAGO_SIGNATURES.get(relation, ("Entity", "Entity"))
            entity_types.setdefault(subj, subj_type)
            entity_types.setdefault(obj, obj_type)
            records.append(
                {
                    "assertion_id": f"yago-train-{idx}",
                    "dataset": "YAGO",
                    "split": "train",
                    "task": "ontology_validation",
                    "subject": entity_record(
                        f"yago:E{subj}", labels.get(subj, f"yago_entity_{subj}"), subj_type
                    ),
                    "relation": relation,
                    "object": entity_record(
                        f"yago:E{obj}", labels.get(obj, f"yago_entity_{obj}"), obj_type
                    ),
                    "context": context_record(
                        "yago_taxonomy",
                        condition=f"valid_time={start_raw}..{end_raw}",
                    ),
                    "evidence": evidence_record(
                        f"YAGO/train:{idx}",
                        "typed_triple",
                        f"YAGO row {idx}: {subj} {relation} {obj}",
                        0.82,
                    ),
                    "confidence": 0.86,
                    "polarity": True,
                    "is_gold": True,
                }
            )

    typed_entities: Dict[str, List[Dict[str, str]]] = {}
    for record in records:
        typed_entities.setdefault(record["subject"]["type"], []).append(record["subject"])
        typed_entities.setdefault(record["object"]["type"], []).append(record["object"])

    def pick_entity(entity_type: str, fallback: Dict[str, str]) -> Dict[str, str]:
        candidates = typed_entities.get(entity_type)
        return candidates[0] if candidates else fallback

    base_relation = "worksAt"
    fallback_subj = entity_record("yago:invalid_location", "Injected Location", "Location")
    fallback_obj = entity_record("yago:invalid_person", "Injected Person", "Person")
    for idx in range(inject_invalid):
        bad_subject = pick_entity("Location", fallback_subj)
        bad_object = pick_entity("Person", fallback_obj)
        records.append(
            {
                "assertion_id": f"yago-invalid-{idx}",
                "dataset": "YAGO",
                "split": "synthetic_invalid",
                "task": "ontology_validation",
                "subject": bad_subject,
                "relation": base_relation,
                "object": bad_object,
                "context": context_record("yago_taxonomy", condition="injected_invalid"),
                "evidence": evidence_record(
                    f"YAGO/injected_invalid:{idx}",
                    "simulated_extraction",
                    "Injected invalid worksAt(Location, Person) candidate.",
                    0.70,
                ),
                "confidence": 0.78,
                "polarity": True,
                "is_gold": False,
                "expected_invalid": True,
            }
        )
    return records


def write_jsonl(path: Path, records: Iterable[Dict[str, Any]]) -> int:
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
            count += 1
    return count


def main() -> None:
    parser = argparse.ArgumentParser(description="Build unified ontology KG sample data.")
    parser.add_argument("--data-root", default="../data", type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--icews-limit", type=int, default=80)
    parser.add_argument("--fever-limit", type=int, default=30)
    parser.add_argument("--yago-limit", type=int, default=40)
    parser.add_argument("--inject-conflicts", type=int, default=8)
    parser.add_argument("--inject-invalid", type=int, default=6)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    records: List[Dict[str, Any]] = []
    records.extend(load_icews(args.data_root, args.icews_limit, args.inject_conflicts))
    records.extend(load_fever(args.data_root, args.fever_limit))
    records.extend(load_yago(args.data_root, args.yago_limit, args.inject_invalid))

    out_path = args.output_dir / "assertions.jsonl"
    count = write_jsonl(out_path, records)
    summary = {
        "output": str(out_path),
        "records": count,
        "datasets": sorted({record["dataset"] for record in records}),
        "tasks": sorted({record["task"] for record in records}),
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
