from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


RELATION_SIGNATURES = {
    "wasBornIn": ("Person", "Location"),
    "diedIn": ("Person", "Location"),
    "worksAt": ("Person", "Organization"),
    "playsFor": ("Person", "Organization"),
    "hasWonPrize": ("Person", "Award"),
    "isMarriedTo": ("Person", "Person"),
    "owns": ("Person", "Entity"),
    "graduatedFrom": ("Person", "Organization"),
    "isAffiliatedTo": ("Person", "Organization"),
    "created": ("Person", "CreativeWork"),
    "isLocatedIn": ("Entity", "Location"),
    "isCitizenOf": ("Person", "Location"),
    "hasCapital": ("Country", "Location"),
    "participatedIn": ("Entity", "Event"),
    "hasOfficialLanguage": ("Country", "Language"),
    "directed": ("Person", "CreativeWork"),
    "actedIn": ("Person", "CreativeWork"),
    "wroteMusicFor": ("Person", "CreativeWork"),
    "hasGender": ("Person", "Gender"),
    "hasMusicalRole": ("Person", "Role"),
    "hasChild": ("Person", "Person"),
    "livesIn": ("Person", "Location"),
    "happenedIn": ("Event", "Location"),
    "isConnectedTo": ("Location", "Location"),
}

TYPE_HINTS = {
    "AFC_": "Organization",
    "FC_": "Organization",
    "_F.C.": "Organization",
    "_FC": "Organization",
    "University_": "Organization",
    "Airport": "Location",
    "City": "Location",
    "County": "Location",
    "Province": "Location",
    "Country": "Country",
    "National_": "Organization",
    "film": "CreativeWork",
    "Album": "CreativeWork",
    "Award": "Award",
    "language": "Language",
    "male": "Gender",
    "female": "Gender",
}


def clean_name(value: str) -> str:
    return value.strip().strip("<>")


def infer_entity_type(label: str) -> str:
    normalized = label.replace("_", " ")
    for needle, inferred in TYPE_HINTS.items():
        if needle in label:
            return inferred
    lower = normalized.lower()
    if "city" in lower or "province" in lower or "county" in lower:
        return "Location"
    if "film" in lower or "album" in lower or "song" in lower:
        return "CreativeWork"
    if "award" in lower or "prize" in lower:
        return "Award"
    return "Entity"


def read_entities(path: Path) -> Dict[str, str]:
    entities: Dict[str, str] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            parts = line.strip().split("\t")
            if len(parts) >= 2:
                entities[parts[1]] = clean_name(parts[0])
    return entities


def read_relations(path: Path) -> Dict[str, str]:
    relations: Dict[str, str] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            parts = line.strip().split("\t")
            if len(parts) >= 2:
                relations[parts[1]] = clean_name(parts[0])
    return relations


def iter_rows(path: Path, limit: Optional[int]) -> Iterable[Tuple[int, str, str, str]]:
    with path.open("r", encoding="utf-8") as handle:
        for idx, line in enumerate(handle):
            if limit is not None and idx >= limit:
                break
            parts = line.strip().split("\t")
            if len(parts) < 3:
                continue
            yield idx, parts[0], parts[1], parts[2]


def record_for(entity_id: str, label: str, entity_type: str) -> Dict[str, str]:
    return {"id": f"yago:E{entity_id}", "label": label, "type": entity_type}


def context_record(split: str, relation: str) -> Dict[str, Optional[str]]:
    return {
        "domain": "yago_type_reasoning",
        "time_start": None,
        "time_end": None,
        "location": None,
        "condition": f"split={split};relation={relation}",
    }


def evidence_record(split: str, idx: int, subj: str, rel: str, obj: str) -> Dict[str, Any]:
    return {
        "source_id": f"YAGO/{split}:{idx}",
        "source_type": "typed_triple",
        "snippet": f"YAGO {split} row {idx}: ({subj}, {rel}, {obj})",
        "weight": 0.88,
    }


def build_records(
    data_root: Path,
    train_limit: Optional[int],
    valid_limit: Optional[int],
    test_limit: Optional[int],
) -> List[Dict[str, Any]]:
    root = data_root / "YAGO"
    entities = read_entities(root / "entity2id.txt")
    relations = read_relations(root / "relation2id.txt")
    records: List[Dict[str, Any]] = []

    split_to_limit = {"train": train_limit, "valid": valid_limit, "test": test_limit}
    split_to_path = {"train": root / "train.txt", "valid": root / "valid.txt", "test": root / "test.txt"}

    entity_pool: Dict[str, List[Tuple[str, str]]] = {}
    for split, path in split_to_path.items():
        if not path.exists():
            continue
        for idx, subj, rel_id, obj in iter_rows(path, split_to_limit[split]):
            relation = relations.get(rel_id, f"yago_relation_{rel_id}")
            if relation not in RELATION_SIGNATURES:
                continue
            subj_label = entities.get(subj, subj)
            obj_label = entities.get(obj, obj)
            subj_type, obj_type = RELATION_SIGNATURES[relation]
            entity_pool.setdefault(subj_type, []).append((subj, subj_label))
            entity_pool.setdefault(obj_type, []).append((obj, obj_label))
            records.append(
                {
                    "assertion_id": f"yago-{split}-{idx}",
                    "dataset": "YAGO",
                    "split": split,
                    "task": "ontology_validation",
                    "subject": record_for(subj, subj_label, subj_type),
                    "relation": relation,
                    "object": record_for(obj, obj_label, obj_type),
                    "context": context_record(split, relation),
                    "evidence": evidence_record(split, idx, subj, relation, obj),
                    "confidence": 0.88,
                    "polarity": True,
                    "is_gold": True,
                }
            )

    test_path = split_to_path["test"]
    if test_path.exists():
        for idx, subj, rel_id, obj in iter_rows(test_path, split_to_limit["test"]):
            relation = relations.get(rel_id, f"yago_relation_{rel_id}")
            if relation not in RELATION_SIGNATURES:
                continue
            subj_label = entities.get(subj, subj)
            obj_label = entities.get(obj, obj)
            subj_type, obj_type = RELATION_SIGNATURES[relation]
            rank_group = f"yago:{idx}"
            records.append(
                {
                    "assertion_id": f"yago-test-{idx}-gold",
                    "dataset": "YAGO",
                    "split": "test",
                    "task": "ontology_validation",
                    "rank_group": rank_group,
                    "candidate_label": "gold",
                    "gold_label": "gold",
                    "is_gold": True,
                    "subject": record_for(subj, subj_label, subj_type),
                    "relation": relation,
                    "object": record_for(obj, obj_label, obj_type),
                    "context": context_record("test", relation),
                    "evidence": evidence_record("test", idx, subj, relation, obj),
                    "confidence": 0.90,
                    "polarity": True,
                }
            )
            for neg_idx in range(3):
                if neg_idx % 2 == 0:
                    pool = entity_pool.get("Location", []) or entity_pool.get("Entity", [])
                    if not pool:
                        continue
                    neg_subj, neg_subj_label = pool[(idx + neg_idx) % len(pool)]
                    neg_obj, neg_obj_label = obj, obj_label
                    neg_subj_type = "Location"
                    neg_obj_type = "Person"
                    corruption = "subject"
                else:
                    pool = entity_pool.get("Person", []) or entity_pool.get("Entity", [])
                    if not pool:
                        continue
                    neg_subj, neg_subj_label = subj, subj_label
                    neg_obj, neg_obj_label = pool[(idx + neg_idx) % len(pool)]
                    neg_subj_type = "Person"
                    neg_obj_type = "Location"
                    corruption = "object"
                records.append(
                    {
                        "assertion_id": f"yago-test-{idx}-neg{neg_idx}",
                        "dataset": "YAGO",
                        "split": "test",
                        "task": "ontology_validation",
                        "rank_group": rank_group,
                        "candidate_label": f"negative_{corruption}",
                        "gold_label": "gold",
                        "is_gold": False,
                        "subject": record_for(neg_subj, neg_subj_label, neg_subj_type),
                        "relation": relation,
                        "object": record_for(neg_obj, neg_obj_label, neg_obj_type),
                        "context": context_record("test", relation),
                        "evidence": evidence_record("test", idx, neg_subj, relation, neg_obj),
                        "confidence": 0.78 if neg_idx == 0 else 0.62,
                        "polarity": True,
                        "corruption": corruption,
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


def parse_optional_limit(value: int) -> Optional[int]:
    return None if value < 0 else value


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare YAGO type reasoning assertions.")
    parser.add_argument("--data-root", default="../data", type=Path)
    parser.add_argument("--output-dir", default="data/processed/yago_type_reasoning", type=Path)
    parser.add_argument("--train-limit", type=int, default=5000, help="Maximum train triples; use -1 for full split.")
    parser.add_argument("--valid-limit", type=int, default=2000, help="Maximum valid triples; use -1 for full split.")
    parser.add_argument("--test-limit", type=int, default=2000, help="Maximum test triples; use -1 for full split.")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    train_limit = parse_optional_limit(args.train_limit)
    valid_limit = parse_optional_limit(args.valid_limit)
    test_limit = parse_optional_limit(args.test_limit)
    records = build_records(args.data_root, train_limit, valid_limit, test_limit)
    out_path = args.output_dir / "assertions.jsonl"
    count = write_jsonl(out_path, records)
    summary = {
        "dataset": "YAGO",
        "train_limit": train_limit,
        "valid_limit": valid_limit,
        "test_limit": test_limit,
        "records": count,
        "output": str(out_path),
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
