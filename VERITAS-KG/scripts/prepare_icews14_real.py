from __future__ import annotations

import argparse
import json
import random
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple


BASE_DATE = date(2014, 1, 1)


def parse_time(raw_value: str) -> Tuple[date, int]:
    raw_int = int(float(raw_value))
    return BASE_DATE + timedelta(days=max(raw_int, 0) // 24), raw_int


def entity_record(entity_id: str) -> Dict[str, str]:
    return {
        "id": f"icews:E{entity_id}",
        "label": f"icews_entity_{entity_id}",
        "type": "Entity",
    }


def context_record(start_raw: str, end_raw: str, split: str) -> Dict[str, Optional[str]]:
    start_date, start_offset = parse_time(start_raw)
    end_date, end_offset = parse_time(end_raw)
    if end_offset < start_offset:
        end_date = start_date
        end_offset = start_offset
    return {
        "domain": "icews14_event",
        "time_start": start_date.isoformat(),
        "time_end": end_date.isoformat(),
        "location": None,
        "condition": f"split={split};raw_time={start_offset}..{end_offset}",
    }


def evidence_record(
    split: str,
    row_index: int,
    subj: str,
    rel: str,
    obj: str,
    start_raw: str,
    end_raw: str,
    weight: float,
) -> Dict[str, Any]:
    return {
        "source_id": f"ICEWS14/{split}:{row_index}",
        "source_type": "icews14_quad",
        "snippet": (
            f"ICEWS14 {split} row {row_index}: "
            f"({subj}, {rel}, {obj}) time={start_raw}..{end_raw}"
        ),
        "weight": round(weight, 6),
    }


def iter_split_rows(path: Path, limit: Optional[int]) -> Iterable[Tuple[int, str, str, str, str, str]]:
    if not path.exists():
        return
    with path.open("r", encoding="utf-8") as handle:
        for idx, line in enumerate(handle):
            if limit is not None and idx >= limit:
                break
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            subj, rel, obj, start_raw, end_raw = parts[:5]
            yield idx, subj, rel, obj, start_raw, end_raw


def make_assertion(
    split: str,
    row_index: int,
    subj: str,
    rel: str,
    obj: str,
    start_raw: str,
    end_raw: str,
    confidence: float,
    evidence_weight: float,
    polarity: bool,
    assertion_suffix: str = "",
    rank_group: Optional[str] = None,
    is_gold: bool = True,
    corruption: Optional[str] = None,
) -> Dict[str, Any]:
    assertion_id = f"icews14-real-{split}-{row_index}{assertion_suffix}"
    record: Dict[str, Any] = {
        "assertion_id": assertion_id,
        "dataset": "ICEWS14",
        "split": split,
        "task": "temporal_assertion_ranking" if rank_group else "temporal_assertion",
        "subject": entity_record(subj),
        "relation": f"icews:R{rel}",
        "object": entity_record(obj),
        "context": context_record(start_raw, end_raw, split),
        "evidence": evidence_record(
            split,
            row_index,
            subj,
            rel,
            obj,
            start_raw,
            end_raw,
            evidence_weight,
        ),
        "confidence": round(confidence, 6),
        "polarity": polarity,
        "is_gold": is_gold,
        "raw_subject_id": subj,
        "raw_relation_id": rel,
        "raw_object_id": obj,
        "raw_time_start": start_raw,
        "raw_time_end": end_raw,
    }
    if rank_group is not None:
        record["rank_group"] = rank_group
        record["candidate_label"] = "gold" if is_gold else f"negative_{corruption}"
        record["gold_label"] = "gold"
    if corruption is not None:
        record["corruption"] = corruption
    if split == "test":
        record["rank_group"] = rank_group or f"icews14:{split}:{row_index}"
        if "candidate_label" not in record:
            record["candidate_label"] = "gold" if is_gold else "negative"
        if "gold_label" not in record:
            record["gold_label"] = "gold"
    return record


def collect_entities(data_root: Path, split_limits: Dict[str, Optional[int]]) -> List[str]:
    entities: Set[str] = set()
    for split, limit in split_limits.items():
        split_path = data_root / "ICEWS14" / f"{split}.txt"
        for _, subj, _, obj, _, _ in iter_split_rows(split_path, limit):
            entities.add(subj)
            entities.add(obj)
    return sorted(entities, key=lambda value: int(value))


def choose_corruption(
    rng: random.Random,
    entities: Sequence[str],
    subj: str,
    obj: str,
    existing_facts: Set[Tuple[str, str, str, str, str]],
    rel: str,
    start_raw: str,
    end_raw: str,
    corrupt_subject: bool,
) -> str:
    for _ in range(100):
        candidate = rng.choice(entities)
        if corrupt_subject:
            if candidate != subj and (candidate, rel, obj, start_raw, end_raw) not in existing_facts:
                return candidate
        elif candidate != obj and (subj, rel, candidate, start_raw, end_raw) not in existing_facts:
            return candidate
    return rng.choice(entities)


def load_real_icews14(
    data_root: Path,
    train_limit: Optional[int],
    test_limit: Optional[int],
    negative_candidates: int,
    seed: int,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    rng = random.Random(seed)
    icews_root = data_root / "ICEWS14"
    split_limits = {"train": train_limit, "valid": None, "test": test_limit}
    entities = collect_entities(data_root, split_limits)
    if not entities:
        raise FileNotFoundError(f"No ICEWS14 entities found under {icews_root}")

    records: List[Dict[str, Any]] = []
    missing_splits: List[str] = []
    existing_facts: Set[Tuple[str, str, str, str, str]] = set()

    for split, limit in split_limits.items():
        split_path = icews_root / f"{split}.txt"
        if not split_path.exists():
            missing_splits.append(split)
            continue
        for row_index, subj, rel, obj, start_raw, end_raw in iter_split_rows(split_path, limit):
            existing_facts.add((subj, rel, obj, start_raw, end_raw))
            if split == "test":
                rank_group = f"icews14:{split}:{row_index}"
                records.append(
                    make_assertion(
                        split,
                        row_index,
                        subj,
                        rel,
                        obj,
                        start_raw,
                        end_raw,
                        confidence=0.72,
                        evidence_weight=0.96,
                        polarity=True,
                        rank_group=rank_group,
                        is_gold=True,
                    )
                )
                for neg_idx in range(negative_candidates):
                    corrupt_subject = neg_idx % 2 == 0
                    if corrupt_subject:
                        corrupted_subj = choose_corruption(
                            rng,
                            entities,
                            subj,
                            obj,
                            existing_facts,
                            rel,
                            start_raw,
                            end_raw,
                            True,
                        )
                        corrupted_obj = obj
                        corruption = "subject"
                    else:
                        corrupted_subj = subj
                        corrupted_obj = choose_corruption(
                            rng,
                            entities,
                            subj,
                            obj,
                            existing_facts,
                            rel,
                            start_raw,
                            end_raw,
                            False,
                        )
                        corruption = "object"
                    records.append(
                        make_assertion(
                            split,
                            row_index,
                            corrupted_subj,
                            rel,
                            corrupted_obj,
                            start_raw,
                            end_raw,
                            confidence=0.76 if neg_idx == 0 else 0.60,
                            evidence_weight=0.34 + (neg_idx % 3) * 0.04,
                            polarity=True,
                            assertion_suffix=f"-neg{neg_idx}",
                            rank_group=rank_group,
                            is_gold=False,
                            corruption=corruption,
                        )
                    )
            else:
                records.append(
                    make_assertion(
                        split,
                        row_index,
                        subj,
                        rel,
                        obj,
                        start_raw,
                        end_raw,
                        confidence=0.82,
                        evidence_weight=0.90,
                        polarity=True,
                    )
                )

    summary = {
        "dataset": "ICEWS14",
        "records": len(records),
        "entities": len(entities),
        "seed": seed,
        "negative_candidates_per_test_fact": negative_candidates,
        "missing_optional_splits": missing_splits,
        "train_limit": train_limit,
        "test_limit": test_limit,
    }
    return records, summary


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
    parser = argparse.ArgumentParser(description="Prepare real ICEWS14 temporal assertions.")
    parser.add_argument("--data-root", default="../data", type=Path)
    parser.add_argument("--output-dir", default="data/processed/icews14_real", type=Path)
    parser.add_argument("--train-limit", type=int, default=5000)
    parser.add_argument("--test-limit", type=int, default=1000)
    parser.add_argument("--negative-candidates", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    records, summary = load_real_icews14(
        data_root=args.data_root,
        train_limit=parse_optional_limit(args.train_limit),
        test_limit=parse_optional_limit(args.test_limit),
        negative_candidates=args.negative_candidates,
        seed=args.seed,
    )
    out_path = args.output_dir / "assertions.jsonl"
    count = write_jsonl(out_path, records)
    summary["output"] = str(out_path)
    summary["records"] = count
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
