from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple


TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")


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


def load_jsonl(path: Path, limit: Optional[int] = None) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for idx, line in enumerate(handle):
            if limit is not None and idx >= limit:
                break
            if line.strip():
                rows.append(json.loads(line))
    return rows


def load_group_sample(
    path: Path,
    groups_limit: Optional[int],
    keep_train: bool = False,
) -> List[Dict[str, Any]]:
    if groups_limit is None or groups_limit <= 0:
        return load_jsonl(path)
    rows: List[Dict[str, Any]] = []
    selected_groups: Set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            group = record.get("rank_group")
            if keep_train and record.get("split") == "train":
                rows.append(record)
                continue
            if not group:
                continue
            group_id = str(group)
            if group_id not in selected_groups:
                if len(selected_groups) >= groups_limit:
                    continue
                selected_groups.add(group_id)
            rows.append(record)
    return rows


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def tokenize(text: str) -> List[str]:
    return [token.lower() for token in TOKEN_RE.findall(text)]


def groups(records: Sequence[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for record in records:
        group = record.get("rank_group")
        if group:
            grouped[str(group)].append(record)
    return grouped


def ranking_accuracy(predictions: Sequence[Dict[str, Any]]) -> float:
    if not predictions:
        return 0.0
    return sum(1 for row in predictions if row["correct"]) / len(predictions)


def macro_f1(predictions: Sequence[Dict[str, Any]]) -> float:
    labels = sorted({row["gold_label"] for row in predictions} | {row["predicted_label"] for row in predictions})
    if not labels:
        return 0.0
    scores: List[float] = []
    for label in labels:
        tp = sum(1 for row in predictions if row["gold_label"] == label and row["predicted_label"] == label)
        fp = sum(1 for row in predictions if row["gold_label"] != label and row["predicted_label"] == label)
        fn = sum(1 for row in predictions if row["gold_label"] == label and row["predicted_label"] != label)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        scores.append(2 * precision * recall / (precision + recall) if precision + recall else 0.0)
    return sum(scores) / len(scores)


def fever_bm25_proxy(records: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    grouped = groups(records)
    docs = []
    for candidates in grouped.values():
        snippet = str(candidates[0].get("evidence", {}).get("snippet", ""))
        docs.append(tokenize(snippet))
    df = Counter(token for doc in docs for token in set(doc))
    n_docs = max(len(docs), 1)
    predictions: List[Dict[str, Any]] = []
    for group_id, candidates in grouped.items():
        gold = next((item for item in candidates if item.get("is_gold")), None)
        if gold is None:
            continue
        claim_tokens = tokenize(str(gold.get("claim_text", gold["subject"]["label"])))
        doc_tokens = tokenize(str(gold.get("evidence", {}).get("snippet", "")))
        doc_len = max(len(doc_tokens), 1)
        tf = Counter(doc_tokens)
        bm25 = 0.0
        for token in claim_tokens:
            idf = math.log(1 + (n_docs - df.get(token, 0) + 0.5) / (df.get(token, 0) + 0.5))
            freq = tf.get(token, 0)
            bm25 += idf * (freq * 2.2) / (freq + 1.2 * (0.25 + 0.75 * doc_len / 80.0)) if freq else 0.0
        source_type = str(gold.get("evidence", {}).get("source_type", ""))
        if source_type == "claim" or "No evidence available" in str(gold.get("evidence", {}).get("snippet", "")):
            predicted = "NOT ENOUGH INFO"
        elif bm25 > 0 and " not " in str(gold.get("claim_text", "")).lower():
            predicted = "REFUTES"
        else:
            predicted = "SUPPORTS"
        predictions.append(
            {
                "rank_group": group_id,
                "predicted_label": predicted,
                "gold_label": gold.get("gold_label"),
                "correct": predicted == gold.get("gold_label"),
                "score": bm25,
            }
        )
    return {
        "baseline": "fever_bm25_evidence_proxy",
        "evaluated": len(predictions),
        "accuracy": ranking_accuracy(predictions),
        "macro_f1": macro_f1(predictions),
        "sampled_details": predictions[:50],
    }


def icews_frequency_baseline(records: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    relation_object_counts: Counter[Tuple[str, str]] = Counter()
    subject_relation_counts: Counter[Tuple[str, str]] = Counter()
    for record in records:
        if record.get("split") == "train":
            relation_object_counts[(record["relation"], record["object"]["id"])] += 1
            subject_relation_counts[(record["subject"]["id"], record["relation"])] += 1
    predictions: List[Dict[str, Any]] = []
    for group_id, candidates in groups(records).items():
        gold = next((item for item in candidates if item.get("is_gold")), None)
        if gold is None:
            continue
        def score(item: Dict[str, Any]) -> Tuple[float, float, float]:
            subj = item["subject"]["id"]
            obj = item["object"]["id"]
            rel = item["relation"]
            freq = relation_object_counts[(rel, obj)] + subject_relation_counts[(subj, rel)]
            return (float(freq), float(item.get("confidence", 0.0)), float(item.get("evidence", {}).get("weight", 0.0)))

        top = max(candidates, key=score)
        predictions.append(
            {
                "rank_group": group_id,
                "predicted_label": top.get("candidate_label"),
                "gold_label": gold.get("gold_label"),
                "correct": bool(top.get("is_gold")),
            }
        )
    return {
        "baseline": "icews_train_frequency_ranker",
        "evaluated": len(predictions),
        "accuracy": ranking_accuracy(predictions),
        "sampled_details": predictions[:50],
    }


def type_compatible(subj_type: str, obj_type: str, relation: str) -> bool:
    expected = YAGO_SIGNATURES.get(relation)
    if expected is None:
        return True
    left, right = expected
    return (left == "Entity" or subj_type == left) and (right == "Entity" or obj_type == right)


def yago_signature_baseline(records: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    evaluated = 0
    invalid = 0
    predictions: List[Dict[str, Any]] = []
    for record in records:
        if record.get("task") != "ontology_validation" or not record.get("rank_group"):
            continue
        evaluated += 1
        valid = type_compatible(record["subject"]["type"], record["object"]["type"], record["relation"])
        invalid += int(not valid)
        predictions.append(
            {
                "assertion_id": record.get("assertion_id"),
                "valid": valid,
                "is_gold": record.get("is_gold"),
            }
        )
    return {
        "baseline": "yago_relation_signature_filter",
        "evaluated": evaluated,
        "invalid_rate": invalid / evaluated if evaluated else 0.0,
        "sampled_details": predictions[:50],
    }


def write_summary(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    lines = [
        "# Stronger Baseline Summary",
        "",
        "| dataset | baseline | evaluated | accuracy | macro_f1 | invalid_rate |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {dataset} | `{baseline}` | {evaluated} | {accuracy} | {macro_f1} | {invalid_rate} |".format(
                dataset=row.get("dataset", ""),
                baseline=row.get("baseline", ""),
                evaluated=row.get("evaluated", ""),
                accuracy=f"{row['accuracy']:.4f}" if "accuracy" in row else "",
                macro_f1=f"{row['macro_f1']:.4f}" if "macro_f1" in row else "",
                invalid_rate=f"{row['invalid_rate']:.4f}" if "invalid_rate" in row else "",
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    fields = ["dataset", "baseline", "evaluated", "accuracy", "macro_f1", "invalid_rate"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def main() -> None:
    parser = argparse.ArgumentParser(description="Run stronger non-neural baseline proxies.")
    parser.add_argument("--fever-input", type=Path)
    parser.add_argument("--icews-input", type=Path)
    parser.add_argument("--yago-input", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--groups-limit", type=int, default=None)
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    rows: List[Dict[str, Any]] = []
    if args.fever_input:
        fever_records = (
            load_group_sample(args.fever_input, args.groups_limit)
            if args.groups_limit
            else load_jsonl(args.fever_input, args.limit)
        )
        result = fever_bm25_proxy(fever_records)
        result["dataset"] = "FEVER"
        rows.append(result)
        write_json(args.output / "fever_bm25_evidence_proxy.json", result)
    if args.icews_input:
        icews_records = (
            load_group_sample(args.icews_input, args.groups_limit, keep_train=True)
            if args.groups_limit
            else load_jsonl(args.icews_input, args.limit)
        )
        result = icews_frequency_baseline(icews_records)
        result["dataset"] = "ICEWS14"
        rows.append(result)
        write_json(args.output / "icews_train_frequency_ranker.json", result)
    if args.yago_input:
        yago_records = (
            load_group_sample(args.yago_input, args.groups_limit)
            if args.groups_limit
            else load_jsonl(args.yago_input, args.limit)
        )
        result = yago_signature_baseline(yago_records)
        result["dataset"] = "YAGO"
        rows.append(result)
        write_json(args.output / "yago_relation_signature_filter.json", result)

    write_json(args.output / "baselines.json", {"rows": rows})
    write_csv(args.output / "baselines.csv", rows)
    write_summary(args.output / "summary.md", rows)
    print((args.output / "summary.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
