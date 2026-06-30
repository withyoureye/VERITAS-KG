from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.run_experiment import VARIANTS, evaluate_variant


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def grouped_records(records: Sequence[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    groups: Dict[str, List[Dict[str, Any]]] = {}
    for record in records:
        group = record.get("rank_group")
        if group:
            groups.setdefault(str(group), []).append(record)
    return groups


def sample_records(records: Sequence[Dict[str, Any]], groups_limit: int, seed: int) -> List[Dict[str, Any]]:
    if groups_limit <= 0:
        return [json.loads(json.dumps(record)) for record in records]
    groups = grouped_records(records)
    if groups:
        rng = random.Random(seed)
        group_ids = sorted(groups)
        rng.shuffle(group_ids)
        selected = set(group_ids[:groups_limit])
        return [
            json.loads(json.dumps(record))
            for record in records
            if str(record.get("rank_group")) in selected
        ]
    return [json.loads(json.dumps(record)) for record in records[:groups_limit]]


def perturb_records(
    records: Sequence[Dict[str, Any]],
    mode: str,
    rate: float,
    seed: int,
) -> List[Dict[str, Any]]:
    rng = random.Random(seed)
    output: List[Dict[str, Any]] = []
    for record in records:
        copied = json.loads(json.dumps(record))
        if rng.random() >= rate:
            output.append(copied)
            continue
        if mode == "drop_evidence":
            copied["evidence"] = {
                "source_id": "",
                "source_type": "dropped",
                "snippet": "",
                "weight": 0.0,
            }
        elif mode == "shuffle_evidence_weight":
            copied.setdefault("evidence", {})["weight"] = round(rng.random(), 6)
        elif mode == "drop_context":
            copied["context"] = {
                "domain": copied.get("context", {}).get("domain", "dropped_context"),
                "time_start": None,
                "time_end": None,
                "location": None,
                "condition": None,
            }
        elif mode == "type_noise":
            if copied.get("subject"):
                copied["subject"]["type"] = rng.choice(["Location", "Organization", "Person", "Entity"])
            if copied.get("object"):
                copied["object"]["type"] = rng.choice(["Location", "Organization", "Person", "Entity"])
        elif mode == "confidence_noise":
            copied["confidence"] = round(rng.random(), 6)
        else:
            raise ValueError(f"Unknown perturbation mode: {mode}")
        output.append(copied)
    return output


def run_condition(
    records: List[Dict[str, Any]],
    output_dir: Path,
    name: str,
    max_conflict_details: int,
) -> List[Dict[str, Any]]:
    condition_dir = output_dir / name
    condition_dir.mkdir(parents=True, exist_ok=True)
    rows: List[Dict[str, Any]] = []
    for variant_name, config in VARIANTS.items():
        metrics = evaluate_variant(
            records,
            variant_name,
            config,
            condition_dir,
            max_conflict_details=max_conflict_details,
        )
        rows.append(metrics)
    return rows


def write_csv(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    fields = [
        "dataset",
        "mode",
        "rate",
        "variant",
        "attempted_assertions",
        "imported_assertions",
        "rejected_assertions",
        "conflict_rate",
        "invalid_assertion_rate",
        "evidence_coverage",
        "traceability_rate",
        "top1_support_accuracy",
        "fact_verification_accuracy",
        "fact_verification_macro_f1",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def write_markdown(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    lines = [
        "# Robustness Results",
        "",
        "| dataset | mode | rate | variant | invalid_rate | evidence_coverage | top1_acc | fact_acc |",
        "|---|---|---:|---|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {dataset} | {mode} | {rate:.2f} | `{variant}` | {invalid_assertion_rate:.4f} | "
            "{evidence_coverage:.4f} | {top1_support_accuracy:.4f} | "
            "{fact_verification_accuracy:.4f} |".format(**row)
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run robustness perturbation experiments.")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--groups-limit", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--mode",
        action="append",
        default=[],
        choices=["drop_evidence", "shuffle_evidence_weight", "drop_context", "type_noise", "confidence_noise"],
    )
    parser.add_argument("--rate", action="append", type=float, default=[])
    parser.add_argument("--max-conflict-details", type=int, default=10)
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    base = sample_records(load_jsonl(args.input), args.groups_limit, args.seed)
    modes = args.mode or ["drop_evidence", "shuffle_evidence_weight", "drop_context", "type_noise"]
    rates = args.rate or [0.0, 0.1, 0.3, 0.5]

    all_rows: List[Dict[str, Any]] = []
    for mode in modes:
        for rate in rates:
            condition = f"{mode}_r{rate:.2f}".replace(".", "p")
            perturbed = perturb_records(base, mode, rate, args.seed)
            rows = run_condition(perturbed, args.output, condition, args.max_conflict_details)
            for row in rows:
                row["dataset"] = args.dataset
                row["mode"] = mode
                row["rate"] = rate
                all_rows.append(row)

    write_json(
        args.output / "robustness.json",
        {
            "dataset": args.dataset,
            "input": str(args.input),
            "groups_limit": args.groups_limit,
            "seed": args.seed,
            "rows": all_rows,
        },
    )
    write_csv(args.output / "robustness.csv", all_rows)
    write_markdown(args.output / "summary.md", all_rows)
    print((args.output / "summary.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
