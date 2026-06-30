from __future__ import annotations

import argparse
import csv
import json
import resource
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.run_experiment import VARIANTS, evaluate_variant
from scripts.run_robustness import sample_records


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def unique_count(records: Sequence[Dict[str, Any]], key: str) -> int:
    return len({str(record.get(key)) for record in records if record.get(key) is not None})


def count_evidence(records: Sequence[Dict[str, Any]]) -> int:
    return sum(
        1
        for record in records
        if record.get("evidence", {}).get("source_id") and record.get("evidence", {}).get("snippet")
    )


def peak_memory_mb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


def run_dataset(name: str, input_path: Path, output_dir: Path, groups_limit: int, seed: int) -> Dict[str, Any]:
    load_start = time.perf_counter()
    records = load_jsonl(input_path)
    load_seconds = time.perf_counter() - load_start
    sampled = sample_records(records, groups_limit, seed)

    eval_dir = output_dir / name / "eval"
    eval_dir.mkdir(parents=True, exist_ok=True)
    eval_start = time.perf_counter()
    metrics = evaluate_variant(sampled, "full", VARIANTS["full"], eval_dir, max_conflict_details=0)
    eval_seconds = time.perf_counter() - eval_start

    entities = {
        str(entity_id)
        for record in sampled
        for entity_id in [record.get("subject", {}).get("id"), record.get("object", {}).get("id")]
        if entity_id is not None
    }
    return {
        "dataset": name,
        "input": str(input_path),
        "groups_limit": groups_limit,
        "loaded_assertions": len(records),
        "evaluated_assertions": len(sampled),
        "entities": len(entities),
        "relations": unique_count(sampled, "relation"),
        "evidence_items": count_evidence(sampled),
        "load_seconds": round(load_seconds, 4),
        "evaluation_seconds": round(eval_seconds, 4),
        "peak_memory_mb": round(peak_memory_mb(), 2),
        "top1_support_accuracy": metrics.get("top1_support_accuracy", 0.0),
        "fact_verification_accuracy": metrics.get("fact_verification_accuracy", 0.0),
        "invalid_assertion_rate": metrics.get("invalid_assertion_rate", 0.0),
    }


def write_csv(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    rows = list(rows)
    fields = [
        "dataset",
        "groups_limit",
        "loaded_assertions",
        "evaluated_assertions",
        "entities",
        "relations",
        "evidence_items",
        "load_seconds",
        "evaluation_seconds",
        "peak_memory_mb",
        "top1_support_accuracy",
        "fact_verification_accuracy",
        "invalid_assertion_rate",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def write_md(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    lines = [
        "# Efficiency Report",
        "",
        "| dataset | loaded assertions | eval assertions | entities | evidence | eval sec | peak MB | main metric |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        main_metric = row.get("fact_verification_accuracy") or row.get("top1_support_accuracy") or (
            1.0 - float(row.get("invalid_assertion_rate", 0.0))
        )
        lines.append(
            "| {dataset} | {loaded_assertions} | {evaluated_assertions} | {entities} | {evidence_items} | "
            "{evaluation_seconds:.4f} | {peak_memory_mb:.2f} | {main_metric:.4f} |".format(
                **row,
                main_metric=float(main_metric),
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate lightweight efficiency statistics.")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--groups-limit", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dataset", action="append", nargs=2, metavar=("NAME", "ASSERTIONS_JSONL"), default=[])
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    rows = [run_dataset(name, Path(input_path), args.output, args.groups_limit, args.seed) for name, input_path in args.dataset]
    write_json(args.output / "efficiency.json", {"rows": rows})
    write_csv(args.output / "efficiency.csv", rows)
    write_md(args.output / "summary.md", rows)
    print((args.output / "summary.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
