from __future__ import annotations

import argparse
import csv
import fcntl
import json
import math
import random
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evidence_scorer import apply_auto_evidence_scores
from ontology_kg import Assertion, OntologyKnowledgeGraph, build_graph_from_records, context_from_dict


VARIANTS = {
    "full": {
        "enforce_ontology": True,
        "use_context": True,
        "use_evidence_weight": True,
    },
    "no_ontology": {
        "enforce_ontology": False,
        "use_context": True,
        "use_evidence_weight": True,
    },
    "no_context": {
        "enforce_ontology": True,
        "use_context": False,
        "use_evidence_weight": True,
    },
    "no_evidence": {
        "enforce_ontology": True,
        "use_context": True,
        "use_evidence_weight": False,
    },
}

METRIC_FIELDS = [
    "attempted_assertions",
    "imported_assertions",
    "rejected_assertions",
    "conflict_count",
    "conflict_rate",
    "invalid_imported_assertions",
    "invalid_assertion_rate",
    "evidence_coverage",
    "traceability_rate",
    "ranking_groups",
    "top1_support_accuracy",
    "fact_verification_accuracy",
    "fact_verification_macro_f1",
]


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, records: Iterable[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


def assertion_id(assertion: Assertion) -> str:
    metadata = assertion.metadata or {}
    return str(metadata.get("assertion_id", "|".join(assertion.key())))


def invalid_imported_count(kg: OntologyKnowledgeGraph) -> int:
    return sum(1 for assertion in kg.assertions if kg.validation_error(assertion) is not None)


def evidence_coverage(kg: OntologyKnowledgeGraph) -> float:
    if not kg.assertions:
        return 0.0
    covered = 0
    for assertion in kg.assertions:
        if assertion.evidence.source_id and assertion.evidence.snippet:
            covered += 1
    return covered / len(kg.assertions)


def rank_groups(kg: OntologyKnowledgeGraph) -> Dict[str, List[Assertion]]:
    groups: Dict[str, List[Assertion]] = defaultdict(list)
    for assertion in kg.assertions:
        metadata = assertion.metadata or {}
        group = metadata.get("rank_group")
        if group:
            groups[str(group)].append(assertion)
    return groups


def ranking_metrics(kg: OntologyKnowledgeGraph) -> Tuple[Dict[str, float], List[Dict[str, Any]]]:
    groups = rank_groups(kg)
    evaluated = 0
    correct = 0
    details: List[Dict[str, Any]] = []
    for group_id, assertions in sorted(groups.items()):
        gold = [a for a in assertions if (a.metadata or {}).get("is_gold") is True]
        if not gold:
            continue
        query_context = (gold[0].metadata or {}).get("query_context")
        candidates = assertions
        if kg.use_context and isinstance(query_context, dict):
            query = context_from_dict(query_context)
            filtered = [a for a in assertions if a.context.overlaps(query)]
            if filtered:
                candidates = filtered
        ranked = kg.rank_assertions(candidates)
        top = ranked[0]
        evaluated += 1
        is_correct = (top.metadata or {}).get("is_gold") is True
        correct += int(is_correct)
        details.append(
            {
                "rank_group": group_id,
                "top_assertion_id": assertion_id(top),
                "top_label": (top.metadata or {}).get("candidate_label"),
                "gold_label": (gold[0].metadata or {}).get("gold_label"),
                "correct": is_correct,
                "top_score": round(top.support_score(kg.use_evidence_weight), 6),
                "candidates": [
                    {
                        "assertion_id": assertion_id(a),
                        "label": (a.metadata or {}).get("candidate_label"),
                        "is_gold": (a.metadata or {}).get("is_gold"),
                        "confidence": a.confidence,
                        "evidence_weight": a.evidence.weight,
                        "score": round(a.support_score(kg.use_evidence_weight), 6),
                    }
                    for a in ranked
                ],
            }
        )
    accuracy = correct / evaluated if evaluated else 0.0
    return {"ranking_groups": evaluated, "top1_support_accuracy": accuracy}, details


def macro_f1(rows: List[Dict[str, Any]]) -> float:
    labels = sorted({row["gold_label"] for row in rows} | {row["top_label"] for row in rows})
    if not labels:
        return 0.0
    scores: List[float] = []
    for label in labels:
        tp = sum(1 for row in rows if row["gold_label"] == label and row["top_label"] == label)
        fp = sum(1 for row in rows if row["gold_label"] != label and row["top_label"] == label)
        fn = sum(1 for row in rows if row["gold_label"] == label and row["top_label"] != label)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        if precision + recall == 0:
            scores.append(0.0)
        else:
            scores.append(2 * precision * recall / (precision + recall))
    return sum(scores) / len(scores)


def fact_verification_metrics(ranking_details: List[Dict[str, Any]]) -> Dict[str, float]:
    rows = [
        row
        for row in ranking_details
        if row.get("gold_label") is not None and row.get("top_label") is not None
    ]
    if not rows:
        return {"fact_verification_accuracy": 0.0, "fact_verification_macro_f1": 0.0}
    accuracy = sum(1 for row in rows if row["correct"]) / len(rows)
    return {
        "fact_verification_accuracy": accuracy,
        "fact_verification_macro_f1": macro_f1(rows),
    }


def conflict_details(
    kg: OntologyKnowledgeGraph,
    conflicts: List[Tuple[Assertion, Assertion]],
    max_items: int,
) -> List[Dict[str, Any]]:
    details: List[Dict[str, Any]] = []
    for left, right in conflicts[:max_items]:
        details.append(
            {
                "left_id": assertion_id(left),
                "right_id": assertion_id(right),
                "same_key": left.key(),
                "left_context": left.context.to_dict(),
                "right_context": right.context.to_dict(),
                "left_evidence": left.evidence.to_dict(),
                "right_evidence": right.evidence.to_dict(),
                "left_explanation": kg.explain(left),
                "right_explanation": kg.explain(right),
            }
        )
    return details


def seed_records(records: Sequence[Dict[str, Any]], seed: int, jitter: float) -> List[Dict[str, Any]]:
    rng = random.Random(seed)
    seeded: List[Dict[str, Any]] = []
    for record in records:
        copied = json.loads(json.dumps(record))
        copied["seed"] = seed
        copied["confidence"] = clamp_score(
            float(copied.get("confidence", 1.0)) + rng.uniform(-jitter, jitter)
        )
        evidence = copied.get("evidence", {})
        evidence["weight"] = round(
            clamp_score(float(evidence.get("weight", 1.0)) + rng.uniform(-jitter, jitter)),
            6,
        )
        copied["evidence"] = evidence
        seeded.append(copied)
    return seeded


def scorer_records(records: Sequence[Dict[str, Any]], scorer: str) -> List[Dict[str, Any]]:
    if scorer == "auto":
        return apply_auto_evidence_scores(records)
    return [json.loads(json.dumps(record)) for record in records]


def clamp_score(value: float) -> float:
    return max(0.0, min(1.0, value))


def evaluate_variant(
    records: List[Dict[str, Any]],
    variant_name: str,
    config: Dict[str, bool],
    output_dir: Path,
    max_conflict_details: int,
) -> Dict[str, Any]:
    kg, report = build_graph_from_records(records, **config)
    conflicts = kg.detect_conflicts()
    invalid_count = invalid_imported_count(kg)
    ranking, ranking_detail = ranking_metrics(kg)
    fact_metrics = fact_verification_metrics(ranking_detail)

    metrics: Dict[str, Any] = {
        "variant": variant_name,
        "attempted_assertions": report.attempted,
        "imported_assertions": report.imported,
        "rejected_assertions": report.rejected,
        "rejection_reasons": report.rejection_reasons,
        "conflict_count": len(conflicts),
        "conflict_rate": len(conflicts) / report.imported if report.imported else 0.0,
        "invalid_imported_assertions": invalid_count,
        "invalid_assertion_rate": invalid_count / report.imported if report.imported else 0.0,
        "evidence_coverage": evidence_coverage(kg),
        "traceability_rate": evidence_coverage(kg),
    }
    metrics.update(ranking)
    metrics.update(fact_metrics)

    variant_dir = output_dir / variant_name
    variant_dir.mkdir(parents=True, exist_ok=True)
    write_json(variant_dir / "metrics.json", metrics)
    write_jsonl(
        variant_dir / "conflicts.jsonl",
        conflict_details(kg, conflicts, max_conflict_details),
    )
    write_jsonl(variant_dir / "rankings.jsonl", ranking_detail)
    return metrics


def write_ablation_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    fields = ["variant"] + METRIC_FIELDS
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def metric_bar(value: float, width: int = 24) -> str:
    filled = int(round(max(0.0, min(1.0, value)) * width))
    return "#" * filled + "." * (width - filled)


def write_summary(path: Path, run_name: str, input_path: Path, rows: List[Dict[str, Any]]) -> None:
    lines = [
        f"# Experiment Summary: {run_name}",
        "",
        f"- Input: `{input_path}`",
        f"- Variants: {', '.join(row['variant'] for row in rows)}",
        "",
        "| variant | imported | rejected | conflict_rate | invalid_rate | evidence_coverage | top1_acc | fact_acc | fact_f1 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {variant} | {imported_assertions} | {rejected_assertions} | "
            "{conflict_rate:.4f} | {invalid_assertion_rate:.4f} | "
            "{evidence_coverage:.4f} | {top1_support_accuracy:.4f} | "
            "{fact_verification_accuracy:.4f} | {fact_verification_macro_f1:.4f} |".format(
                **row
            )
        )
    lines.extend(["", "## Top-1 Accuracy Chart", ""])
    for row in rows:
        lines.append(
            f"- `{row['variant']}` {metric_bar(float(row['top1_support_accuracy']))} "
            f"{row['top1_support_accuracy']:.4f}"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def detect_task(records: Sequence[Dict[str, Any]]) -> str:
    tasks = {str(record.get("task", "")) for record in records}
    if "fact_verification" in tasks:
        return "fact_verification"
    if "temporal_assertion_ranking" in tasks:
        return "temporal_assertion_ranking"
    return "mixed"


def mean_std(values: Sequence[float]) -> Tuple[float, float]:
    if not values:
        return 0.0, 0.0
    if len(values) == 1:
        return float(values[0]), 0.0
    return float(statistics.mean(values)), float(statistics.stdev(values))


def aggregate_seed_runs(seed_rows: Sequence[Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
    grouped: Dict[str, Dict[str, List[float]]] = {}
    for row in seed_rows:
        variant = str(row["variant"])
        grouped.setdefault(variant, {field: [] for field in METRIC_FIELDS})
        for field in METRIC_FIELDS:
            value = row.get(field)
            if isinstance(value, (int, float)):
                grouped[variant][field].append(float(value))
    aggregate: Dict[str, Dict[str, float]] = {}
    for variant, field_values in grouped.items():
        aggregate[variant] = {}
        for field, values in field_values.items():
            mean_value, std_value = mean_std(values)
            aggregate[variant][f"{field}_mean"] = mean_value
            aggregate[variant][f"{field}_std"] = std_value
    return aggregate


def write_summary_csv(path: Path, run_name: str, aggregated: Dict[str, Dict[str, float]]) -> None:
    fields = ["run_name", "variant"]
    for metric in [
        "conflict_rate",
        "invalid_assertion_rate",
        "evidence_coverage",
        "traceability_rate",
        "top1_support_accuracy",
        "fact_verification_accuracy",
        "fact_verification_macro_f1",
    ]:
        fields.extend([f"{metric}_mean", f"{metric}_std"])
    path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = path.with_suffix(path.suffix + ".lock")
    with lock_path.open("w", encoding="utf-8") as lock_handle:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
        existing_rows: List[Dict[str, Any]] = []
        if path.exists():
            with path.open("r", newline="", encoding="utf-8") as handle:
                reader = csv.DictReader(handle)
                existing_rows = [
                    row
                    for row in reader
                    if row.get("run_name") != run_name
                ]
        new_rows: List[Dict[str, Any]] = []
        for variant, metrics in sorted(aggregated.items()):
            row = {"run_name": run_name, "variant": variant}
            row.update({field: metrics.get(field, "") for field in fields if field not in row})
            new_rows.append(row)
        temp_path = path.with_suffix(path.suffix + ".tmp")
        with temp_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for row in existing_rows + new_rows:
                writer.writerow({field: row.get(field, "") for field in fields})
        temp_path.replace(path)
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)


def write_ablation_svg(path: Path, aggregated: Dict[str, Dict[str, float]]) -> None:
    variants = list(sorted(aggregated))
    if not variants:
        path.write_text("", encoding="utf-8")
        return
    width = 760
    height = 280
    bar_area = width - 160
    row_height = 48
    svg_lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#fbfaf6"/>',
        '<text x="24" y="28" font-family="monospace" font-size="18" fill="#222">Top-1 Support Accuracy by Variant</text>',
    ]
    max_value = max(
        aggregated[variant].get("top1_support_accuracy_mean", 0.0) for variant in variants
    )
    scale = bar_area / max(max_value, 1.0)
    for idx, variant in enumerate(variants):
        y = 60 + idx * row_height
        mean_value = aggregated[variant].get("top1_support_accuracy_mean", 0.0)
        std_value = aggregated[variant].get("top1_support_accuracy_std", 0.0)
        bar_width = mean_value * scale
        svg_lines.append(
            f'<text x="24" y="{y + 18}" font-family="monospace" font-size="14" fill="#333">{variant}</text>'
        )
        svg_lines.append(
            f'<rect x="180" y="{y}" width="{bar_width:.2f}" height="22" rx="3" fill="#3f7d58"/>'
        )
        svg_lines.append(
            f'<text x="{190 + bar_width:.2f}" y="{y + 16}" font-family="monospace" font-size="12" fill="#333">{mean_value:.4f} ± {std_value:.4f}</text>'
        )
    svg_lines.append("</svg>")
    path.write_text("\n".join(svg_lines) + "\n", encoding="utf-8")


def write_seed_summary(path: Path, run_name: str, seeds: Sequence[int], aggregated: Dict[str, Dict[str, float]]) -> None:
    lines = [
        f"# Seed Summary: {run_name}",
        "",
        f"- Seeds: {', '.join(str(seed) for seed in seeds)}",
        "",
        "| variant | conflict_rate(mean±std) | invalid_rate(mean±std) | top1_acc(mean±std) | fact_acc(mean±std) |",
        "|---|---:|---:|---:|---:|",
    ]
    for variant, metrics in sorted(aggregated.items()):
        lines.append(
            "| {variant} | {conflict_rate_mean:.4f} ± {conflict_rate_std:.4f} | "
            "{invalid_assertion_rate_mean:.4f} ± {invalid_assertion_rate_std:.4f} | "
            "{top1_support_accuracy_mean:.4f} ± {top1_support_accuracy_std:.4f} | "
            "{fact_verification_accuracy_mean:.4f} ± {fact_verification_accuracy_std:.4f} |".format(
                variant=variant,
                **metrics,
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_seeds(seed_argument: Sequence[int], num_seeds: int) -> List[int]:
    if seed_argument:
        return list(seed_argument)
    return list(range(num_seeds))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run ontology KG ablation experiments.")
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--run-name", default="experiment")
    parser.add_argument("--max-conflict-details", type=int, default=50)
    parser.add_argument("--seed", action="append", type=int, default=[])
    parser.add_argument("--num-seeds", type=int, default=1)
    parser.add_argument("--seed-jitter", type=float, default=0.03)
    parser.add_argument("--evidence-scorer", choices=["manual", "auto"], default="manual")
    args = parser.parse_args()

    base_records = load_jsonl(args.input)
    args.output.mkdir(parents=True, exist_ok=True)
    seeds = parse_seeds(args.seed, args.num_seeds)
    task_name = detect_task(base_records)
    base_records = scorer_records(base_records, args.evidence_scorer)

    all_seed_rows: List[Dict[str, Any]] = []
    last_metrics_rows: List[Dict[str, Any]] = []
    for seed in seeds:
        seed_dir = args.output / f"seed_{seed}"
        seed_dir.mkdir(parents=True, exist_ok=True)
        records = seed_records(base_records, seed, args.seed_jitter)
        metrics_rows = [
            evaluate_variant(
                records,
                variant_name,
                config,
                seed_dir,
                args.max_conflict_details,
            )
            for variant_name, config in VARIANTS.items()
        ]
        for row in metrics_rows:
            row["seed"] = seed
            all_seed_rows.append(row)
        write_json(seed_dir / "metrics.json", {"run_name": args.run_name, "seed": seed, "variants": metrics_rows})
        write_ablation_csv(seed_dir / "ablation.csv", metrics_rows)
        write_summary(seed_dir / "summary.md", f"{args.run_name}/seed_{seed}", args.input, metrics_rows)
        last_metrics_rows = metrics_rows

    aggregated = aggregate_seed_runs(all_seed_rows)
    summary_payload = {
        "run_name": args.run_name,
        "task": task_name,
        "seeds": seeds,
        "evidence_scorer": args.evidence_scorer,
        "aggregated": aggregated,
        "last_seed_variants": last_metrics_rows,
    }
    write_json(args.output / "metrics.json", summary_payload)
    write_jsonl(args.output / "seed_metrics.jsonl", all_seed_rows)
    write_summary_csv(ROOT / "results" / "summary.csv", args.run_name, aggregated)
    write_seed_summary(args.output / "summary.md", args.run_name, seeds, aggregated)
    write_ablation_svg(args.output / "ablation_bar.svg", aggregated)

    print((args.output / "summary.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
