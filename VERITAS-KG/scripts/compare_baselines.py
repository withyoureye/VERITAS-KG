from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Sequence


def load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def append_row(rows: List[Dict[str, Any]], dataset: str, baseline: str, metrics: Dict[str, Any]) -> None:
    row = {"dataset": dataset, "baseline": baseline}
    row.update(metrics)
    rows.append(row)


def summary_md(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    lines = [
        "# Baseline Comparison",
        "",
        "| dataset | baseline | evaluated | accuracy | macro_f1 | invalid_rate | top1_acc | fact_acc |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {dataset} | `{baseline}` | {evaluated} | {accuracy} | {macro_f1} | {invalid_rate} | {top1} | {fact} |".format(
                dataset=row.get("dataset", ""),
                baseline=row.get("baseline", ""),
                evaluated=row.get("evaluated", ""),
                accuracy=f"{row['accuracy']:.4f}" if "accuracy" in row else "",
                macro_f1=f"{row['macro_f1']:.4f}" if "macro_f1" in row else "",
                invalid_rate=f"{row['invalid_rate']:.4f}" if "invalid_rate" in row else "",
                top1=f"{row['top1_support_accuracy_mean']:.4f}" if "top1_support_accuracy_mean" in row else "",
                fact=f"{row['fact_verification_accuracy_mean']:.4f}" if "fact_verification_accuracy_mean" in row else "",
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def csv_out(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    fields = [
        "dataset",
        "baseline",
        "evaluated",
        "accuracy",
        "macro_f1",
        "invalid_rate",
        "top1_support_accuracy_mean",
        "fact_verification_accuracy_mean",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare baseline and main results.")
    parser.add_argument("--main-results", required=True, type=Path)
    parser.add_argument("--baseline-results", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    rows: List[Dict[str, Any]] = []

    main_payload = load_json(args.main_results)
    for variant, metrics in sorted(main_payload.get("aggregated", {}).items()):
        append_row(
            rows,
            main_payload.get("run_name", "main"),
            f"main/{variant}",
            {
                "evaluated": metrics.get("ranking_groups_mean", ""),
                "top1_support_accuracy_mean": metrics.get("top1_support_accuracy_mean", 0.0),
                "fact_verification_accuracy_mean": metrics.get("fact_verification_accuracy_mean", 0.0),
                "invalid_rate": metrics.get("invalid_assertion_rate_mean", 0.0),
            },
        )
    for file_path in sorted(args.baseline_results.glob("*.json")):
        if file_path.name in {"baselines.json", "qwen_fever_baseline.json"}:
            continue
        payload = load_json(file_path)
        dataset = payload.get("dataset", file_path.stem)
        baseline = payload.get("baseline", file_path.stem)
        metrics: Dict[str, Any] = {
            "evaluated": payload.get("evaluated", 0),
            "accuracy": payload.get("accuracy", 0.0),
            "macro_f1": payload.get("macro_f1", 0.0),
            "invalid_rate": payload.get("invalid_rate", 0.0),
        }
        append_row(rows, dataset, baseline, metrics)

    csv_out(args.output / "comparison.csv", rows)
    summary_md(args.output / "summary.md", rows)
    (args.output / "comparison.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print((args.output / "summary.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
