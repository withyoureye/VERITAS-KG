from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Set

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from fuse_fever_kg_qwen import evaluate, load_kg_rankings, load_qwen  # noqa: E402


def stable_score(value: str, seed: int) -> float:
    digest = hashlib.sha256(f"{seed}:{value}".encode("utf-8")).hexdigest()
    return int(digest[:16], 16) / float(16**16 - 1)


def subset_rows(rows: Dict[str, Dict[str, Any]], selected: Set[str]) -> Dict[str, Dict[str, Any]]:
    return {key: value for key, value in rows.items() if key in selected}


def label_distribution(qwen_rows: Dict[str, Dict[str, Any]], group_ids: Iterable[str]) -> Dict[str, int]:
    counts: Counter[str] = Counter()
    for group_id in group_ids:
        row = qwen_rows.get(group_id)
        if row is not None:
            counts[str(row.get("gold_label"))] += 1
    return dict(sorted(counts.items()))


def strip_details(row: Dict[str, Any]) -> Dict[str, Any]:
    return {key: value for key, value in row.items() if key != "details"}


def write_summary(path: Path, payload: Dict[str, Any]) -> None:
    calibration = payload["calibration"]
    evaluation = payload["evaluation"]
    selected = payload["selected_alpha"]
    lines = [
        "# FEVER KG/Qwen Calibrated Fusion",
        "",
        f"- Split seed: `{payload['split_seed']}`",
        f"- Calibration ratio: `{payload['calibration_ratio']}`",
        f"- Calibration groups: `{payload['calibration_groups']}`",
        f"- Evaluation groups: `{payload['evaluation_groups']}`",
        f"- Selected alpha on calibration: `{selected}`",
        "",
        "## Calibration Alpha Sweep",
        "",
        "| alpha | accuracy | macro_f1 |",
        "|---:|---:|---:|",
    ]
    for row in calibration:
        lines.append(f"| {row['alpha']:.2f} | {row['accuracy']:.4f} | {row['macro_f1']:.4f} |")
    lines.extend(
        [
            "",
            "## Frozen Evaluation",
            "",
            "| alpha | evaluated | accuracy | macro_f1 |",
            "|---:|---:|---:|---:|",
            f"| {evaluation['alpha']:.2f} | {evaluation['evaluated']} | {evaluation['accuracy']:.4f} | {evaluation['macro_f1']:.4f} |",
            "",
            "## Label Distribution",
            "",
            f"- Calibration: `{payload['calibration_label_distribution']}`",
            f"- Evaluation: `{payload['evaluation_label_distribution']}`",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Calibrated FEVER KG/Qwen fusion with held-out evaluation.")
    parser.add_argument("--kg-rankings", required=True, type=Path)
    parser.add_argument("--qwen", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--alpha", action="append", type=float, default=[])
    parser.add_argument("--qwen-boost", type=float, default=1.0)
    parser.add_argument("--calibration-ratio", type=float, default=0.2)
    parser.add_argument("--split-seed", type=int, default=13)
    parser.add_argument("--details-limit", type=int, default=-1)
    args = parser.parse_args()

    kg_rankings = load_kg_rankings(args.kg_rankings)
    qwen_rows = load_qwen(args.qwen)
    group_ids = sorted(set(kg_rankings) & set(qwen_rows))
    calibration_ids = {
        group_id
        for group_id in group_ids
        if stable_score(group_id, args.split_seed) < args.calibration_ratio
    }
    evaluation_ids = set(group_ids) - calibration_ids
    alphas = args.alpha or [0.0, 0.25, 0.5, 0.75, 0.9, 1.0]

    calibration_rows = [
        evaluate(
            kg_rankings,
            subset_rows(qwen_rows, calibration_ids),
            alpha,
            args.qwen_boost,
            details_limit=0,
        )
        for alpha in alphas
    ]
    best_calibration = max(calibration_rows, key=lambda row: (row["accuracy"], row["macro_f1"]))
    selected_alpha = float(best_calibration["alpha"])
    evaluation = evaluate(
        kg_rankings,
        subset_rows(qwen_rows, evaluation_ids),
        selected_alpha,
        args.qwen_boost,
        details_limit=args.details_limit,
    )
    payload = {
        "baseline": "fever_kg_qwen_calibrated_fusion",
        "split_seed": args.split_seed,
        "calibration_ratio": args.calibration_ratio,
        "calibration_groups": len(calibration_ids),
        "evaluation_groups": len(evaluation_ids),
        "calibration_label_distribution": label_distribution(qwen_rows, calibration_ids),
        "evaluation_label_distribution": label_distribution(qwen_rows, evaluation_ids),
        "selected_alpha": selected_alpha,
        "calibration": [strip_details(row) for row in calibration_rows],
        "evaluation": evaluation,
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "fusion_calibrated.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    write_summary(args.output / "summary.md", payload)
    print((args.output / "summary.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
