from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List


def load_qwen(path: Path) -> Dict[str, Dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {str(row["rank_group"]): row for row in payload.get("details", [])}


def load_fusion(path: Path) -> Dict[str, Dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if "best" in payload:
        details = payload.get("best", {}).get("details", [])
    else:
        details = payload.get("evaluation", {}).get("details", [])
    return {str(row["rank_group"]): row for row in details}


def percentile(values: List[float], q: float) -> float:
    values = sorted(values)
    if not values:
        return 0.0
    idx = min(len(values) - 1, max(0, int(round((len(values) - 1) * q))))
    return values[idx]


def main() -> None:
    parser = argparse.ArgumentParser(description="Paired bootstrap significance test.")
    parser.add_argument("--qwen", required=True, type=Path)
    parser.add_argument("--fusion", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--samples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    qwen = load_qwen(args.qwen)
    fusion = load_fusion(args.fusion)
    group_ids = sorted(set(qwen) & set(fusion))
    rows = []
    for group_id in group_ids:
        q = qwen[group_id]
        f = fusion[group_id]
        rows.append(
            {
                "rank_group": group_id,
                "qwen_correct": bool(q.get("correct")),
                "fusion_correct": bool(f.get("correct")),
            }
        )
    if not rows:
        raise ValueError("No overlapping paired rows found.")

    qwen_acc = sum(row["qwen_correct"] for row in rows) / len(rows)
    fusion_acc = sum(row["fusion_correct"] for row in rows) / len(rows)
    observed_delta = fusion_acc - qwen_acc

    rng = random.Random(args.seed)
    deltas: List[float] = []
    non_positive = 0
    n = len(rows)
    for _ in range(args.samples):
        sample = [rows[rng.randrange(n)] for _ in range(n)]
        q_acc = sum(row["qwen_correct"] for row in sample) / n
        f_acc = sum(row["fusion_correct"] for row in sample) / n
        delta = f_acc - q_acc
        deltas.append(delta)
        if delta <= 0:
            non_positive += 1
    p_value = (non_positive + 1) / (args.samples + 1)

    payload = {
        "paired_items": n,
        "qwen_accuracy": qwen_acc,
        "fusion_accuracy": fusion_acc,
        "observed_delta": observed_delta,
        "bootstrap_samples": args.samples,
        "delta_mean": mean(deltas),
        "delta_ci95": [percentile(deltas, 0.025), percentile(deltas, 0.975)],
        "p_value_one_sided": p_value,
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "significance.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# Paired Bootstrap Significance",
        "",
        f"- Paired items: `{n}`",
        f"- Qwen accuracy: `{qwen_acc:.4f}`",
        f"- Fusion accuracy: `{fusion_acc:.4f}`",
        f"- Delta: `{observed_delta:.4f}`",
        f"- 95% bootstrap CI for delta: `[{payload['delta_ci95'][0]:.4f}, {payload['delta_ci95'][1]:.4f}]`",
        f"- One-sided bootstrap p-value: `{p_value:.6f}`",
    ]
    (args.output / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print((args.output / "summary.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
