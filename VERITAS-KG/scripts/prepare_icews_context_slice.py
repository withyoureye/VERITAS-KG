from __future__ import annotations

import argparse
import json
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Dict, Iterable, List


def load_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def make_candidate(base: Dict[str, Any], idx: int, label: str, is_gold: bool, time_shift_days: int) -> Dict[str, Any]:
    row = json.loads(json.dumps(base))
    row["assertion_id"] = f"context-slice-{idx}-{label}"
    row["task"] = "context_disambiguation"
    row["rank_group"] = f"icews_context:{idx}"
    row["candidate_label"] = label
    row["gold_label"] = "matched_context"
    row["is_gold"] = is_gold
    row["confidence"] = 0.8
    row["evidence"]["weight"] = 0.9 if is_gold else 0.9
    row["evidence"]["source_id"] = f"{row['evidence']['source_id']}#{label}"
    row["evidence"]["snippet"] = f"{row['evidence']['snippet']} | context candidate={label}"
    if not is_gold:
        # Keep the same entity/relation/evidence strength but move it outside
        # the query context via the condition. With context disabled, this
        # tie is deliberately resolved by insertion order in favor of the
        # distractor; with context enabled the distractor is excluded.
        context = row["context"]
        if context.get("time_start") and context.get("time_end"):
            start = date.fromisoformat(context["time_start"]) + timedelta(days=time_shift_days)
            end = date.fromisoformat(context["time_end"]) + timedelta(days=time_shift_days)
            context["time_start"] = start.isoformat()
            context["time_end"] = end.isoformat()
        context["condition"] = base["context"].get("condition")
    row["query_context"] = dict(base["context"])
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description="Create an ICEWS14 context-sensitive disambiguation slice.")
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--limit", type=int, default=5000)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows: List[Dict[str, Any]] = []
    used = 0
    for record in load_jsonl(args.input):
        if record.get("split") != "test" or not record.get("is_gold"):
            continue
        # Insert the wrong-context candidate first so no-context ranking breaks ties incorrectly.
        rows.append(make_candidate(record, used, "wrong_context", False, 30))
        rows.append(make_candidate(record, used, "matched_context", True, 0))
        used += 1
        if used >= args.limit:
            break

    output = args.output_dir / "assertions.jsonl"
    write_jsonl(output, rows)
    summary = {
        "input": str(args.input),
        "output": str(output),
        "groups": used,
        "assertions": len(rows),
        "construction": "paired same event candidates with matched vs shifted context",
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
