from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from run_fever_wiki_retrieval_reader import (  # noqa: E402
    build_doc_stats,
    classify,
    load_sentence_corpus,
    macro_f1,
    retrieve,
)


LABELS = ["SUPPORTS", "REFUTES", "NOT ENOUGH INFO"]


def load_gold_claims(path: Path, groups_limit: int) -> List[Dict[str, Any]]:
    groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            group = str(record.get("rank_group", ""))
            if not group:
                continue
            if group not in groups and len(groups) >= groups_limit:
                continue
            groups[group].append(record)
    claims: List[Dict[str, Any]] = []
    for group_id in sorted(groups):
        gold = next((row for row in groups[group_id] if row.get("is_gold")), None)
        if gold is not None:
            claims.append(gold)
    return claims


def gold_line_set(record: Dict[str, Any]) -> set[str]:
    return {str(line) for line in record.get("evidence_lines", []) if line}


def evaluate_threshold(
    claims: Sequence[Dict[str, Any]],
    retrievals: Dict[str, List[Dict[str, Any]]],
    top_k: int,
    threshold: float,
) -> Dict[str, Any]:
    rows: List[Dict[str, Any]] = []
    for claim_record in claims:
        group = str(claim_record["rank_group"])
        retrieved = retrievals[group][:top_k]
        claim = str(claim_record.get("claim_text", claim_record["subject"]["label"]))
        predicted = classify(claim, retrieved, threshold)
        rows.append(
            {
                "rank_group": group,
                "gold_label": claim_record["gold_label"],
                "predicted_label": predicted,
                "correct": predicted == claim_record["gold_label"],
            }
        )
    accuracy = sum(1 for row in rows if row["correct"]) / len(rows) if rows else 0.0
    return {
        "top_k": top_k,
        "threshold": threshold,
        "accuracy": accuracy,
        "macro_f1": macro_f1(rows),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze FEVER wiki retrieval recall and reader calibration.")
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--wiki-index", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--groups-limit", type=int, default=9999)
    parser.add_argument("--max-k", type=int, default=50)
    parser.add_argument("--top-k", action="append", type=int, default=[])
    parser.add_argument("--threshold", action="append", type=float, default=[])
    args = parser.parse_args()

    top_ks = args.top_k or [1, 3, 5, 10, 20, 50]
    thresholds = args.threshold or [0.0, 0.5, 1.0, 2.0, 4.0, 8.0, 12.0, 16.0]
    claims = load_gold_claims(args.input, args.groups_limit)
    corpus = load_sentence_corpus(args.wiki_index)
    tokenized, df, avg_len, inverted = build_doc_stats(corpus)

    retrievals: Dict[str, List[Dict[str, Any]]] = {}
    for claim_record in claims:
        claim = str(claim_record.get("claim_text", claim_record["subject"]["label"]))
        retrievals[str(claim_record["rank_group"])] = retrieve(
            claim,
            corpus,
            tokenized,
            df,
            avg_len,
            inverted,
            max(args.max_k, max(top_ks)),
        )

    verifiable = [claim for claim in claims if claim.get("gold_label") != "NOT ENOUGH INFO"]
    recall_rows: List[Dict[str, Any]] = []
    for k in top_ks:
        hits = 0
        total = 0
        for claim_record in verifiable:
            gold_lines = gold_line_set(claim_record)
            if not gold_lines:
                continue
            total += 1
            retrieved_ids = {row["sentence_id"] for row in retrievals[str(claim_record["rank_group"])][:k]}
            hits += int(bool(gold_lines & retrieved_ids))
        recall_rows.append({"k": k, "verifiable_claims": total, "gold_evidence_recall": hits / total if total else 0.0})

    calibration = [
        evaluate_threshold(claims, retrievals, k, threshold)
        for k in top_ks
        for threshold in thresholds
    ]
    best = max(calibration, key=lambda row: (row["accuracy"], row["macro_f1"])) if calibration else {}
    payload = {
        "evaluated_claims": len(claims),
        "corpus_sentences": len(corpus),
        "recall": recall_rows,
        "calibration": calibration,
        "best": best,
    }

    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "analysis.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# FEVER Retrieval Diagnostics",
        "",
        f"- Evaluated claims: {len(claims)}",
        f"- Corpus sentences: {len(corpus)}",
        "",
        "## Gold Evidence Recall",
        "",
        "| k | verifiable_claims | recall |",
        "|---:|---:|---:|",
    ]
    for row in recall_rows:
        lines.append(f"| {row['k']} | {row['verifiable_claims']} | {row['gold_evidence_recall']:.4f} |")
    lines.extend(
        [
            "",
            "## Best Reader Calibration",
            "",
            f"- Top-k: {best.get('top_k', '')}",
            f"- Threshold: {best.get('threshold', '')}",
            f"- Accuracy: {float(best.get('accuracy', 0.0)):.4f}",
            f"- Macro-F1: {float(best.get('macro_f1', 0.0)):.4f}",
        ]
    )
    (args.output / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print((args.output / "summary.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
