from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


LABEL_TO_POLARITY = {
    "SUPPORTS": True,
    "REFUTES": False,
    "NOT ENOUGH INFO": True,
}


def entity_record(claim_id: str, claim: str) -> Dict[str, str]:
    return {
        "id": f"fever:C{claim_id}",
        "label": claim,
        "type": "Claim",
    }


def verdict_record(label: str) -> Dict[str, str]:
    return {
        "id": f"fever:V{label.replace(' ', '_')}",
        "label": label,
        "type": "Verdict",
    }


def load_wiki_index(path: Optional[Path]) -> Dict[str, Any]:
    if path is None:
        return {}
    if not path.exists():
        raise FileNotFoundError(f"Missing FEVER wiki index: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    pages = payload.get("pages", {})
    return pages if isinstance(pages, dict) else {}


def evidence_text_from_index(evidence_lines: List[str], wiki_index: Dict[str, Any]) -> str:
    sentences: List[str] = []
    seen = set()
    for evidence_line in evidence_lines:
        if evidence_line in seen:
            continue
        seen.add(evidence_line)
        page, _, line = evidence_line.partition("#")
        page_record = wiki_index.get(page)
        if not isinstance(page_record, dict):
            continue
        lines = page_record.get("lines", {})
        if not isinstance(lines, dict):
            continue
        sentence = " ".join(str(lines.get(line, "")).split())
        if sentence:
            sentences.append(f"{page}#{line}: {sentence}")
    return " ".join(sentences[:4])


def evidence_provenance(raw_evidence: Any, wiki_index: Optional[Dict[str, Any]] = None) -> Tuple[str, str, str, List[str]]:
    if not isinstance(raw_evidence, list):
        return "claim_only", "claim", "No evidence available", []
    passages: List[str] = []
    line_ids: List[str] = []
    for group in raw_evidence:
        if not isinstance(group, list):
            continue
        for item in group:
            if isinstance(item, list) and len(item) >= 4 and item[2] is not None:
                passages.append(f"{item[2]}#{item[3]}")
                line_ids.append(f"{item[2]}#{item[3]}")
    if passages:
        source_id = passages[0]
        indexed_text = evidence_text_from_index(line_ids, wiki_index or {})
        if indexed_text:
            return source_id, "wikipedia_sentence", indexed_text, line_ids
        return source_id, "wikipedia_sentence_pointer", "; ".join(passages[:4]), line_ids
    return "claim_only", "claim", "No evidence available", []


def context_record(label: str, split: str) -> Dict[str, Optional[str]]:
    return {
        "domain": "fever_claim",
        "time_start": None,
        "time_end": None,
        "location": None,
        "condition": f"split={split};candidate_label={label}",
    }


def load_fever_real(
    data_root: Path,
    split: str,
    limit: Optional[int],
    wiki_index: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    path = data_root / "fever" / f"paper_{split}.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"Missing FEVER file: {path}")

    records: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for idx, line in enumerate(handle):
            if limit is not None and idx >= limit:
                break
            item = json.loads(line)
            claim_id = str(item["id"])
            claim = str(item["claim"])
            gold_label = str(item["label"])
            source_id, source_type, snippet, evidence_lines = evidence_provenance(item.get("evidence"), wiki_index)
            rank_group = f"fever:{split}:{claim_id}"
            for candidate_label in ["SUPPORTS", "REFUTES", "NOT ENOUGH INFO"]:
                is_gold = candidate_label == gold_label
                evidence_present = source_type != "claim" and bool(evidence_lines)
                if candidate_label == "NOT ENOUGH INFO":
                    confidence = 0.52 if not evidence_present else 0.42
                    evidence_weight = 0.62 if not evidence_present else 0.34
                elif candidate_label == "SUPPORTS":
                    confidence = 0.46 if evidence_present else 0.31
                    evidence_weight = 0.48 if evidence_present else 0.22
                else:
                    confidence = 0.44 if evidence_present else 0.30
                    evidence_weight = 0.46 if evidence_present else 0.22
                records.append(
                    {
                        "assertion_id": f"fever-real-{split}-{claim_id}-{candidate_label.replace(' ', '_')}",
                        "dataset": "FEVER",
                        "split": split,
                        "task": "fact_verification",
                        "rank_group": rank_group,
                        "is_gold": is_gold,
                        "gold_label": gold_label,
                        "candidate_label": candidate_label,
                        "subject": entity_record(claim_id, claim),
                        "relation": "has_verdict",
                        "object": verdict_record(candidate_label),
                        "context": context_record(candidate_label, split),
                        "evidence": {
                            "source_id": f"FEVER/{source_id}",
                            "source_type": source_type,
                            "snippet": snippet,
                            "weight": evidence_weight,
                        },
                        "confidence": confidence,
                        "polarity": LABEL_TO_POLARITY[candidate_label],
                        "claim_text": claim,
                        "evidence_source": source_id,
                        "evidence_type": source_type,
                        "evidence_lines": evidence_lines,
                    }
                )
    return records


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
    parser = argparse.ArgumentParser(description="Prepare real FEVER evidence-traceable assertions.")
    parser.add_argument("--data-root", default="../data", type=Path)
    parser.add_argument("--output-dir", default="data/processed/fever_real", type=Path)
    parser.add_argument("--split", choices=["dev", "test"], default="dev")
    parser.add_argument("--limit", type=int, default=2000, help="Maximum claims to load; use -1 for full split.")
    parser.add_argument("--wiki-index", default=None, type=Path, help="Optional compact FEVER wiki sentence index.")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    wiki_index = load_wiki_index(args.wiki_index)
    records = load_fever_real(args.data_root, args.split, parse_optional_limit(args.limit), wiki_index)
    out_path = args.output_dir / "assertions.jsonl"
    count = write_jsonl(out_path, records)
    summary = {
        "dataset": "FEVER",
        "split": args.split,
        "limit": parse_optional_limit(args.limit),
        "records": count,
        "output": str(out_path),
        "wiki_index": str(args.wiki_index) if args.wiki_index else None,
        "wiki_index_pages": len(wiki_index),
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
