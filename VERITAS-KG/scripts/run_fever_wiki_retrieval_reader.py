from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Set, Tuple


TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")
LABELS = ["SUPPORTS", "REFUTES", "NOT ENOUGH INFO"]
NEGATION_TERMS = {"not", "never", "no", "false", "disassociated", "refused", "without"}


def tokenize(text: str) -> List[str]:
    return [token.lower() for token in TOKEN_RE.findall(text)]


def load_claims(path: Path, groups_limit: int) -> List[Dict[str, Any]]:
    claims: List[Dict[str, Any]] = []
    seen = set()
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            group = record.get("rank_group")
            if not group or group in seen:
                continue
            if len(seen) >= groups_limit:
                continue
            seen.add(group)
            if record.get("is_gold"):
                claims.append(record)
            else:
                claims.append(record)
    gold_by_group: Dict[str, Dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            group = record.get("rank_group")
            if group in seen and record.get("is_gold"):
                gold_by_group[str(group)] = record
    return [gold_by_group[group] for group in sorted(gold_by_group)]


def load_sentence_corpus(index_path: Path) -> List[Dict[str, str]]:
    payload = json.loads(index_path.read_text(encoding="utf-8"))
    corpus: List[Dict[str, str]] = []
    seen = set()
    for page_id, page in payload.get("pages", {}).items():
        lines = page.get("lines", {}) if isinstance(page, dict) else {}
        if not isinstance(lines, dict):
            continue
        for line_no, text in lines.items():
            sentence = str(text).strip()
            if not sentence:
                continue
            sentence_id = f"{page_id}#{line_no}"
            if sentence_id in seen:
                continue
            seen.add(sentence_id)
            corpus.append({"sentence_id": sentence_id, "text": sentence})
    return corpus


def build_doc_stats(corpus: Sequence[Dict[str, str]]) -> Tuple[List[List[str]], Counter, float, Dict[str, Set[int]]]:
    tokenized = [tokenize(row["text"]) for row in corpus]
    df = Counter(token for doc in tokenized for token in set(doc))
    inverted: Dict[str, Set[int]] = {}
    for idx, doc in enumerate(tokenized):
        for token in set(doc):
            inverted.setdefault(token, set()).add(idx)
    avg_len = sum(len(doc) for doc in tokenized) / len(tokenized) if tokenized else 1.0
    return tokenized, df, avg_len, inverted


def bm25_score(query: Sequence[str], doc: Sequence[str], df: Counter, n_docs: int, avg_len: float) -> float:
    tf = Counter(doc)
    score = 0.0
    k1 = 1.2
    b = 0.75
    doc_len = max(len(doc), 1)
    for token in query:
        freq = tf.get(token, 0)
        if not freq:
            continue
        idf = math.log(1 + (n_docs - df.get(token, 0) + 0.5) / (df.get(token, 0) + 0.5))
        denom = freq + k1 * (1 - b + b * doc_len / max(avg_len, 1e-9))
        score += idf * (freq * (k1 + 1)) / denom
    return score


def retrieve(
    claim: str,
    corpus: Sequence[Dict[str, str]],
    tokenized: Sequence[Sequence[str]],
    df: Counter,
    avg_len: float,
    inverted: Dict[str, Set[int]],
    top_k: int,
) -> List[Dict[str, Any]]:
    query = tokenize(claim)
    candidate_ids: Set[int] = set()
    for token in query:
        candidate_ids.update(inverted.get(token, set()))
    if not candidate_ids:
        candidate_ids = set(range(len(corpus)))
    scored = [
        {
            "sentence_id": row["sentence_id"],
            "text": row["text"],
            "score": bm25_score(query, doc, df, len(corpus), avg_len),
        }
        for idx in candidate_ids
        for row, doc in [(corpus[idx], tokenized[idx])]
    ]
    return sorted(scored, key=lambda row: row["score"], reverse=True)[:top_k]


def classify(claim: str, retrieved: Sequence[Dict[str, Any]], threshold: float) -> str:
    if not retrieved or float(retrieved[0]["score"]) < threshold:
        return "NOT ENOUGH INFO"
    claim_tokens = set(tokenize(claim))
    evidence_tokens = set(tokenize(" ".join(row["text"] for row in retrieved)))
    overlap = len(claim_tokens & evidence_tokens) / len(claim_tokens) if claim_tokens else 0.0
    if any(token in claim_tokens for token in NEGATION_TERMS):
        return "REFUTES"
    if overlap < 0.15:
        return "NOT ENOUGH INFO"
    return "SUPPORTS"


def macro_f1(rows: Sequence[Dict[str, Any]]) -> float:
    scores: List[float] = []
    for label in LABELS:
        tp = sum(1 for row in rows if row["gold_label"] == label and row["predicted_label"] == label)
        fp = sum(1 for row in rows if row["gold_label"] != label and row["predicted_label"] == label)
        fn = sum(1 for row in rows if row["gold_label"] == label and row["predicted_label"] != label)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        scores.append(2 * precision * recall / (precision + recall) if precision + recall else 0.0)
    return sum(scores) / len(scores)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a FEVER BM25 retrieval + lightweight reader baseline over local wiki sentences.")
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--wiki-index", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--groups-limit", type=int, default=9999)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--threshold", type=float, default=2.0)
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    claims = load_claims(args.input, args.groups_limit)
    corpus = load_sentence_corpus(args.wiki_index)
    tokenized, df, avg_len, inverted = build_doc_stats(corpus)
    rows: List[Dict[str, Any]] = []
    for claim_record in claims:
        claim = str(claim_record.get("claim_text", claim_record["subject"]["label"]))
        retrieved = retrieve(claim, corpus, tokenized, df, avg_len, inverted, args.top_k)
        predicted = classify(claim, retrieved, args.threshold)
        rows.append(
            {
                "rank_group": claim_record["rank_group"],
                "claim": claim,
                "gold_label": claim_record["gold_label"],
                "predicted_label": predicted,
                "correct": predicted == claim_record["gold_label"],
                "top_sentence_id": retrieved[0]["sentence_id"] if retrieved else "",
                "top_score": retrieved[0]["score"] if retrieved else 0.0,
            }
        )

    accuracy = sum(1 for row in rows if row["correct"]) / len(rows) if rows else 0.0
    result = {
        "baseline": "fever_bm25_wiki_retrieval_lightweight_reader",
        "evaluated": len(rows),
        "corpus_sentences": len(corpus),
        "top_k": args.top_k,
        "threshold": args.threshold,
        "accuracy": accuracy,
        "macro_f1": macro_f1(rows),
        "details": rows[:100],
    }
    (args.output / "baseline.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (args.output / "summary.md").write_text(
        "\n".join(
            [
                "# FEVER Wiki Retrieval+Reader Baseline",
                "",
                f"- Evaluated: {len(rows)}",
                f"- Corpus sentences: {len(corpus)}",
                f"- Top-k: {args.top_k}",
                f"- Threshold: {args.threshold}",
                f"- Accuracy: {accuracy:.4f}",
                f"- Macro-F1: {result['macro_f1']:.4f}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    print((args.output / "summary.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
