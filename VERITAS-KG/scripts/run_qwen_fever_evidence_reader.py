from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from run_fever_wiki_retrieval_reader import (  # noqa: E402
    build_doc_stats,
    load_sentence_corpus,
    macro_f1,
    retrieve,
)
from run_qwen_fever_baseline import label_from_text, load_group_sample  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Qwen as a FEVER evidence-aware reader over retrieved wiki sentences.")
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--wiki-index", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--model", default=Path(os.environ.get("QWEN_MODEL", "Qwen/Qwen3-8B")), type=Path)
    parser.add_argument("--groups-limit", type=int, default=100)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--max-new-tokens", type=int, default=8)
    args = parser.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    args.output.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(str(args.model), trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        str(args.model),
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else "auto",
        device_map={"": 0} if torch.cuda.is_available() else None,
        trust_remote_code=True,
    )
    model.eval()

    corpus = load_sentence_corpus(args.wiki_index)
    tokenized, df, avg_len, inverted = build_doc_stats(corpus)
    rows: List[Dict[str, Any]] = []
    for candidates in load_group_sample(args.input, args.groups_limit):
        gold = next((item for item in candidates if item.get("is_gold")), None)
        if gold is None:
            continue
        claim = str(gold.get("claim_text", gold["subject"]["label"]))
        retrieved = retrieve(claim, corpus, tokenized, df, avg_len, inverted, args.top_k)
        evidence_text = "\n".join(
            f"{idx + 1}. {row['sentence_id']}: {row['text']}" for idx, row in enumerate(retrieved)
        )
        prompt = (
            "Classify this FEVER claim using exactly one label: SUPPORTS, REFUTES, "
            "or NOT ENOUGH INFO. Use only the evidence sentences.\n\n"
            f"Claim: {claim}\n\nEvidence:\n{evidence_text}\n\nLabel:"
        )
        messages = [{"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        inputs = tokenizer([text], return_tensors="pt", truncation=True, max_length=2048).to(model.device)
        with torch.inference_mode():
            generated = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        output_ids = generated[0][len(inputs.input_ids[0]) :]
        raw = tokenizer.decode(output_ids, skip_special_tokens=True).strip()
        predicted = label_from_text(raw)
        rows.append(
            {
                "rank_group": gold["rank_group"],
                "claim": claim,
                "gold_label": gold["gold_label"],
                "predicted_label": predicted,
                "raw_output": raw,
                "top_sentence_id": retrieved[0]["sentence_id"] if retrieved else "",
                "correct": predicted == gold["gold_label"],
            }
        )

    accuracy = sum(1 for row in rows if row["correct"]) / len(rows) if rows else 0.0
    result = {
        "baseline": "qwen3_8b_fever_evidence_reader",
        "model": str(args.model),
        "evaluated": len(rows),
        "top_k": args.top_k,
        "accuracy": accuracy,
        "macro_f1": macro_f1(rows),
        "details": rows,
    }
    (args.output / "qwen_fever_evidence_reader.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    lines = [
        "# Qwen FEVER Evidence Reader",
        "",
        f"- Model: `{args.model}`",
        f"- Evaluated: {len(rows)}",
        f"- Top-k: {args.top_k}",
        f"- Accuracy: {accuracy:.4f}",
        f"- Macro-F1: {result['macro_f1']:.4f}",
    ]
    (args.output / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print((args.output / "summary.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
