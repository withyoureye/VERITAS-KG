from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from run_fever_wiki_retrieval_reader import load_sentence_corpus, macro_f1  # noqa: E402
from run_qwen_fever_baseline import label_from_text, load_group_sample  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Run dense retrieval + Qwen FEVER reader.")
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--wiki-index", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--retriever", default="sentence-transformers/all-roberta-large-v1")
    parser.add_argument("--qwen-model", default=Path(os.environ.get("QWEN_MODEL", "Qwen/Qwen3-8B")), type=Path)
    parser.add_argument("--groups-limit", type=int, default=9999)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-new-tokens", type=int, default=8)
    args = parser.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from transformers import AutoModel, AutoTokenizer as AutoRetrieverTokenizer

    args.output.mkdir(parents=True, exist_ok=True)

    corpus = load_sentence_corpus(args.wiki_index)
    corpus_texts = [row["text"] for row in corpus]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    retriever_tokenizer = AutoRetrieverTokenizer.from_pretrained(
        args.retriever,
        local_files_only=True,
    )
    retriever_model = AutoModel.from_pretrained(
        args.retriever,
        local_files_only=True,
    ).to(device)
    retriever_model.eval()

    def encode_texts(texts: List[str]) -> torch.Tensor:
        outputs: List[torch.Tensor] = []
        for start in range(0, len(texts), args.batch_size):
            batch = texts[start : start + args.batch_size]
            encoded = retriever_tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=256,
                return_tensors="pt",
            ).to(device)
            with torch.inference_mode():
                hidden = retriever_model(**encoded).last_hidden_state
                mask = encoded["attention_mask"].unsqueeze(-1).float()
                pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
                pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
            outputs.append(pooled.detach().cpu())
        return torch.cat(outputs, dim=0).to(device)

    corpus_embeddings = encode_texts(corpus_texts)

    tokenizer = AutoTokenizer.from_pretrained(str(args.qwen_model), trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        str(args.qwen_model),
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else "auto",
        device_map={"": 0} if torch.cuda.is_available() else None,
        trust_remote_code=True,
    )
    model.eval()

    rows: List[Dict[str, Any]] = []
    for candidates in load_group_sample(args.input, args.groups_limit):
        gold = next((item for item in candidates if item.get("is_gold")), None)
        if gold is None:
            continue
        claim = str(gold.get("claim_text", gold["subject"]["label"]))
        query_embedding = encode_texts([claim])
        scores = torch.matmul(query_embedding, corpus_embeddings.T).squeeze(0)
        top_scores, top_indices = torch.topk(scores, k=min(args.top_k, len(corpus)))
        retrieved = [
            {
                "sentence_id": corpus[int(idx)]["sentence_id"],
                "text": corpus[int(idx)]["text"],
                "score": float(score),
            }
            for score, idx in zip(top_scores.detach().cpu(), top_indices.detach().cpu())
        ]
        evidence_text = "\n".join(
            f"{idx + 1}. {row['sentence_id']}: {row['text']}" for idx, row in enumerate(retrieved)
        )
        prompt = (
            "Classify this FEVER claim using exactly one label: SUPPORTS, REFUTES, "
            "or NOT ENOUGH INFO. Use only the evidence sentences.\n\n"
            f"Claim: {claim}\n\nEvidence:\n{evidence_text}\n\nLabel:"
        )
        text = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
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
        "baseline": "dense_sentence_transformer_qwen_reader",
        "retriever": args.retriever,
        "qwen_model": str(args.qwen_model),
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
        "# Dense Retrieval + Qwen FEVER Reader",
        "",
        f"- Retriever: `{args.retriever}`",
        f"- Qwen model: `{args.qwen_model}`",
        f"- Evaluated: {len(rows)}",
        f"- Top-k: {args.top_k}",
        f"- Accuracy: {accuracy:.4f}",
        f"- Macro-F1: {result['macro_f1']:.4f}",
    ]
    (args.output / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print((args.output / "summary.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
