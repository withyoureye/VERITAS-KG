from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence


LABELS = ["SUPPORTS", "REFUTES", "NOT ENOUGH INFO"]


def load_group_sample(path: Path, groups_limit: int) -> List[List[Dict[str, Any]]]:
    groups: Dict[str, List[Dict[str, Any]]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            group = record.get("rank_group")
            if not group:
                continue
            group_id = str(group)
            if group_id not in groups and len(groups) >= groups_limit:
                continue
            groups.setdefault(group_id, []).append(record)
    return [groups[key] for key in sorted(groups)]


def label_from_text(text: str) -> str:
    normalized = text.upper()
    for label in LABELS:
        if label in normalized:
            return label
    if re.search(r"\bREFUTE|FALSE|NO\b", normalized):
        return "REFUTES"
    if re.search(r"NOT ENOUGH|UNKNOWN|INSUFFICIENT", normalized):
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
    parser = argparse.ArgumentParser(description="Run a small Qwen FEVER LLM-only baseline.")
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--model", default=Path(os.environ.get("QWEN_MODEL", "Qwen/Qwen3-8B")), type=Path)
    parser.add_argument("--groups-limit", type=int, default=100)
    parser.add_argument("--max-new-tokens", type=int, default=8)
    args = parser.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    args.output.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(str(args.model), trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        str(args.model),
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else "auto",
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()

    rows: List[Dict[str, Any]] = []
    for candidates in load_group_sample(args.input, args.groups_limit):
        gold = next((item for item in candidates if item.get("is_gold")), None)
        if gold is None:
            continue
        claim = str(gold.get("claim_text", gold["subject"]["label"]))
        prompt = (
            "Classify this FEVER claim using exactly one label: SUPPORTS, REFUTES, "
            "or NOT ENOUGH INFO.\n\n"
            f"Claim: {claim}\n\nLabel:"
        )
        messages = [{"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        inputs = tokenizer([text], return_tensors="pt").to(model.device)
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
                "correct": predicted == gold["gold_label"],
            }
        )

    accuracy = sum(1 for row in rows if row["correct"]) / len(rows) if rows else 0.0
    result = {
        "baseline": "qwen3_8b_llm_only",
        "model": str(args.model),
        "evaluated": len(rows),
        "accuracy": accuracy,
        "macro_f1": macro_f1(rows),
        "details": rows,
    }
    (args.output / "qwen_fever_baseline.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    lines = [
        "# Qwen FEVER Baseline",
        "",
        f"- Model: `{args.model}`",
        f"- Evaluated: {len(rows)}",
        f"- Accuracy: {accuracy:.4f}",
        f"- Macro-F1: {result['macro_f1']:.4f}",
    ]
    (args.output / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print((args.output / "summary.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
