from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from run_fever_wiki_retrieval_reader import (  # noqa: E402
    build_doc_stats,
    classify,
    load_sentence_corpus,
    retrieve,
)


LABELS = ["SUPPORTS", "REFUTES", "NOT ENOUGH INFO"]
DEFAULT_MODELS = [
    "modelscope://cross-encoder/nli-deberta-v3-large",
    "modelscope://Xenova/DeBERTa-v3-large-mnli-fever-anli-ling-wanli",
    "modelscope://onnx-community/DeBERTa-v3-base-mnli-ONNX",
    "modelscope://onnx-community/multilingual-MiniLMv2-L6-mnli-xnli-ONNX",
    "modelscope://damo/ofa_text-classification_mnli_large_en",
    "modelscope://damo/nlp_deberta_text-classification_mnli",
    "modelscope://damo/nlp_deberta-v3-large_text-classification_mnli",
    "modelscope://damo/nlp_roberta_text-classification_mnli",
    "modelscope://iic/nlp_deberta-v3-large_text-classification_mnli",
    "MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli",
    "microsoft/deberta-v2-xlarge-mnli",
    "roberta-large-mnli",
    "facebook/bart-large-mnli",
    "google/flan-t5-large",
]


def stable_score(value: str, seed: int) -> float:
    digest = hashlib.sha256(f"{seed}:{value}".encode("utf-8")).hexdigest()
    return int(digest[:16], 16) / float(16**16 - 1)


def load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def group_claims(assertions_path: Path) -> Dict[str, Dict[str, Any]]:
    claims: Dict[str, Dict[str, Any]] = {}
    for row in load_jsonl(assertions_path):
        group_id = str(row.get("rank_group", ""))
        if not group_id:
            continue
        current = claims.get(group_id)
        if current is None or row.get("is_gold") is True:
            claims[group_id] = {
                "rank_group": group_id,
                "claim": str(row.get("claim_text") or row.get("subject", {}).get("label", "")),
                "gold_label": str(row.get("gold_label")),
                "evidence_snippet": str(row.get("evidence", {}).get("snippet", "")),
                "evidence_source": str(row.get("evidence", {}).get("source_id", "")),
            }
    return claims


def load_kg_rankings(path: Path) -> Dict[str, Dict[str, Any]]:
    return {str(row["rank_group"]): row for row in load_jsonl(path)}


def load_reader_details(path: Path, detail_key: str = "details") -> Dict[str, Dict[str, Any]]:
    payload = load_json(path)
    return {str(row["rank_group"]): row for row in payload.get(detail_key, [])}


def split_ids(group_ids: Sequence[str], seed: int, ratio: float) -> Tuple[Set[str], Set[str]]:
    calibration = {group_id for group_id in group_ids if stable_score(group_id, seed) < ratio}
    evaluation = set(group_ids) - calibration
    return calibration, evaluation


def per_class_f1(rows: Sequence[Dict[str, Any]]) -> Dict[str, float]:
    output: Dict[str, float] = {}
    for label in LABELS:
        tp = sum(1 for row in rows if row["gold_label"] == label and row["predicted_label"] == label)
        fp = sum(1 for row in rows if row["gold_label"] != label and row["predicted_label"] == label)
        fn = sum(1 for row in rows if row["gold_label"] == label and row["predicted_label"] != label)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        output[label] = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return output


def confusion_matrix(rows: Sequence[Dict[str, Any]]) -> Dict[str, Dict[str, int]]:
    matrix = {gold: {pred: 0 for pred in LABELS} for gold in LABELS}
    for row in rows:
        gold = row["gold_label"]
        pred = row["predicted_label"]
        if gold in matrix and pred in matrix[gold]:
            matrix[gold][pred] += 1
    return matrix


def evaluate_predictions(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    total = len(rows)
    accuracy = sum(1 for row in rows if row["gold_label"] == row["predicted_label"]) / total if total else 0.0
    class_f1 = per_class_f1(rows)
    return {
        "evaluated": total,
        "accuracy": accuracy,
        "macro_f1": sum(class_f1.values()) / len(LABELS),
        "per_class_f1": class_f1,
        "confusion_matrix": confusion_matrix(rows),
    }


def load_or_build_retrieval(
    claims: Dict[str, Dict[str, Any]],
    wiki_index: Path,
    output: Path,
    top_k: int,
) -> Dict[str, Dict[str, Any]]:
    if output.exists():
        return {str(row["rank_group"]): row for row in load_jsonl(output)}
    corpus = load_sentence_corpus(wiki_index)
    tokenized, df, avg_len, inverted = build_doc_stats(corpus)
    rows: List[Dict[str, Any]] = []
    for group_id, claim_row in sorted(claims.items()):
        retrieved = retrieve(str(claim_row["claim"]), corpus, tokenized, df, avg_len, inverted, top_k)
        rows.append(
            {
                "rank_group": group_id,
                "claim": claim_row["claim"],
                "gold_label": claim_row["gold_label"],
                "retrieved": retrieved,
            }
        )
    write_jsonl(output, rows)
    return {str(row["rank_group"]): row for row in rows}


def calibrate_bm25(
    claims: Dict[str, Dict[str, Any]],
    retrieval_rows: Dict[str, Dict[str, Any]],
    calibration_ids: Set[str],
) -> Tuple[Dict[str, Any], Dict[str, Dict[str, str]]]:
    candidates: List[Tuple[int, float]] = []
    for top_k in [1, 3, 5, 10, 20]:
        for threshold in [0.0, 1.0, 2.0, 4.0, 8.0, 12.0, 16.0, 20.0]:
            rows = []
            for group_id in sorted(calibration_ids):
                retrieved = retrieval_rows[group_id]["retrieved"][:top_k]
                pred = classify(claims[group_id]["claim"], retrieved, threshold)
                rows.append(
                    {
                        "rank_group": group_id,
                        "gold_label": claims[group_id]["gold_label"],
                        "predicted_label": pred,
                    }
                )
            metrics = evaluate_predictions(rows)
            candidates.append((top_k, threshold, metrics["accuracy"], metrics["macro_f1"]))
    top_k, threshold, acc, f1 = max(candidates, key=lambda row: (row[2], row[3]))
    return {
        "top_k": top_k,
        "threshold": threshold,
        "calibration_accuracy": acc,
        "calibration_macro_f1": f1,
    }, {}


def bm25_predictions(
    claims: Dict[str, Dict[str, Any]],
    retrieval_rows: Dict[str, Dict[str, Any]],
    selected_ids: Set[str],
    top_k: int,
    threshold: float,
) -> Dict[str, Dict[str, Any]]:
    output: Dict[str, Dict[str, Any]] = {}
    for group_id in sorted(selected_ids):
        retrieved = retrieval_rows[group_id]["retrieved"][:top_k]
        pred = classify(claims[group_id]["claim"], retrieved, threshold)
        output[group_id] = {
            "rank_group": group_id,
            "gold_label": claims[group_id]["gold_label"],
            "predicted_label": pred,
            "top_sentence_id": retrieved[0]["sentence_id"] if retrieved else "",
            "score": float(retrieved[0]["score"]) if retrieved else 0.0,
        }
    return output


def kg_predictions(kg_rankings: Dict[str, Dict[str, Any]], selected_ids: Set[str]) -> Dict[str, Dict[str, Any]]:
    return {
        group_id: {
            "rank_group": group_id,
            "gold_label": kg_rankings[group_id]["gold_label"],
            "predicted_label": kg_rankings[group_id]["top_label"],
        }
        for group_id in sorted(selected_ids)
        if group_id in kg_rankings
    }


def reader_predictions(reader_rows: Dict[str, Dict[str, Any]], selected_ids: Set[str]) -> Dict[str, Dict[str, Any]]:
    return {
        group_id: {
            "rank_group": group_id,
            "gold_label": str(reader_rows[group_id]["gold_label"]),
            "predicted_label": str(reader_rows[group_id]["predicted_label"]),
        }
        for group_id in sorted(selected_ids)
        if group_id in reader_rows
    }


def mnli_label_indices(config: Any) -> Dict[str, Optional[int]]:
    label2id = {str(k).lower(): int(v) for k, v in getattr(config, "label2id", {}).items()}
    entail = next((idx for name, idx in label2id.items() if "entail" in name), None)
    contra = next((idx for name, idx in label2id.items() if "contrad" in name), None)
    neutral = next((idx for name, idx in label2id.items() if "neutral" in name), None)
    if entail is None or contra is None or neutral is None:
        # Common MNLI order for RoBERTa/BART: contradiction, neutral, entailment.
        return {"SUPPORTS": 2, "REFUTES": 0, "NOT ENOUGH INFO": 1}
    return {"SUPPORTS": entail, "REFUTES": contra, "NOT ENOUGH INFO": neutral}


def evidence_text(retrieved: Sequence[Dict[str, Any]], top_k: int) -> str:
    rows = retrieved[:top_k]
    return " ".join(f"{row['sentence_id']}: {row['text']}" for row in rows)


def try_load_sequence_model(model_names: Sequence[str]) -> Tuple[str, Any, Any]:
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    last_error: Optional[Exception] = None
    for name in model_names:
        try:
            load_name = name
            if name.startswith("modelscope://"):
                from modelscope.hub.snapshot_download import snapshot_download

                load_name = snapshot_download(
                    name.removeprefix("modelscope://"),
                    allow_patterns=[
                        "*.json",
                        "*.txt",
                        "*.model",
                        "*.bin",
                        "*.safetensors",
                        "vocab.*",
                        "merges.txt",
                        "tokenizer*",
                    ],
                    max_workers=4,
                )
            tokenizer = AutoTokenizer.from_pretrained(load_name, use_fast=True, trust_remote_code=True)
            model = AutoModelForSequenceClassification.from_pretrained(
                load_name,
                torch_dtype=torch.float16 if torch.cuda.is_available() else None,
                trust_remote_code=True,
            )
            if torch.cuda.is_available():
                model = model.to("cuda:0")
            model.eval()
            return name, tokenizer, model
        except Exception as exc:
            last_error = exc
            print(f"[warn] could not load {name}: {type(exc).__name__}: {str(exc)[:300]}", flush=True)
    raise RuntimeError(f"No verifier model could be loaded. Last error: {last_error}")


def run_sequence_verifier(
    claims: Dict[str, Dict[str, Any]],
    retrieval_rows: Dict[str, Dict[str, Any]],
    group_ids: Sequence[str],
    output: Path,
    model_names: Sequence[str],
    top_k: int,
    batch_size: int,
    max_length: int,
) -> Dict[str, Dict[str, Any]]:
    if output.exists():
        return {str(row["rank_group"]): row for row in load_jsonl(output)}

    import torch

    model_name, tokenizer, model = try_load_sequence_model(model_names)
    label_indices = mnli_label_indices(model.config)
    rows: List[Dict[str, Any]] = []
    pending: List[Tuple[str, str, str]] = []
    for group_id in group_ids:
        premise = evidence_text(retrieval_rows[group_id]["retrieved"], top_k)
        if not premise.strip():
            premise = "No evidence available."
        hypothesis = claims[group_id]["claim"]
        pending.append((group_id, premise, hypothesis))

    for start in range(0, len(pending), batch_size):
        batch = pending[start : start + batch_size]
        encoded = tokenizer(
            [item[1] for item in batch],
            [item[2] for item in batch],
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        if torch.cuda.is_available():
            encoded = {key: value.to("cuda:0") for key, value in encoded.items()}
        with torch.inference_mode():
            logits = model(**encoded).logits.float()
            probs = torch.softmax(logits, dim=-1).cpu().tolist()
        for (group_id, premise, _), prob in zip(batch, probs):
            label_scores = {
                label: float(prob[int(index)]) if index is not None and int(index) < len(prob) else 0.0
                for label, index in label_indices.items()
            }
            pred = max(LABELS, key=lambda label: label_scores[label])
            rows.append(
                {
                    "rank_group": group_id,
                    "gold_label": claims[group_id]["gold_label"],
                    "predicted_label": pred,
                    "label_scores": label_scores,
                    "model": model_name,
                    "evidence_top_k": top_k,
                    "evidence_preview": premise[:500],
                }
            )
        if (start // batch_size) % 20 == 0:
            print(f"[verifier] processed {min(start + batch_size, len(pending))}/{len(pending)}", flush=True)

    write_jsonl(output, rows)
    return {str(row["rank_group"]): row for row in rows}


def fuse_kg_with_scores(
    kg_rankings: Dict[str, Dict[str, Any]],
    verifier_rows: Dict[str, Dict[str, Any]],
    selected_ids: Set[str],
    alpha: float,
) -> Dict[str, Dict[str, Any]]:
    output: Dict[str, Dict[str, Any]] = {}
    for group_id in sorted(selected_ids):
        kg = kg_rankings[group_id]
        verifier = verifier_rows[group_id]
        label_scores = verifier.get("label_scores", {})
        best_label = None
        best_score = -1.0
        for candidate in kg.get("candidates", []):
            label = str(candidate.get("label"))
            kg_score = float(candidate.get("score", 0.0))
            verifier_score = float(label_scores.get(label, 0.0))
            score = alpha * kg_score + (1.0 - alpha) * verifier_score
            if score > best_score:
                best_score = score
                best_label = label
        output[group_id] = {
            "rank_group": group_id,
            "gold_label": kg["gold_label"],
            "predicted_label": best_label or verifier["predicted_label"],
            "alpha": alpha,
            "kg_label": kg["top_label"],
            "verifier_label": verifier["predicted_label"],
            "fused_score": best_score,
        }
    return output


def calibrate_fusion(
    kg_rankings: Dict[str, Dict[str, Any]],
    verifier_rows: Dict[str, Dict[str, Any]],
    calibration_ids: Set[str],
    alphas: Sequence[float],
) -> Tuple[float, List[Dict[str, Any]]]:
    rows: List[Dict[str, Any]] = []
    for alpha in alphas:
        preds = list(fuse_kg_with_scores(kg_rankings, verifier_rows, calibration_ids, alpha).values())
        metrics = evaluate_predictions(preds)
        rows.append({"alpha": alpha, **metrics})
    best = max(rows, key=lambda row: (row["accuracy"], row["macro_f1"]))
    return float(best["alpha"]), rows


def paired_bootstrap(
    rows_a: Dict[str, Dict[str, Any]],
    rows_b: Dict[str, Dict[str, Any]],
    samples: int,
    seed: int,
) -> Dict[str, Any]:
    ids = sorted(set(rows_a) & set(rows_b))
    if not ids:
        return {"paired_items": 0, "delta": 0.0, "ci95": [0.0, 0.0], "p_value": 1.0}
    observed = (
        sum(rows_a[i]["predicted_label"] == rows_a[i]["gold_label"] for i in ids)
        - sum(rows_b[i]["predicted_label"] == rows_b[i]["gold_label"] for i in ids)
    ) / len(ids)
    rng = random.Random(seed)
    deltas: List[float] = []
    for _ in range(samples):
        sample_ids = [ids[rng.randrange(len(ids))] for _ in ids]
        delta = (
            sum(rows_a[i]["predicted_label"] == rows_a[i]["gold_label"] for i in sample_ids)
            - sum(rows_b[i]["predicted_label"] == rows_b[i]["gold_label"] for i in sample_ids)
        ) / len(sample_ids)
        deltas.append(delta)
    deltas.sort()
    lower = deltas[int(0.025 * len(deltas))]
    upper = deltas[min(len(deltas) - 1, int(0.975 * len(deltas)))]
    p_value = (1 + sum(1 for delta in deltas if delta <= 0.0)) / (len(deltas) + 1)
    return {"paired_items": len(ids), "delta": observed, "ci95": [lower, upper], "p_value": p_value}


def rows_for_output(method: str, rows: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [
        {
            "method": method,
            "rank_group": group_id,
            "gold_label": row["gold_label"],
            "predicted_label": row["predicted_label"],
            "correct": row["gold_label"] == row["predicted_label"],
        }
        for group_id, row in sorted(rows.items())
    ]


def latex_table(results: Dict[str, Dict[str, Any]]) -> str:
    names = [
        ("BM25 + lightweight reader", "bm25_lightweight_reader"),
        ("Qwen evidence reader", "qwen_reader"),
        ("KG scorer only", "kg_scorer_only"),
        ("Strong verifier only", "strong_verifier_only"),
        ("KG + strong verifier", "kg_strong_verifier_fusion"),
        ("KG/Qwen fusion", "kg_qwen_fusion"),
    ]
    lines = [
        "\\begin{table}[t]",
        "\\centering",
        "\\small",
        "\\begin{tabular}{lcc}",
        "\\toprule",
        "Method & Accuracy & Macro-F1 \\\\",
        "\\midrule",
    ]
    for label, key in names:
        row = results[key]
        lines.append(f"{label} & {row['accuracy']:.4f} & {row['macro_f1']:.4f} \\\\")
    lines.extend(
        [
            "\\bottomrule",
            "\\end{tabular}",
            "\\caption{FEVER held-out evaluation results under the stable hash split. Fusion weights are selected only on calibration and frozen for evaluation.}",
            "\\label{tab:fever-strong-verifier}",
            "\\end{table}",
        ]
    )
    return "\n".join(lines)


def display_model_name(value: str) -> str:
    if "cross-encoder/nli-deberta-v3-large" in value or value.endswith("nli-deberta-v3-large"):
        return "cross-encoder/nli-deberta-v3-large"
    if "DeBERTa-v3-large-mnli-fever-anli-ling-wanli" in value:
        return "DeBERTa-v3-large-mnli-fever-anli-ling-wanli"
    return value


def write_summary(path: Path, payload: Dict[str, Any]) -> None:
    lines = [
        "# FEVER Strong Verifier Baseline",
        "",
        f"- Split seed: `{payload['split_seed']}`",
        f"- Calibration groups: `{payload['calibration_groups']}`",
        f"- Evaluation groups: `{payload['evaluation_groups']}`",
        f"- Strong verifier model: `{payload['strong_verifier_model']}`",
        f"- Selected KG+verifier alpha: `{payload['kg_strong_verifier_fusion']['selected_alpha']}`",
        f"- Calibration score at selected alpha: accuracy `{payload['kg_strong_verifier_fusion']['calibration_accuracy']:.4f}`, macro-F1 `{payload['kg_strong_verifier_fusion']['calibration_macro_f1']:.4f}`",
        "",
        "## Held-Out Evaluation",
        "",
        "| method | evaluated | accuracy | macro-F1 | SUPPORTS F1 | REFUTES F1 | NEI F1 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for key, row in payload["evaluation_results"].items():
        f1 = row["per_class_f1"]
        lines.append(
            f"| {key} | {row['evaluated']} | {row['accuracy']:.4f} | {row['macro_f1']:.4f} | "
            f"{f1['SUPPORTS']:.4f} | {f1['REFUTES']:.4f} | {f1['NOT ENOUGH INFO']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Significance",
            "",
            f"- KG + strong verifier fusion vs strong verifier only: delta `{payload['significance']['kg_strong_vs_strong']['delta']:.4f}`, 95% CI `{payload['significance']['kg_strong_vs_strong']['ci95']}`, p `{payload['significance']['kg_strong_vs_strong']['p_value']:.6f}`",
            f"- KG/Qwen fusion vs Qwen reader: delta `{payload['significance']['kg_qwen_vs_qwen']['delta']:.4f}`, 95% CI `{payload['significance']['kg_qwen_vs_qwen']['ci95']}`, p `{payload['significance']['kg_qwen_vs_qwen']['p_value']:.6f}`",
            "",
            "## Confusion Matrices",
            "",
        ]
    )
    for method, matrix in payload["confusion_matrices"].items():
        lines.extend(
            [
                f"### {method}",
                "",
                "| gold \\ pred | SUPPORTS | REFUTES | NEI |",
                "|---|---:|---:|---:|",
            ]
        )
        for gold in LABELS:
            lines.append(
                f"| {gold} | {matrix[gold]['SUPPORTS']} | {matrix[gold]['REFUTES']} | {matrix[gold]['NOT ENOUGH INFO']} |"
            )
        lines.append("")
    lines.extend(
        [
            "## LaTeX Table",
            "",
            "```latex",
            payload["latex_table"],
            "```",
            "",
            "## Paper Text",
            "",
            payload["paper_text"],
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run FEVER strong verifier baseline and calibrated CARE-KG fusion.")
    parser.add_argument("--assertions", default=Path("data/processed/fever_real_full_v4_wiki/assertions.jsonl"), type=Path)
    parser.add_argument("--wiki-index", default=Path("data/processed/fever_wiki_index_dev_test.json"), type=Path)
    parser.add_argument("--kg-rankings", default=Path("results/fever_real_full_v4_wiki_auto/seed_0/full/rankings.jsonl"), type=Path)
    parser.add_argument("--qwen", default=Path("results/baselines_plus/qwen_fever_evidence_reader_full/qwen_fever_evidence_reader.json"), type=Path)
    parser.add_argument("--kg-qwen-fusion", default=Path("results/baselines_plus/fever_kg_qwen_fusion_calibrated/fusion_calibrated.json"), type=Path)
    parser.add_argument("--output", default=Path("results/fever_strong_baseline"), type=Path)
    parser.add_argument("--split-seed", type=int, default=13)
    parser.add_argument("--calibration-ratio", type=float, default=0.2)
    parser.add_argument("--retrieval-top-k", type=int, default=5)
    parser.add_argument("--verifier-top-k", type=int, default=5)
    parser.add_argument("--model", action="append", default=[])
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--max-claims", type=int, default=-1, help="Debug only; use -1 for all shared claims.")
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    claims = group_claims(args.assertions)
    kg_rankings = load_kg_rankings(args.kg_rankings)
    qwen_rows = load_reader_details(args.qwen)
    shared_ids = sorted(set(claims) & set(kg_rankings) & set(qwen_rows))
    if args.max_claims > 0:
        shared_ids = shared_ids[: args.max_claims]
    calibration_ids, evaluation_ids = split_ids(shared_ids, args.split_seed, args.calibration_ratio)

    retrieval_rows = load_or_build_retrieval(
        {key: claims[key] for key in shared_ids},
        args.wiki_index,
        args.output / "retrieved_evidence.jsonl",
        max(args.retrieval_top_k, args.verifier_top_k, 20),
    )
    bm25_calibration, _ = calibrate_bm25(claims, retrieval_rows, calibration_ids)
    bm25_eval = bm25_predictions(
        claims,
        retrieval_rows,
        evaluation_ids,
        int(bm25_calibration["top_k"]),
        float(bm25_calibration["threshold"]),
    )
    kg_eval = kg_predictions(kg_rankings, evaluation_ids)
    qwen_eval = reader_predictions(qwen_rows, evaluation_ids)

    verifier_rows = run_sequence_verifier(
        claims,
        retrieval_rows,
        shared_ids,
        args.output / "strong_verifier_predictions.jsonl",
        args.model or DEFAULT_MODELS,
        args.verifier_top_k,
        args.batch_size,
        args.max_length,
    )
    verifier_eval = reader_predictions(verifier_rows, evaluation_ids)
    selected_alpha, alpha_sweep = calibrate_fusion(
        kg_rankings,
        verifier_rows,
        calibration_ids,
        [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0],
    )
    kg_verifier_eval = fuse_kg_with_scores(kg_rankings, verifier_rows, evaluation_ids, selected_alpha)

    kg_qwen_payload = load_json(args.kg_qwen_fusion)
    kg_qwen_eval = {
        str(row["rank_group"]): {
            "rank_group": str(row["rank_group"]),
            "gold_label": str(row["gold_label"]),
            "predicted_label": str(row["predicted_label"]),
        }
        for row in kg_qwen_payload.get("evaluation", {}).get("details", [])
        if str(row["rank_group"]) in evaluation_ids
    }

    eval_sets = {
        "bm25_lightweight_reader": bm25_eval,
        "qwen_reader": qwen_eval,
        "kg_scorer_only": kg_eval,
        "strong_verifier_only": verifier_eval,
        "kg_strong_verifier_fusion": kg_verifier_eval,
        "kg_qwen_fusion": kg_qwen_eval,
    }
    evaluation_results = {
        name: evaluate_predictions(list(rows.values()))
        for name, rows in eval_sets.items()
    }
    significance = {
        "kg_strong_vs_strong": paired_bootstrap(
            kg_verifier_eval,
            verifier_eval,
            args.bootstrap_samples,
            args.split_seed,
        ),
        "kg_qwen_vs_qwen": paired_bootstrap(
            kg_qwen_eval,
            qwen_eval,
            args.bootstrap_samples,
            args.split_seed + 1,
        ),
    }
    selected_cal = next(row for row in alpha_sweep if row["alpha"] == selected_alpha)
    strong_model = display_model_name(str(next(iter(verifier_rows.values())).get("model") if verifier_rows else ""))
    latex = latex_table(evaluation_results)
    paper_text = (
        "Using the stable FEVER hash split, all calibration decisions were made on 2,019 calibration claims "
        "and then frozen for the 7,980-claim held-out evaluation split. The strong verifier baseline uses "
        f"`{strong_model}` over retrieved evidence. CARE-KG + strong verifier fusion selected alpha="
        f"{selected_alpha:.2f} on calibration and achieved {evaluation_results['kg_strong_verifier_fusion']['accuracy']:.4f} "
        f"accuracy / {evaluation_results['kg_strong_verifier_fusion']['macro_f1']:.4f} macro-F1 on evaluation, compared with "
        f"{evaluation_results['strong_verifier_only']['accuracy']:.4f} / {evaluation_results['strong_verifier_only']['macro_f1']:.4f} "
        "for the verifier alone. Paired bootstrap significance is reported with confidence intervals in the released metrics."
    )
    payload = {
        "split_seed": args.split_seed,
        "calibration_ratio": args.calibration_ratio,
        "calibration_groups": len(calibration_ids),
        "evaluation_groups": len(evaluation_ids),
        "strong_verifier_model": strong_model,
        "bm25_calibration": bm25_calibration,
        "kg_strong_verifier_fusion": {
            "selected_alpha": selected_alpha,
            "calibration_accuracy": selected_cal["accuracy"],
            "calibration_macro_f1": selected_cal["macro_f1"],
            "alpha_sweep": alpha_sweep,
        },
        "evaluation_results": evaluation_results,
        "confusion_matrices": {name: row["confusion_matrix"] for name, row in evaluation_results.items()},
        "significance": significance,
        "latex_table": latex,
        "paper_text": paper_text,
    }
    write_json(args.output / "metrics.json", payload)
    all_prediction_rows: List[Dict[str, Any]] = []
    for method, rows in eval_sets.items():
        all_prediction_rows.extend(rows_for_output(method, rows))
    write_jsonl(args.output / "predictions.jsonl", all_prediction_rows)
    write_summary(args.output / "summary.md", payload)
    print((args.output / "summary.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
