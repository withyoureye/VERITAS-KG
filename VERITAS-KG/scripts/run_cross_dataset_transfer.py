from __future__ import annotations

import argparse
import json
import math
import random
import sys
import zipfile
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from run_fever_strong_verifier_baseline import (  # noqa: E402
    LABELS,
    confusion_matrix,
    mnli_label_indices,
    paired_bootstrap,
    try_load_sequence_model,
)


SCIFACT_TO_FEVER = {
    "SUPPORT": "SUPPORTS",
    "SUPPORTS": "SUPPORTS",
    "CONTRADICT": "REFUTES",
    "CONTRADICTS": "REFUTES",
    "REFUTE": "REFUTES",
    "REFUTES": "REFUTES",
    "NOT ENOUGH INFO": "NOT ENOUGH INFO",
    "NEI": "NOT ENOUGH INFO",
}


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def softmax(values: Sequence[float]) -> List[float]:
    if not values:
        return []
    top = max(values)
    exps = [math.exp(value - top) for value in values]
    total = sum(exps)
    return [value / total for value in exps]


def normalize(scores: Dict[str, float]) -> Dict[str, float]:
    total = sum(max(0.0, scores.get(label, 0.0)) for label in LABELS)
    if total <= 0:
        return {label: 1.0 / len(LABELS) for label in LABELS}
    return {label: max(0.0, scores.get(label, 0.0)) / total for label in LABELS}


def token_set(text: str) -> set[str]:
    return {token for token in "".join(ch.lower() if ch.isalnum() else " " for ch in text).split() if len(token) > 2}


def jaccard(a: str, b: str) -> float:
    left = token_set(a)
    right = token_set(b)
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def has_negation(text: str) -> bool:
    padded = f" {text.lower()} "
    markers = [
        " no ",
        " not ",
        " never ",
        " neither ",
        " nor ",
        " without ",
        " fails ",
        " failed ",
        " unable ",
        " cannot ",
        "n't ",
        "contrary",
        "inconsistent",
        "unrelated",
        "lack ",
        "lacks ",
    ]
    return any(marker in padded for marker in markers)


def evaluate(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    total = len(rows)
    class_f1: Dict[str, float] = {}
    for label in LABELS:
        tp = sum(1 for row in rows if row["gold_label"] == label and row["predicted_label"] == label)
        fp = sum(1 for row in rows if row["gold_label"] != label and row["predicted_label"] == label)
        fn = sum(1 for row in rows if row["gold_label"] == label and row["predicted_label"] != label)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        class_f1[label] = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "evaluated": total,
        "accuracy": sum(row["gold_label"] == row["predicted_label"] for row in rows) / total if total else 0.0,
        "macro_f1": sum(class_f1.values()) / len(LABELS),
        "per_class_f1": class_f1,
        "confusion_matrix": confusion_matrix(rows),
        "label_distribution": dict(Counter(row["gold_label"] for row in rows)),
    }


def load_scifact_corpus(path: Path) -> Dict[str, Dict[str, Any]]:
    corpus: Dict[str, Dict[str, Any]] = {}
    for row in read_jsonl(path):
        corpus[str(row["doc_id"])] = row
    return corpus


def doc_text(doc: Dict[str, Any], max_sentences: int = 5) -> str:
    title = str(doc.get("title", "")).strip()
    abstract = [str(sentence).strip() for sentence in doc.get("abstract", []) if str(sentence).strip()]
    pieces = [title] if title else []
    pieces.extend(abstract[:max_sentences])
    return " ".join(pieces)


def scifact_gold_label(row: Dict[str, Any]) -> str:
    evidence = row.get("evidence") or {}
    labels: List[str] = []
    for docs in evidence.values():
        for item in docs:
            label = SCIFACT_TO_FEVER.get(str(item.get("label", "")).upper())
            if label:
                labels.append(label)
    if not labels:
        return "NOT ENOUGH INFO"
    counts = Counter(labels)
    return counts.most_common(1)[0][0]


def scifact_evidence_text(row: Dict[str, Any], corpus: Dict[str, Dict[str, Any]], mode: str) -> Tuple[str, Dict[str, Any]]:
    evidence = row.get("evidence") or {}
    cited_doc_ids = [str(doc_id) for doc_id in row.get("cited_doc_ids") or []]
    snippets: List[str] = []
    used_docs: List[str] = []
    used_sentences: List[str] = []

    if mode == "gold_rationale":
        for doc_id, docs in evidence.items():
            doc = corpus.get(str(doc_id), {})
            abstract = [str(sentence).strip() for sentence in doc.get("abstract", [])]
            title = str(doc.get("title", "")).strip()
            for item in docs:
                for sentence_id in item.get("sentences", []):
                    if isinstance(sentence_id, int) and 0 <= sentence_id < len(abstract):
                        prefix = f"{doc_id}#{sentence_id}"
                        snippets.append(f"{prefix}: {abstract[sentence_id]}")
                        used_sentences.append(prefix)
                        if str(doc_id) not in used_docs:
                            used_docs.append(str(doc_id))
            if not snippets and title:
                snippets.append(f"{doc_id}#title: {title}")
                used_docs.append(str(doc_id))
    elif mode == "cited_doc":
        for doc_id in cited_doc_ids:
            doc = corpus.get(str(doc_id))
            if not doc:
                continue
            snippets.append(f"{doc_id}: {doc_text(doc, max_sentences=5)}")
            used_docs.append(str(doc_id))
    else:
        raise ValueError(f"Unknown SciFact evidence mode: {mode}")

    evidence_text = " ".join(snippets).strip()
    return evidence_text or "No evidence available.", {
        "used_docs": used_docs,
        "used_sentences": used_sentences,
        "has_gold_rationale": bool(evidence),
        "has_cited_doc": bool(cited_doc_ids),
        "mode": mode,
    }


def load_scifact_claims(claims_path: Path, corpus_path: Path, mode: str) -> List[Dict[str, Any]]:
    corpus = load_scifact_corpus(corpus_path)
    rows: List[Dict[str, Any]] = []
    for row in read_jsonl(claims_path):
        evidence, metadata = scifact_evidence_text(row, corpus, mode)
        rows.append(
            {
                "dataset": "SciFact",
                "id": f"scifact:{claims_path.stem}:{row['id']}",
                "claim": str(row["claim"]),
                "evidence": evidence,
                "gold_label": scifact_gold_label(row),
                "metadata": metadata,
            }
        )
    return rows


def find_vitaminc_file(root: Path) -> Optional[Path]:
    candidates = [
        root / "dev.jsonl",
        root / "test.jsonl",
        root / "vitaminc" / "dev.jsonl",
        root / "vitaminc" / "test.jsonl",
        root / "data" / "vitaminc" / "dev.jsonl",
        root / "data" / "vitaminc" / "test.jsonl",
    ]
    candidates.extend(sorted(root.rglob("*.jsonl")) if root.exists() and root.is_dir() else [])
    for path in candidates:
        if not path.exists() or path.stat().st_size < 1000:
            continue
        try:
            first = json.loads(path.read_text(encoding="utf-8").splitlines()[0])
        except Exception:
            continue
        if {"claim", "evidence", "label"}.issubset(first):
            return path
    return None


def maybe_extract_vitaminc(zip_path: Path, output_dir: Path) -> Optional[Path]:
    if zip_path.exists() and zip_path.stat().st_size > 1000 and zipfile.is_zipfile(zip_path):
        output_dir.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zip_path) as archive:
            archive.extractall(output_dir)
    return find_vitaminc_file(output_dir)


def map_vitaminc_label(value: str) -> str:
    label = value.strip().upper().replace("_", " ")
    return SCIFACT_TO_FEVER.get(label, label if label in LABELS else "NOT ENOUGH INFO")


def load_vitaminc_claims(path: Path, limit: Optional[int]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for idx, row in enumerate(read_jsonl(path)):
        if limit is not None and idx >= limit:
            break
        evidence = row.get("evidence", "")
        if isinstance(evidence, list):
            evidence_text = " ".join(str(item) for item in evidence)
        else:
            evidence_text = str(evidence)
        rows.append(
            {
                "dataset": "VitaminC",
                "id": f"vitaminc:{path.stem}:{idx}",
                "claim": str(row.get("claim", "")),
                "evidence": evidence_text or "No evidence available.",
                "gold_label": map_vitaminc_label(str(row.get("label", ""))),
                "metadata": {"source_file": str(path)},
            }
        )
    return rows


def run_verifier(
    rows: Sequence[Dict[str, Any]],
    output_path: Path,
    model_names: Sequence[str],
    batch_size: int,
    max_length: int,
    claim_only: bool = False,
) -> List[Dict[str, Any]]:
    if output_path.exists():
        return read_jsonl(output_path)

    import torch

    model_name, tokenizer, model = try_load_sequence_model(model_names)
    label_indices = mnli_label_indices(model.config)
    predictions: List[Dict[str, Any]] = []
    pending = list(rows)

    for start in range(0, len(pending), batch_size):
        batch = pending[start : start + batch_size]
        premises = ["No evidence available." if claim_only else row["evidence"] for row in batch]
        hypotheses = [row["claim"] for row in batch]
        encoded = tokenizer(
            premises,
            hypotheses,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        if torch.cuda.is_available():
            encoded = {key: value.to("cuda:0") for key, value in encoded.items()}
        with torch.inference_mode():
            probs = torch.softmax(model(**encoded).logits.float(), dim=-1).cpu().tolist()
        for row, prob in zip(batch, probs):
            label_scores = {
                label: float(prob[int(index)]) if index is not None and int(index) < len(prob) else 0.0
                for label, index in label_indices.items()
            }
            pred = max(LABELS, key=lambda label: label_scores[label])
            predictions.append(
                {
                    "id": row["id"],
                    "dataset": row["dataset"],
                    "gold_label": row["gold_label"],
                    "predicted_label": pred,
                    "label_scores": normalize(label_scores),
                    "model": model_name,
                    "claim_only": claim_only,
                }
            )
        if (start // batch_size) % 10 == 0:
            print(f"[verifier claim_only={claim_only}] processed {min(start + batch_size, len(pending))}/{len(pending)}", flush=True)

    write_jsonl(output_path, predictions)
    return predictions


def kg_evidence_scores(row: Dict[str, Any]) -> Dict[str, float]:
    evidence = row["evidence"]
    meta = row.get("metadata", {})
    claim = row["claim"]
    if evidence == "No evidence available.":
        return {"SUPPORTS": 0.10, "REFUTES": 0.10, "NOT ENOUGH INFO": 0.80}

    relevance = jaccard(claim, evidence)
    reliability = 0.50 + min(0.35, 0.08 * len(meta.get("used_docs", [])) + 0.04 * len(meta.get("used_sentences", [])))
    reliability += min(0.10, relevance)
    reliability = min(0.95, reliability)
    refute_bias = 0.06 if has_negation(claim) != has_negation(evidence) else 0.0
    support_bias = 0.02 if relevance > 0.12 else 0.0
    nei = max(0.05, 1.0 - reliability)
    # Evidence presence is reliable, but the score intentionally does not read
    # dataset labels or rationale labels. Entail/refute separation remains for
    # the verifier; this avoids turning transfer into a hidden gold-label oracle.
    return normalize(
        {
            "SUPPORTS": 0.5 * reliability + support_bias,
            "REFUTES": 0.5 * reliability + refute_bias,
            "NOT ENOUGH INFO": nei,
        }
    )


def evidence_only_scores(row: Dict[str, Any]) -> Dict[str, float]:
    evidence = row["evidence"]
    if evidence == "No evidence available.":
        return {"SUPPORTS": 0.05, "REFUTES": 0.05, "NOT ENOUGH INFO": 0.90}
    if has_negation(evidence):
        return {"SUPPORTS": 0.42, "REFUTES": 0.48, "NOT ENOUGH INFO": 0.10}
    return {"SUPPORTS": 0.48, "REFUTES": 0.42, "NOT ENOUGH INFO": 0.10}


def scores_to_rows(rows: Sequence[Dict[str, Any]], scores_by_id: Dict[str, Dict[str, float]], method: str) -> List[Dict[str, Any]]:
    output: List[Dict[str, Any]] = []
    for row in rows:
        scores = normalize(scores_by_id[row["id"]])
        pred = max(LABELS, key=lambda label: scores[label])
        output.append(
            {
                "method": method,
                "id": row["id"],
                "dataset": row["dataset"],
                "gold_label": row["gold_label"],
                "predicted_label": pred,
                "label_scores": scores,
            }
        )
    return output


def verifier_rows(predictions: Sequence[Dict[str, Any]], method: str) -> List[Dict[str, Any]]:
    return [
        {
            "method": method,
            "id": row["id"],
            "dataset": row["dataset"],
            "gold_label": row["gold_label"],
            "predicted_label": row["predicted_label"],
            "label_scores": normalize(row.get("label_scores", {})),
        }
        for row in predictions
    ]


def fuse_rows(
    rows: Sequence[Dict[str, Any]],
    kg_scores_by_id: Dict[str, Dict[str, float]],
    verifier_by_id: Dict[str, Dict[str, Any]],
    alpha: float,
    method: str,
) -> List[Dict[str, Any]]:
    output: List[Dict[str, Any]] = []
    for row in rows:
        kg_scores = normalize(kg_scores_by_id[row["id"]])
        verifier_scores = normalize(verifier_by_id[row["id"]].get("label_scores", {}))
        combined = {
            label: alpha * kg_scores.get(label, 0.0) + (1.0 - alpha) * verifier_scores.get(label, 0.0)
            for label in LABELS
        }
        pred = max(LABELS, key=lambda label: combined[label])
        output.append(
            {
                "method": method,
                "id": row["id"],
                "dataset": row["dataset"],
                "gold_label": row["gold_label"],
                "predicted_label": pred,
                "label_scores": normalize(combined),
                "alpha": alpha,
            }
        )
    return output


def evaluate_dataset(
    name: str,
    rows: Sequence[Dict[str, Any]],
    output_dir: Path,
    model_names: Sequence[str],
    alpha: float,
    batch_size: int,
    max_length: int,
    bootstrap_samples: int,
    seed: int,
) -> Dict[str, Any]:
    dataset_dir = output_dir / name
    dataset_dir.mkdir(parents=True, exist_ok=True)
    verifier = run_verifier(rows, dataset_dir / "strong_verifier.jsonl", model_names, batch_size, max_length)
    claim_only = run_verifier(rows, dataset_dir / "claim_only_verifier.jsonl", model_names, batch_size, max_length, claim_only=True)
    verifier_by_id = {row["id"]: row for row in verifier}
    kg_scores_by_id = {row["id"]: kg_evidence_scores(row) for row in rows}
    evidence_only_by_id = {row["id"]: evidence_only_scores(row) for row in rows}

    method_rows = {
        "strong_verifier_only": verifier_rows(verifier, "strong_verifier_only"),
        "kg_evidence_scorer_only": scores_to_rows(rows, kg_scores_by_id, "kg_evidence_scorer_only"),
        "kg_strong_verifier_fusion_fixed_alpha": fuse_rows(
            rows, kg_scores_by_id, verifier_by_id, alpha, "kg_strong_verifier_fusion_fixed_alpha"
        ),
        "claim_only_verifier": verifier_rows(claim_only, "claim_only_verifier"),
        "evidence_only_scorer": scores_to_rows(rows, evidence_only_by_id, "evidence_only_scorer"),
    }
    metrics = {method: evaluate(preds) for method, preds in method_rows.items()}
    significance = {
        "fusion_vs_strong_verifier": paired_bootstrap(
            {row["id"]: row for row in method_rows["kg_strong_verifier_fusion_fixed_alpha"]},
            {row["id"]: row for row in method_rows["strong_verifier_only"]},
            bootstrap_samples,
            seed,
        ),
        "fusion_vs_kg_scorer": paired_bootstrap(
            {row["id"]: row for row in method_rows["kg_strong_verifier_fusion_fixed_alpha"]},
            {row["id"]: row for row in method_rows["kg_evidence_scorer_only"]},
            bootstrap_samples,
            seed + 1,
        ),
    }
    write_jsonl(
        dataset_dir / "predictions.jsonl",
        (
            {
                **row,
                "correct": row["gold_label"] == row["predicted_label"],
            }
            for method in method_rows.values()
            for row in method
        ),
    )
    write_json(dataset_dir / "metrics.json", {"metrics": metrics, "significance": significance})
    return {
        "dataset": name,
        "examples": len(rows),
        "label_distribution": dict(Counter(row["gold_label"] for row in rows)),
        "metrics": metrics,
        "significance": significance,
    }


def latex_table(results: Dict[str, Any]) -> str:
    lines = [
        "\\begin{table}[t]",
        "\\centering",
        "\\small",
        "\\begin{tabular}{llcc}",
        "\\toprule",
        "Dataset & Method & Accuracy & Macro-F1 \\\\",
        "\\midrule",
    ]
    order = [
        "strong_verifier_only",
        "kg_evidence_scorer_only",
        "kg_strong_verifier_fusion_fixed_alpha",
        "claim_only_verifier",
        "evidence_only_scorer",
    ]
    for dataset, payload in results.items():
        if payload.get("status") == "blocked":
            continue
        for method in order:
            row = payload["metrics"][method]
            lines.append(f"{dataset} & {method.replace('_', ' ')} & {row['accuracy']:.4f} & {row['macro_f1']:.4f} \\\\")
    lines.extend(
        [
            "\\bottomrule",
            "\\end{tabular}",
            "\\caption{Cross-dataset transfer with fusion weight fixed from FEVER calibration. No SciFact/VitaminC labels are used for tuning.}",
            "\\label{tab:cross-dataset-transfer}",
            "\\end{table}",
        ]
    )
    return "\n".join(lines)


def write_summary(path: Path, payload: Dict[str, Any]) -> None:
    lines = [
        "# Cross-Dataset Transfer",
        "",
        "Protocol: fusion alpha is fixed from FEVER calibration and is not tuned on SciFact or VitaminC.",
        f"- Fixed alpha: `{payload['fixed_alpha']}`",
        f"- Model: `{payload['model'][0] if payload['model'] else 'default verifier cascade'}`",
        "",
    ]
    for name, result in payload["datasets"].items():
        lines.extend([f"## {name}", ""])
        if result.get("status") == "blocked":
            lines.extend([f"- Status: blocked", f"- Reason: {result['reason']}", ""])
            continue
        lines.extend(
            [
                f"- Examples: `{result['examples']}`",
                f"- Label distribution: `{result['label_distribution']}`",
                "",
                "| method | accuracy | macro-F1 | SUPPORTS F1 | REFUTES F1 | NEI F1 |",
                "|---|---:|---:|---:|---:|---:|",
            ]
        )
        for method, row in result["metrics"].items():
            f1 = row["per_class_f1"]
            lines.append(
                f"| {method} | {row['accuracy']:.4f} | {row['macro_f1']:.4f} | "
                f"{f1['SUPPORTS']:.4f} | {f1['REFUTES']:.4f} | {f1['NOT ENOUGH INFO']:.4f} |"
            )
        lines.extend(["", "Significance:"])
        for test_name, test in result["significance"].items():
            lines.append(
                f"- `{test_name}`: delta `{test['delta']:.4f}`, 95% CI `{test['ci95']}`, p `{test['p_value']:.6f}`"
            )
        lines.append("")
    lines.extend(["## LaTeX", "", "```latex", payload["latex_table"], "```", ""])
    lines.extend(
        [
            "## Notes",
            "",
            "- `scifact_cited_doc` uses cited documents from the official SciFact file as evidence text and does not use rationale labels for scoring.",
            "- `scifact_gold_rationale` uses official rationale sentences as evidence input and should be treated as a diagnostic verifier/fusion transfer setting, not an end-to-end retrieval result.",
            "- VitaminC is skipped when only Git LFS pointer files or a zero-byte/corrupt zip are present.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate FEVER-calibrated fusion on SciFact/VitaminC without retuning.")
    parser.add_argument("--scifact-claims", default=Path("../data/downloads/scifact_official/data/claims_dev.jsonl"), type=Path)
    parser.add_argument("--scifact-corpus", default=Path("../data/downloads/scifact_official/data/corpus.jsonl"), type=Path)
    parser.add_argument("--vitaminc-root", default=Path("../data/downloads/vitaminc_official"), type=Path)
    parser.add_argument("--vitaminc-zip", default=Path("../data/downloads/vitaminc_official/vitaminc.zip"), type=Path)
    parser.add_argument("--output", default=Path("results/cross_dataset_transfer"), type=Path)
    parser.add_argument("--model", action="append", default=[])
    parser.add_argument("--fixed-alpha", type=float, default=0.9)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--bootstrap-samples", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--vitaminc-limit", type=int, default=5000)
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    model_names = args.model or ["cross-encoder/nli-deberta-v3-large"]
    datasets: Dict[str, Any] = {}

    for mode in ["cited_doc", "gold_rationale"]:
        rows = load_scifact_claims(args.scifact_claims, args.scifact_corpus, mode)
        datasets[f"scifact_{mode}"] = evaluate_dataset(
            f"scifact_{mode}",
            rows,
            args.output,
            model_names,
            args.fixed_alpha,
            args.batch_size,
            args.max_length,
            args.bootstrap_samples,
            args.seed,
        )

    vitaminc_path = maybe_extract_vitaminc(args.vitaminc_zip, args.vitaminc_root / "extracted") or find_vitaminc_file(args.vitaminc_root)
    if vitaminc_path is None:
        datasets["vitaminc"] = {
            "status": "blocked",
            "reason": (
                f"No real VitaminC JSONL found under {args.vitaminc_root}; "
                f"{args.vitaminc_zip} is missing, zero-byte, corrupt, or only Git LFS pointers are available."
            ),
        }
    else:
        limit = None if args.vitaminc_limit < 0 else args.vitaminc_limit
        rows = load_vitaminc_claims(vitaminc_path, limit)
        datasets["vitaminc"] = evaluate_dataset(
            "vitaminc",
            rows,
            args.output,
            model_names,
            args.fixed_alpha,
            args.batch_size,
            args.max_length,
            args.bootstrap_samples,
            args.seed,
        )

    payload = {
        "protocol": "FEVER-calibrated fixed-alpha cross-dataset transfer; no target-data tuning.",
        "fixed_alpha": args.fixed_alpha,
        "model": model_names,
        "datasets": datasets,
    }
    payload["latex_table"] = latex_table(datasets)
    write_json(args.output / "metrics.json", payload)
    write_summary(args.output / "summary.md", payload)
    print((args.output / "summary.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
