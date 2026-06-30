from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Set, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from run_fever_strong_verifier_baseline import (  # noqa: E402
    DEFAULT_MODELS,
    LABELS,
    confusion_matrix,
    evaluate_predictions,
    evidence_text,
    group_claims,
    load_jsonl,
    load_kg_rankings,
    load_or_build_retrieval,
    mnli_label_indices,
    paired_bootstrap,
    try_load_sequence_model,
)


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def stable_score(value: str, seed: int) -> float:
    digest = hashlib.sha256(f"{seed}:{value}".encode("utf-8")).hexdigest()
    return int(digest[:16], 16) / float(16**16 - 1)


def split_ids(group_ids: Sequence[str], seed: int, ratio: float) -> Tuple[Set[str], Set[str]]:
    calibration = {group_id for group_id in group_ids if stable_score(group_id, seed) < ratio}
    return calibration, set(group_ids) - calibration


def run_verifier_topk(
    claims: Dict[str, Dict[str, Any]],
    retrieval_rows: Dict[str, Dict[str, Any]],
    group_ids: Sequence[str],
    output: Path,
    model_names: Sequence[str],
    top_k_values: Sequence[int],
    batch_size: int,
    max_length: int,
) -> Dict[int, Dict[str, Dict[str, Any]]]:
    existing: Dict[int, Dict[str, Dict[str, Any]]] = {}
    missing: List[int] = []
    for top_k in top_k_values:
        path = output / f"strong_verifier_top{top_k}.jsonl"
        if path.exists():
            existing[top_k] = {str(row["rank_group"]): row for row in load_jsonl(path)}
        else:
            missing.append(top_k)
    if not missing:
        return existing

    import torch

    model_name, tokenizer, model = try_load_sequence_model(model_names)
    label_indices = mnli_label_indices(model.config)

    for top_k in missing:
        rows: List[Dict[str, Any]] = []
        pending: List[Tuple[str, str, str, float]] = []
        for group_id in group_ids:
            retrieved = retrieval_rows[group_id]["retrieved"]
            premise = evidence_text(retrieved, top_k)
            if not premise.strip():
                premise = "No evidence available."
            top_score = float(retrieved[0]["score"]) if retrieved else 0.0
            pending.append((group_id, premise, claims[group_id]["claim"], top_score))

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
            for (group_id, premise, _, top_score), prob in zip(batch, probs):
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
                        "retrieval_top_score": top_score,
                        "evidence_preview": premise[:500],
                    }
                )
            if (start // batch_size) % 25 == 0:
                print(f"[top_k={top_k}] processed {min(start + batch_size, len(pending))}/{len(pending)}", flush=True)

        path = output / f"strong_verifier_top{top_k}.jsonl"
        write_jsonl(path, rows)
        existing[top_k] = {str(row["rank_group"]): row for row in rows}
    return existing


def thresholded_verifier_rows(
    rows: Dict[str, Dict[str, Any]],
    selected_ids: Set[str],
    threshold: float,
) -> Dict[str, Dict[str, Any]]:
    output = {}
    for group_id in sorted(selected_ids):
        row = rows[group_id]
        top_score = float(row.get("retrieval_top_score", 0.0))
        pred = "NOT ENOUGH INFO" if top_score < threshold else str(row["predicted_label"])
        output[group_id] = {
            "rank_group": group_id,
            "gold_label": row["gold_label"],
            "predicted_label": pred,
            "threshold": threshold,
            "top_score": top_score,
        }
    return output


def adjusted_label_scores(row: Dict[str, Any], threshold: float) -> Dict[str, float]:
    if float(row.get("retrieval_top_score", 0.0)) < threshold:
        return {"SUPPORTS": 0.0, "REFUTES": 0.0, "NOT ENOUGH INFO": 1.0}
    return {label: float(row.get("label_scores", {}).get(label, 0.0)) for label in LABELS}


def kg_rows(kg_rankings: Dict[str, Dict[str, Any]], selected_ids: Set[str]) -> Dict[str, Dict[str, Any]]:
    return {
        group_id: {
            "rank_group": group_id,
            "gold_label": kg_rankings[group_id]["gold_label"],
            "predicted_label": kg_rankings[group_id]["top_label"],
        }
        for group_id in sorted(selected_ids)
        if group_id in kg_rankings
    }


def fuse(
    kg_rankings: Dict[str, Dict[str, Any]],
    verifier_rows: Dict[str, Dict[str, Any]],
    selected_ids: Set[str],
    alpha: float,
    threshold: float,
) -> Dict[str, Dict[str, Any]]:
    output = {}
    for group_id in sorted(selected_ids):
        kg = kg_rankings[group_id]
        label_scores = adjusted_label_scores(verifier_rows[group_id], threshold)
        best_label = None
        best_score = -1.0
        for candidate in kg.get("candidates", []):
            label = str(candidate.get("label"))
            score = alpha * float(candidate.get("score", 0.0)) + (1.0 - alpha) * float(label_scores.get(label, 0.0))
            if score > best_score:
                best_score = score
                best_label = label
        output[group_id] = {
            "rank_group": group_id,
            "gold_label": kg["gold_label"],
            "predicted_label": best_label or verifier_rows[group_id]["predicted_label"],
            "alpha": alpha,
            "threshold": threshold,
            "kg_label": kg["top_label"],
            "verifier_label": verifier_rows[group_id]["predicted_label"],
            "fused_score": best_score,
        }
    return output


def calibrate_verifier(
    topk_rows: Dict[int, Dict[str, Dict[str, Any]]],
    calibration_ids: Set[str],
    thresholds: Sequence[float],
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    sweep = []
    for top_k, rows in sorted(topk_rows.items()):
        for threshold in thresholds:
            preds = thresholded_verifier_rows(rows, calibration_ids, threshold)
            metrics = evaluate_predictions(list(preds.values()))
            sweep.append({"top_k": top_k, "threshold": threshold, **metrics})
    best = max(sweep, key=lambda row: (row["accuracy"], row["macro_f1"]))
    return best, sweep


def calibrate_fusion(
    kg_rankings: Dict[str, Dict[str, Any]],
    topk_rows: Dict[int, Dict[str, Dict[str, Any]]],
    calibration_ids: Set[str],
    thresholds: Sequence[float],
    alphas: Sequence[float],
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    sweep = []
    for top_k, rows in sorted(topk_rows.items()):
        for threshold in thresholds:
            for alpha in alphas:
                preds = fuse(kg_rankings, rows, calibration_ids, alpha, threshold)
                metrics = evaluate_predictions(list(preds.values()))
                sweep.append({"top_k": top_k, "threshold": threshold, "alpha": alpha, **metrics})
    best = max(sweep, key=lambda row: (row["accuracy"], row["macro_f1"]))
    return best, sweep


def alpha_curve(
    kg_rankings: Dict[str, Dict[str, Any]],
    verifier_rows: Dict[str, Dict[str, Any]],
    calibration_ids: Set[str],
    evaluation_ids: Set[str],
    threshold: float,
    alphas: Sequence[float],
) -> List[Dict[str, Any]]:
    rows = []
    for alpha in alphas:
        cal = evaluate_predictions(list(fuse(kg_rankings, verifier_rows, calibration_ids, alpha, threshold).values()))
        ev = evaluate_predictions(list(fuse(kg_rankings, verifier_rows, evaluation_ids, alpha, threshold).values()))
        rows.append(
            {
                "alpha": alpha,
                "calibration_accuracy": cal["accuracy"],
                "calibration_macro_f1": cal["macro_f1"],
                "evaluation_accuracy": ev["accuracy"],
                "evaluation_macro_f1": ev["macro_f1"],
            }
        )
    return rows


def write_summary(path: Path, payload: Dict[str, Any]) -> None:
    lines = [
        "# FEVER Strong Verifier Sweep",
        "",
        "BM25 retrieval plus DeBERTa/RoBERTa-style NLI verifier with calibration-only selection of top-k, NEI retrieval threshold, and fusion alpha.",
        "",
        f"- Split seed: `{payload['split_seed']}`",
        f"- Calibration claims: `{payload['calibration_groups']}`",
        f"- Evaluation claims: `{payload['evaluation_groups']}`",
        f"- Top-k grid: `{payload['top_k_values']}`",
        f"- Threshold grid: `{payload['thresholds']}`",
        f"- Alpha grid: `{payload['alphas']}`",
        "",
        "## Calibration Choices",
        "",
        f"- Verifier-only: top_k `{payload['selected_verifier']['top_k']}`, threshold `{payload['selected_verifier']['threshold']}`, calibration accuracy `{payload['selected_verifier']['accuracy']:.4f}`",
        f"- KG+verifier fusion: top_k `{payload['selected_fusion']['top_k']}`, threshold `{payload['selected_fusion']['threshold']}`, alpha `{payload['selected_fusion']['alpha']}`, calibration accuracy `{payload['selected_fusion']['accuracy']:.4f}`",
        "",
        "## Held-Out Evaluation",
        "",
        "| method | evaluated | accuracy | macro-F1 | SUPPORTS F1 | REFUTES F1 | NEI F1 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name, row in payload["evaluation_results"].items():
        f1 = row["per_class_f1"]
        lines.append(
            f"| {name} | {row['evaluated']} | {row['accuracy']:.4f} | {row['macro_f1']:.4f} | "
            f"{f1['SUPPORTS']:.4f} | {f1['REFUTES']:.4f} | {f1['NOT ENOUGH INFO']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Fusion Alpha Curve At Selected top-k/threshold",
            "",
            "| alpha | calibration acc | evaluation acc | evaluation macro-F1 |",
            "|---:|---:|---:|---:|",
        ]
    )
    for row in payload["alpha_curve"]:
        lines.append(
            f"| {row['alpha']:.2f} | {row['calibration_accuracy']:.4f} | {row['evaluation_accuracy']:.4f} | {row['evaluation_macro_f1']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Significance",
            "",
        ]
    )
    for name, row in payload["significance"].items():
        lines.append(
            f"- `{name}`: delta `{row['delta']:.4f}`, 95% CI `{row['ci95']}`, p `{row['p_value']:.6f}`"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run calibrated FEVER strong verifier sweep.")
    parser.add_argument("--assertions", default=Path("data/processed/fever_real_full_v4_wiki/assertions.jsonl"), type=Path)
    parser.add_argument("--wiki-index", default=Path("data/processed/fever_wiki_index_dev_test.json"), type=Path)
    parser.add_argument("--kg-rankings", default=Path("results/fever_real_full_v4_wiki_auto/seed_0/full/rankings.jsonl"), type=Path)
    parser.add_argument("--output", default=Path("results/fever_strong_verifier_sweep"), type=Path)
    parser.add_argument("--model", action="append", default=[])
    parser.add_argument("--top-k", action="append", type=int, default=[])
    parser.add_argument("--threshold", action="append", type=float, default=[])
    parser.add_argument("--alpha", action="append", type=float, default=[])
    parser.add_argument("--split-seed", type=int, default=13)
    parser.add_argument("--calibration-ratio", type=float, default=0.2)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--bootstrap-samples", type=int, default=5000)
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    top_k_values = args.top_k or [1, 3, 5, 10]
    thresholds = args.threshold or [0.0, 4.0, 8.0, 12.0, 16.0, 20.0, 24.0]
    alphas = args.alpha or [round(i / 10, 2) for i in range(11)]

    claims = group_claims(args.assertions)
    kg_rankings = load_kg_rankings(args.kg_rankings)
    shared_ids = sorted(set(claims) & set(kg_rankings))
    calibration_ids, evaluation_ids = split_ids(shared_ids, args.split_seed, args.calibration_ratio)
    retrieval_rows = load_or_build_retrieval(
        {key: claims[key] for key in shared_ids},
        args.wiki_index,
        args.output / "retrieved_evidence.jsonl",
        max(max(top_k_values), 20),
    )
    verifier_by_topk = run_verifier_topk(
        claims,
        retrieval_rows,
        shared_ids,
        args.output,
        args.model or DEFAULT_MODELS,
        top_k_values,
        args.batch_size,
        args.max_length,
    )

    selected_verifier, verifier_sweep = calibrate_verifier(verifier_by_topk, calibration_ids, thresholds)
    selected_fusion, fusion_sweep = calibrate_fusion(kg_rankings, verifier_by_topk, calibration_ids, thresholds, alphas)

    verifier_eval = thresholded_verifier_rows(
        verifier_by_topk[int(selected_verifier["top_k"])],
        evaluation_ids,
        float(selected_verifier["threshold"]),
    )
    kg_eval = kg_rows(kg_rankings, evaluation_ids)
    fusion_eval = fuse(
        kg_rankings,
        verifier_by_topk[int(selected_fusion["top_k"])],
        evaluation_ids,
        float(selected_fusion["alpha"]),
        float(selected_fusion["threshold"]),
    )
    curve = alpha_curve(
        kg_rankings,
        verifier_by_topk[int(selected_fusion["top_k"])],
        calibration_ids,
        evaluation_ids,
        float(selected_fusion["threshold"]),
        alphas,
    )

    evaluation_results = {
        "strong_verifier_only_calibrated": evaluate_predictions(list(verifier_eval.values())),
        "kg_scorer_only": evaluate_predictions(list(kg_eval.values())),
        "kg_strong_verifier_fusion_calibrated": evaluate_predictions(list(fusion_eval.values())),
    }
    significance = {
        "fusion_vs_verifier": paired_bootstrap(fusion_eval, verifier_eval, args.bootstrap_samples, args.split_seed),
        "fusion_vs_kg": paired_bootstrap(fusion_eval, kg_eval, args.bootstrap_samples, args.split_seed + 1),
    }
    payload = {
        "split_seed": args.split_seed,
        "calibration_ratio": args.calibration_ratio,
        "calibration_groups": len(calibration_ids),
        "evaluation_groups": len(evaluation_ids),
        "top_k_values": top_k_values,
        "thresholds": thresholds,
        "alphas": alphas,
        "selected_verifier": selected_verifier,
        "selected_fusion": selected_fusion,
        "verifier_calibration_sweep": verifier_sweep,
        "fusion_calibration_sweep": fusion_sweep,
        "alpha_curve": curve,
        "evaluation_results": evaluation_results,
        "confusion_matrices": {
            "strong_verifier_only_calibrated": confusion_matrix(list(verifier_eval.values())),
            "kg_scorer_only": confusion_matrix(list(kg_eval.values())),
            "kg_strong_verifier_fusion_calibrated": confusion_matrix(list(fusion_eval.values())),
        },
        "significance": significance,
    }
    write_json(args.output / "metrics.json", payload)
    write_jsonl(
        args.output / "predictions.jsonl",
        (
            {"method": method, **row, "correct": row["gold_label"] == row["predicted_label"]}
            for method, rows in [
                ("strong_verifier_only_calibrated", verifier_eval),
                ("kg_scorer_only", kg_eval),
                ("kg_strong_verifier_fusion_calibrated", fusion_eval),
            ]
            for row in rows.values()
        ),
    )
    write_summary(args.output / "summary.md", payload)
    print((args.output / "summary.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
