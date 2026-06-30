from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from run_fever_strong_verifier_baseline import (  # noqa: E402
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


def run_verifier_topk(
    claims: Dict[str, Dict[str, Any]],
    retrieval_rows: Dict[str, Dict[str, Any]],
    group_ids: Sequence[str],
    output_dir: Path,
    split_name: str,
    model_names: Sequence[str],
    top_k_values: Sequence[int],
    batch_size: int,
    max_length: int,
) -> Dict[int, Dict[str, Dict[str, Any]]]:
    existing: Dict[int, Dict[str, Dict[str, Any]]] = {}
    missing: List[int] = []
    for top_k in top_k_values:
        path = output_dir / f"{split_name}_strong_verifier_top{top_k}.jsonl"
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
        pending = []
        for group_id in group_ids:
            retrieved = retrieval_rows[group_id]["retrieved"]
            premise = evidence_text(retrieved, top_k) or "No evidence available."
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
                probs = torch.softmax(model(**encoded).logits.float(), dim=-1).cpu().tolist()
            for (group_id, premise, _, top_score), prob in zip(batch, probs):
                label_scores = {
                    label: float(prob[int(index)]) if index is not None and int(index) < len(prob) else 0.0
                    for label, index in label_indices.items()
                }
                rows.append(
                    {
                        "rank_group": group_id,
                        "gold_label": claims[group_id]["gold_label"],
                        "predicted_label": max(LABELS, key=lambda label: label_scores[label]),
                        "label_scores": label_scores,
                        "model": model_name,
                        "evidence_top_k": top_k,
                        "retrieval_top_score": top_score,
                        "evidence_preview": premise[:500],
                    }
                )
            if (start // batch_size) % 25 == 0:
                print(f"[{split_name} top_k={top_k}] processed {min(start + batch_size, len(pending))}/{len(pending)}", flush=True)
        path = output_dir / f"{split_name}_strong_verifier_top{top_k}.jsonl"
        write_jsonl(path, rows)
        existing[top_k] = {str(row["rank_group"]): row for row in rows}
    return existing


def thresholded_verifier_rows(
    rows: Dict[str, Dict[str, Any]],
    group_ids: Sequence[str],
    threshold: float,
) -> Dict[str, Dict[str, Any]]:
    output: Dict[str, Dict[str, Any]] = {}
    for group_id in sorted(group_ids):
        row = rows[group_id]
        pred = "NOT ENOUGH INFO" if float(row.get("retrieval_top_score", 0.0)) < threshold else str(row["predicted_label"])
        output[group_id] = {
            "rank_group": group_id,
            "gold_label": row["gold_label"],
            "predicted_label": pred,
            "threshold": threshold,
            "top_score": row.get("retrieval_top_score", 0.0),
        }
    return output


def adjusted_label_scores(row: Dict[str, Any], threshold: float) -> Dict[str, float]:
    if float(row.get("retrieval_top_score", 0.0)) < threshold:
        return {"SUPPORTS": 0.0, "REFUTES": 0.0, "NOT ENOUGH INFO": 1.0}
    return {label: float(row.get("label_scores", {}).get(label, 0.0)) for label in LABELS}


def kg_rows(kg_rankings: Dict[str, Dict[str, Any]], group_ids: Sequence[str]) -> Dict[str, Dict[str, Any]]:
    return {
        group_id: {
            "rank_group": group_id,
            "gold_label": kg_rankings[group_id]["gold_label"],
            "predicted_label": kg_rankings[group_id]["top_label"],
        }
        for group_id in sorted(group_ids)
        if group_id in kg_rankings
    }


def fuse(
    kg_rankings: Dict[str, Dict[str, Any]],
    verifier_rows: Dict[str, Dict[str, Any]],
    group_ids: Sequence[str],
    alpha: float,
    threshold: float,
) -> Dict[str, Dict[str, Any]]:
    output: Dict[str, Dict[str, Any]] = {}
    for group_id in sorted(group_ids):
        kg = kg_rankings[group_id]
        verifier_scores = adjusted_label_scores(verifier_rows[group_id], threshold)
        best_label = None
        best_score = -1.0
        for candidate in kg.get("candidates", []):
            label = str(candidate.get("label"))
            score = alpha * float(candidate.get("score", 0.0)) + (1.0 - alpha) * verifier_scores.get(label, 0.0)
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
    group_ids: Sequence[str],
    thresholds: Sequence[float],
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    sweep: List[Dict[str, Any]] = []
    for top_k, rows in sorted(topk_rows.items()):
        for threshold in thresholds:
            preds = thresholded_verifier_rows(rows, group_ids, threshold)
            sweep.append({"top_k": top_k, "threshold": threshold, **evaluate_predictions(list(preds.values()))})
    return max(sweep, key=lambda row: (row["accuracy"], row["macro_f1"])), sweep


def calibrate_fusion(
    kg_rankings: Dict[str, Dict[str, Any]],
    topk_rows: Dict[int, Dict[str, Dict[str, Any]]],
    group_ids: Sequence[str],
    thresholds: Sequence[float],
    alphas: Sequence[float],
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    sweep: List[Dict[str, Any]] = []
    for top_k, rows in sorted(topk_rows.items()):
        for threshold in thresholds:
            for alpha in alphas:
                preds = fuse(kg_rankings, rows, group_ids, alpha, threshold)
                sweep.append(
                    {"top_k": top_k, "threshold": threshold, "alpha": alpha, **evaluate_predictions(list(preds.values()))}
                )
    return max(sweep, key=lambda row: (row["accuracy"], row["macro_f1"])), sweep


def alpha_curve(
    dev_kg: Dict[str, Dict[str, Any]],
    test_kg: Dict[str, Dict[str, Any]],
    dev_verifier: Dict[str, Dict[str, Any]],
    test_verifier: Dict[str, Dict[str, Any]],
    dev_ids: Sequence[str],
    test_ids: Sequence[str],
    threshold: float,
    alphas: Sequence[float],
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for alpha in alphas:
        dev_metrics = evaluate_predictions(list(fuse(dev_kg, dev_verifier, dev_ids, alpha, threshold).values()))
        test_metrics = evaluate_predictions(list(fuse(test_kg, test_verifier, test_ids, alpha, threshold).values()))
        rows.append(
            {
                "alpha": alpha,
                "dev_accuracy": dev_metrics["accuracy"],
                "dev_macro_f1": dev_metrics["macro_f1"],
                "test_accuracy": test_metrics["accuracy"],
                "test_macro_f1": test_metrics["macro_f1"],
            }
        )
    return rows


def write_summary(path: Path, payload: Dict[str, Any]) -> None:
    lines = [
        "# FEVER Dev-to-Test-Style Protocol",
        "",
        "This run uses public FEVER-style files available in this workspace: `paper_dev.jsonl` for calibration and `paper_test.jsonl` for frozen evaluation. It does not use the hidden official FEVER test labels.",
        "",
        f"- Dev calibration claims: `{payload['dev_groups']}`",
        f"- Test-style evaluation claims: `{payload['test_groups']}`",
        f"- Model: `{payload['model']}`",
        f"- Selected verifier: top_k `{payload['selected_verifier']['top_k']}`, threshold `{payload['selected_verifier']['threshold']}`",
        f"- Selected fusion: top_k `{payload['selected_fusion']['top_k']}`, threshold `{payload['selected_fusion']['threshold']}`, alpha `{payload['selected_fusion']['alpha']}`",
        "",
        "## Frozen Test-Style Evaluation",
        "",
        "| method | evaluated | accuracy | macro-F1 | SUPPORTS F1 | REFUTES F1 | NEI F1 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for method, metrics in payload["test_results"].items():
        f1 = metrics["per_class_f1"]
        lines.append(
            f"| {method} | {metrics['evaluated']} | {metrics['accuracy']:.4f} | {metrics['macro_f1']:.4f} | "
            f"{f1['SUPPORTS']:.4f} | {f1['REFUTES']:.4f} | {f1['NOT ENOUGH INFO']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Fusion Weight Curve",
            "",
            "| alpha | dev acc | test acc | test macro-F1 |",
            "|---:|---:|---:|---:|",
        ]
    )
    for row in payload["alpha_curve"]:
        lines.append(f"| {row['alpha']:.1f} | {row['dev_accuracy']:.4f} | {row['test_accuracy']:.4f} | {row['test_macro_f1']:.4f} |")
    lines.extend(["", "## Significance", ""])
    for name, row in payload["significance"].items():
        lines.append(f"- `{name}`: delta `{row['delta']:.4f}`, 95% CI `{row['ci95']}`, p `{row['p_value']:.6f}`")
    lines.extend(
        [
            "",
            "## Paper-Ready Text",
            "",
            payload["paper_text"],
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def rows_for_output(method: str, rows: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [
        {
            "method": method,
            **row,
            "correct": row["gold_label"] == row["predicted_label"],
        }
        for row in rows.values()
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run public FEVER dev-to-test-style verifier/fusion protocol.")
    parser.add_argument("--dev-assertions", default=Path("data/processed/fever_dev_teststyle_v4_wiki/assertions.jsonl"), type=Path)
    parser.add_argument("--test-assertions", default=Path("data/processed/fever_test_teststyle_v4_wiki/assertions.jsonl"), type=Path)
    parser.add_argument("--wiki-index", default=Path("data/processed/fever_wiki_index_dev_test.json"), type=Path)
    parser.add_argument("--dev-kg", default=Path("results/fever_dev_teststyle_kg/seed_0/full/rankings.jsonl"), type=Path)
    parser.add_argument("--test-kg", default=Path("results/fever_test_teststyle_kg/seed_0/full/rankings.jsonl"), type=Path)
    parser.add_argument("--output", default=Path("results/fever_dev_to_teststyle"), type=Path)
    parser.add_argument("--model", action="append", default=[])
    parser.add_argument("--top-k", action="append", type=int, default=[])
    parser.add_argument("--threshold", action="append", type=float, default=[])
    parser.add_argument("--alpha", action="append", type=float, default=[])
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--bootstrap-samples", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=13)
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    top_k_values = args.top_k or [1, 3, 5, 10]
    thresholds = args.threshold or [0.0, 4.0, 8.0, 12.0, 16.0, 20.0, 24.0]
    alphas = args.alpha or [round(i / 10, 1) for i in range(11)]
    model_names = args.model or ["MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli"]

    dev_claims = group_claims(args.dev_assertions)
    test_claims = group_claims(args.test_assertions)
    dev_kg = load_kg_rankings(args.dev_kg)
    test_kg = load_kg_rankings(args.test_kg)
    dev_ids = sorted(set(dev_claims) & set(dev_kg))
    test_ids = sorted(set(test_claims) & set(test_kg))
    dev_retrieval = load_or_build_retrieval({key: dev_claims[key] for key in dev_ids}, args.wiki_index, args.output / "dev_retrieved_evidence.jsonl", max(max(top_k_values), 20))
    test_retrieval = load_or_build_retrieval({key: test_claims[key] for key in test_ids}, args.wiki_index, args.output / "test_retrieved_evidence.jsonl", max(max(top_k_values), 20))

    dev_verifier_by_topk = run_verifier_topk(dev_claims, dev_retrieval, dev_ids, args.output, "dev", model_names, top_k_values, args.batch_size, args.max_length)
    test_verifier_by_topk = run_verifier_topk(test_claims, test_retrieval, test_ids, args.output, "test", model_names, top_k_values, args.batch_size, args.max_length)

    selected_verifier, verifier_sweep = calibrate_verifier(dev_verifier_by_topk, dev_ids, thresholds)
    selected_fusion, fusion_sweep = calibrate_fusion(dev_kg, dev_verifier_by_topk, dev_ids, thresholds, alphas)
    verifier_eval = thresholded_verifier_rows(test_verifier_by_topk[int(selected_verifier["top_k"])], test_ids, float(selected_verifier["threshold"]))
    kg_eval = kg_rows(test_kg, test_ids)
    fusion_eval = fuse(
        test_kg,
        test_verifier_by_topk[int(selected_fusion["top_k"])],
        test_ids,
        float(selected_fusion["alpha"]),
        float(selected_fusion["threshold"]),
    )
    curve = alpha_curve(
        dev_kg,
        test_kg,
        dev_verifier_by_topk[int(selected_fusion["top_k"])],
        test_verifier_by_topk[int(selected_fusion["top_k"])],
        dev_ids,
        test_ids,
        float(selected_fusion["threshold"]),
        alphas,
    )
    test_results = {
        "strong_verifier_only": evaluate_predictions(list(verifier_eval.values())),
        "kg_scorer_only": evaluate_predictions(list(kg_eval.values())),
        "kg_strong_verifier_fusion": evaluate_predictions(list(fusion_eval.values())),
    }
    significance = {
        "fusion_vs_strong_verifier": paired_bootstrap(fusion_eval, verifier_eval, args.bootstrap_samples, args.seed),
        "fusion_vs_kg": paired_bootstrap(fusion_eval, kg_eval, args.bootstrap_samples, args.seed + 1),
    }
    paper_text = (
        "We additionally report a public FEVER dev-to-test-style protocol. All hyperparameters are selected on "
        "`paper_dev.jsonl` and frozen for `paper_test.jsonl`; no hidden official test labels are used. Under this protocol, "
        f"the FEVER/ANLI DeBERTa verifier reaches {test_results['strong_verifier_only']['accuracy']:.4f} accuracy / "
        f"{test_results['strong_verifier_only']['macro_f1']:.4f} macro-F1, while KG+verifier fusion reaches "
        f"{test_results['kg_strong_verifier_fusion']['accuracy']:.4f} / {test_results['kg_strong_verifier_fusion']['macro_f1']:.4f}. "
        f"The paired bootstrap delta is {significance['fusion_vs_strong_verifier']['delta']:.4f}."
    )
    payload = {
        "protocol": "public FEVER dev calibration -> paper_test frozen evaluation; not official hidden test.",
        "dev_groups": len(dev_ids),
        "test_groups": len(test_ids),
        "model": model_names[0],
        "top_k_values": top_k_values,
        "thresholds": thresholds,
        "alphas": alphas,
        "selected_verifier": selected_verifier,
        "selected_fusion": selected_fusion,
        "verifier_dev_sweep": verifier_sweep,
        "fusion_dev_sweep": fusion_sweep,
        "test_results": test_results,
        "confusion_matrices": {name: metrics["confusion_matrix"] for name, metrics in test_results.items()},
        "alpha_curve": curve,
        "significance": significance,
        "paper_text": paper_text,
    }
    write_json(args.output / "metrics.json", payload)
    write_jsonl(
        args.output / "predictions.jsonl",
        rows_for_output("strong_verifier_only", verifier_eval)
        + rows_for_output("kg_scorer_only", kg_eval)
        + rows_for_output("kg_strong_verifier_fusion", fusion_eval),
    )
    write_summary(args.output / "summary.md", payload)
    print((args.output / "summary.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
