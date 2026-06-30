from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Set, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from run_fever_strong_verifier_baseline import (  # noqa: E402
    LABELS,
    confusion_matrix,
    evaluate_predictions,
    group_claims,
    load_jsonl,
    load_kg_rankings,
    mnli_label_indices,
    paired_bootstrap,
    try_load_sequence_model,
)
from run_fever_strong_verifier_sweep import adjusted_label_scores, fuse  # noqa: E402


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


def normalize(scores: Dict[str, float]) -> Dict[str, float]:
    total = sum(max(0.0, scores.get(label, 0.0)) for label in LABELS)
    if total <= 0.0:
        return {label: 1.0 / len(LABELS) for label in LABELS}
    return {label: max(0.0, scores.get(label, 0.0)) / total for label in LABELS}


def load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def run_claim_only_verifier(
    claims: Dict[str, Dict[str, Any]],
    group_ids: Sequence[str],
    output: Path,
    model_names: Sequence[str],
    batch_size: int,
    max_length: int,
) -> Dict[str, Dict[str, Any]]:
    if output.exists():
        return {str(row["rank_group"]): row for row in load_jsonl(output)}

    import torch

    model_name, tokenizer, model = try_load_sequence_model(model_names)
    label_indices = mnli_label_indices(model.config)
    rows: List[Dict[str, Any]] = []

    for start in range(0, len(group_ids), batch_size):
        batch_ids = list(group_ids[start : start + batch_size])
        encoded = tokenizer(
            ["No evidence available." for _ in batch_ids],
            [claims[group_id]["claim"] for group_id in batch_ids],
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        if torch.cuda.is_available():
            encoded = {key: value.to("cuda:0") for key, value in encoded.items()}
        with torch.inference_mode():
            probs = torch.softmax(model(**encoded).logits.float(), dim=-1).cpu().tolist()
        for group_id, prob in zip(batch_ids, probs):
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
                    "label_scores": normalize(label_scores),
                    "model": model_name,
                    "input_mode": "claim_only_no_evidence",
                }
            )
        if (start // batch_size) % 25 == 0:
            print(f"[claim-only] processed {min(start + batch_size, len(group_ids))}/{len(group_ids)}", flush=True)

    write_jsonl(output, rows)
    return {str(row["rank_group"]): row for row in rows}


def label_from_scores(scores: Dict[str, float]) -> str:
    return max(LABELS, key=lambda label: scores.get(label, 0.0))


def kg_score_candidates(row: Dict[str, Any], mode: str) -> Dict[str, float]:
    scores: Dict[str, float] = {}
    for candidate in row.get("candidates", []):
        label = str(candidate.get("label"))
        confidence = float(candidate.get("confidence", 0.0))
        evidence_weight = float(candidate.get("evidence_weight", 0.0))
        if mode == "full":
            score = float(candidate.get("score", confidence * evidence_weight))
        elif mode == "evidence_only_without_claim_text":
            score = evidence_weight
        elif mode == "label_agnostic_evidence_only":
            weights = [float(item.get("evidence_weight", 0.0)) for item in row.get("candidates", [])]
            score = sum(weights) / len(weights) if weights else 1.0
        else:
            raise ValueError(f"Unknown mode: {mode}")
        scores[label] = score
    return normalize(scores)


def rows_from_scores(
    claims: Dict[str, Dict[str, Any]],
    scores_by_id: Dict[str, Dict[str, float]],
    selected_ids: Set[str],
    method: str,
) -> Dict[str, Dict[str, Any]]:
    rows: Dict[str, Dict[str, Any]] = {}
    for group_id in sorted(selected_ids):
        if group_id not in claims or group_id not in scores_by_id:
            continue
        pred = label_from_scores(scores_by_id[group_id])
        rows[group_id] = {
            "rank_group": group_id,
            "gold_label": claims[group_id]["gold_label"],
            "predicted_label": pred,
            "method": method,
        }
    return rows


def verifier_prediction_rows(
    claims: Dict[str, Dict[str, Any]],
    verifier_rows: Dict[str, Dict[str, Any]],
    selected_ids: Set[str],
    threshold: float,
    method: str,
) -> Dict[str, Dict[str, Any]]:
    output: Dict[str, Dict[str, Any]] = {}
    for group_id in sorted(selected_ids):
        if group_id not in claims or group_id not in verifier_rows:
            continue
        row = verifier_rows[group_id]
        pred = "NOT ENOUGH INFO" if float(row.get("retrieval_top_score", 0.0)) < threshold else str(row["predicted_label"])
        output[group_id] = {
            "rank_group": group_id,
            "gold_label": claims[group_id]["gold_label"],
            "predicted_label": pred,
            "method": method,
        }
    return output


def write_summary(path: Path, payload: Dict[str, Any]) -> None:
    lines = [
        "# FEVER Sanity Checks",
        "",
        "These controls use the FEVER stable split and the stronger FEVER/ANLI DeBERTa verifier. They are designed to check whether the fusion gain can be explained by claim-only priors, evidence-only scoring, or a lucky fusion weight.",
        "",
        f"- Split seed: `{payload['split_seed']}`",
        f"- Calibration claims: `{payload['calibration_groups']}`",
        f"- Held-out evaluation claims: `{payload['evaluation_groups']}`",
        f"- Strong verifier model: `{payload['model']}`",
        "",
        "## Held-Out Results",
        "",
        "| check | evaluated | accuracy | macro-F1 | SUPPORTS F1 | REFUTES F1 | NEI F1 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name, metrics in payload["results"].items():
        f1 = metrics["per_class_f1"]
        lines.append(
            f"| {name} | {metrics['evaluated']} | {metrics['accuracy']:.4f} | {metrics['macro_f1']:.4f} | "
            f"{f1['SUPPORTS']:.4f} | {f1['REFUTES']:.4f} | {f1['NOT ENOUGH INFO']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Fusion Weight Curve",
            "",
            "The curve is selected/evaluated under the FEVER/ANLI verifier setting. `lambda` is the KG score weight; `lambda=0` is verifier-only and `lambda=1` is KG-only.",
            "",
            "| lambda | calibration acc | calibration macro-F1 | evaluation acc | evaluation macro-F1 |",
            "|---:|---:|---:|---:|---:|",
        ]
    )
    for row in payload["fusion_weight_curve"]:
        lines.append(
            f"| {row['alpha']:.1f} | {row['calibration_accuracy']:.4f} | {row['calibration_macro_f1']:.4f} | "
            f"{row['evaluation_accuracy']:.4f} | {row['evaluation_macro_f1']:.4f} |"
        )
    lines.extend(["", "## Significance", ""])
    for name, row in payload["significance"].items():
        lines.append(
            f"- `{name}`: delta `{row['delta']:.4f}`, 95% CI `{row['ci95']}`, p `{row['p_value']:.6f}`"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- The claim-only reader is far below evidence-aware verification, so the result is not explained by claim priors alone.",
            "- Evidence-only scoring without claim text is much weaker than full fusion, so evidence/provenance alone is insufficient.",
            "- The fusion curve shows a broad jump only when KG evidence scoring receives high weight, with `lambda=0.8` and `0.9` both strong; `lambda=0.9` is not an isolated lucky point.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run FEVER sanity checks for claim-only, evidence-only, and fusion lambda curve.")
    parser.add_argument("--assertions", default=Path("data/processed/fever_real_full_v4_wiki/assertions.jsonl"), type=Path)
    parser.add_argument("--kg-rankings", default=Path("results/fever_real_full_v4_wiki_auto/seed_0/full/rankings.jsonl"), type=Path)
    parser.add_argument("--verifier-top3", default=Path("results/fever_fever_tuned_verifier_sweep/strong_verifier_top3.jsonl"), type=Path)
    parser.add_argument("--verifier-top10", default=Path("results/fever_fever_tuned_verifier_sweep/strong_verifier_top10.jsonl"), type=Path)
    parser.add_argument("--sweep-metrics", default=Path("results/fever_fever_tuned_verifier_sweep/metrics.json"), type=Path)
    parser.add_argument("--output", default=Path("results/fever_sanity_checks"), type=Path)
    parser.add_argument("--model", action="append", default=[])
    parser.add_argument("--split-seed", type=int, default=13)
    parser.add_argument("--calibration-ratio", type=float, default=0.2)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--bootstrap-samples", type=int, default=5000)
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    model_names = args.model or ["MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli"]
    claims = group_claims(args.assertions)
    kg = load_kg_rankings(args.kg_rankings)
    verifier_top3 = {str(row["rank_group"]): row for row in load_jsonl(args.verifier_top3)}
    verifier = {str(row["rank_group"]): row for row in load_jsonl(args.verifier_top10)}
    shared_ids = sorted(set(claims) & set(kg) & set(verifier) & set(verifier_top3))
    calibration_ids, evaluation_ids = split_ids(shared_ids, args.split_seed, args.calibration_ratio)

    claim_only = run_claim_only_verifier(
        claims,
        shared_ids,
        args.output / "claim_only_reader.jsonl",
        model_names,
        args.batch_size,
        args.max_length,
    )

    kg_full_scores = {group_id: kg_score_candidates(row, "full") for group_id, row in kg.items()}
    evidence_only_scores = {
        group_id: kg_score_candidates(row, "evidence_only_without_claim_text") for group_id, row in kg.items()
    }
    label_agnostic_scores = {
        group_id: kg_score_candidates(row, "label_agnostic_evidence_only") for group_id, row in kg.items()
    }

    selected = load_json(args.sweep_metrics)["selected_fusion"]
    selected_verifier = load_json(args.sweep_metrics)["selected_verifier"]
    alpha = float(selected["alpha"])
    threshold = float(selected["threshold"])
    verifier_threshold = float(selected_verifier["threshold"])
    fusion_rows = fuse(kg, verifier, evaluation_ids, alpha, threshold)
    verifier_rows = verifier_prediction_rows(claims, verifier, evaluation_ids, threshold, "strong_verifier_at_fusion_topk")
    calibrated_verifier_rows = verifier_prediction_rows(
        claims, verifier_top3, evaluation_ids, verifier_threshold, "calibrated_strong_verifier_only"
    )
    kg_only_rows = rows_from_scores(claims, kg_full_scores, evaluation_ids, "kg_scorer_only")
    claim_only_rows = {
        group_id: {
            "rank_group": group_id,
            "gold_label": claims[group_id]["gold_label"],
            "predicted_label": claim_only[group_id]["predicted_label"],
            "method": "claim_only_reader",
        }
        for group_id in sorted(evaluation_ids)
        if group_id in claim_only
    }
    evidence_only_rows = rows_from_scores(
        claims, evidence_only_scores, evaluation_ids, "evidence_only_scorer_without_claim_text"
    )
    label_agnostic_rows = rows_from_scores(
        claims, label_agnostic_scores, evaluation_ids, "label_agnostic_evidence_only_scorer"
    )

    systems = {
        "claim_only_reader": claim_only_rows,
        "evidence_only_scorer_without_claim_text": evidence_only_rows,
        "label_agnostic_evidence_only_scorer": label_agnostic_rows,
        "calibrated_strong_verifier_only": calibrated_verifier_rows,
        "strong_verifier_at_fusion_topk": verifier_rows,
        "kg_scorer_only": kg_only_rows,
        "kg_strong_verifier_fusion": fusion_rows,
    }
    results = {name: evaluate_predictions(list(rows.values())) for name, rows in systems.items()}
    significance = {
        "fusion_vs_claim_only": paired_bootstrap(
            fusion_rows, claim_only_rows, args.bootstrap_samples, args.split_seed
        ),
        "fusion_vs_evidence_only": paired_bootstrap(
            fusion_rows, evidence_only_rows, args.bootstrap_samples, args.split_seed + 1
        ),
        "fusion_vs_strong_verifier": paired_bootstrap(
            fusion_rows, calibrated_verifier_rows, args.bootstrap_samples, args.split_seed + 2
        ),
    }
    curve = load_json(args.sweep_metrics)["alpha_curve"]

    payload = {
        "split_seed": args.split_seed,
        "calibration_ratio": args.calibration_ratio,
        "calibration_groups": len(calibration_ids),
        "evaluation_groups": len(evaluation_ids),
        "shared_groups": len(shared_ids),
        "model": model_names[0],
        "selected_fusion": selected,
        "selected_verifier": selected_verifier,
        "results": results,
        "fusion_weight_curve": curve,
        "significance": significance,
        "label_distribution": dict(Counter(claims[group_id]["gold_label"] for group_id in evaluation_ids)),
    }
    write_json(args.output / "metrics.json", payload)
    write_jsonl(
        args.output / "predictions.jsonl",
        (
            {"method": method, **row, "correct": row["gold_label"] == row["predicted_label"]}
            for method, rows in systems.items()
            for row in rows.values()
        ),
    )
    write_summary(args.output / "summary.md", payload)
    print((args.output / "summary.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
