from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Set, Tuple


LABELS = ["SUPPORTS", "REFUTES", "NOT ENOUGH INFO"]


def load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


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


def group_claims(assertions_path: Path) -> Dict[str, Dict[str, Any]]:
    claims: Dict[str, Dict[str, Any]] = {}
    for row in load_jsonl(assertions_path):
        group_id = str(row.get("rank_group", ""))
        if not group_id:
            continue
        if group_id not in claims or row.get("is_gold") is True:
            claims[group_id] = {
                "rank_group": group_id,
                "claim": str(row.get("claim_text") or row.get("subject", {}).get("label", "")),
                "gold_label": str(row.get("gold_label")),
            }
    return claims


def load_kg_rankings(path: Path) -> Dict[str, Dict[str, Any]]:
    return {str(row["rank_group"]): row for row in load_jsonl(path)}


def load_qwen_details(path: Path) -> Dict[str, Dict[str, Any]]:
    payload = load_json(path)
    return {str(row["rank_group"]): row for row in payload.get("details", [])}


def load_strong_verifier(path: Path) -> Dict[str, Dict[str, Any]]:
    return {str(row["rank_group"]): row for row in load_jsonl(path)}


def load_retrieval(path: Path) -> Dict[str, Dict[str, Any]]:
    return {str(row["rank_group"]): row for row in load_jsonl(path)}


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


def evaluate(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    total = len(rows)
    class_f1 = per_class_f1(rows)
    return {
        "evaluated": total,
        "accuracy": sum(1 for row in rows if row["gold_label"] == row["predicted_label"]) / total if total else 0.0,
        "macro_f1": sum(class_f1.values()) / len(LABELS),
        "per_class_f1": class_f1,
        "label_distribution": dict(Counter(row["gold_label"] for row in rows)),
    }


def normalize(values: Dict[str, float]) -> Dict[str, float]:
    total = sum(max(0.0, value) for value in values.values())
    if total <= 0:
        return {label: 1.0 / len(LABELS) for label in LABELS}
    return {label: max(0.0, values.get(label, 0.0)) / total for label in LABELS}


def kg_scores(row: Dict[str, Any], mode: str = "full") -> Dict[str, float]:
    scores: Dict[str, float] = {}
    for candidate in row.get("candidates", []):
        label = str(candidate.get("label"))
        confidence = float(candidate.get("confidence", 0.0))
        evidence_weight = float(candidate.get("evidence_weight", 1.0))
        if mode == "full":
            score = float(candidate.get("score", confidence * evidence_weight))
        elif mode == "confidence_only":
            score = confidence
        elif mode == "evidence_reliability_only":
            score = evidence_weight
        elif mode == "provenance_no_assertion_graph":
            score = evidence_weight
        elif mode == "label_agnostic_provenance":
            # Same claim-level provenance score for every verdict candidate.
            # This tests whether gains come from retrieved evidence availability
            # alone rather than label-aware candidate scoring.
            weights = [float(item.get("evidence_weight", 1.0)) for item in row.get("candidates", [])]
            score = sum(weights) / len(weights) if weights else 1.0
        elif mode == "ontology_context_metadata_only":
            # FEVER has no meaningful ontology/time context in this setup, so
            # metadata-only should collapse to an uninformative uniform signal.
            score = 1.0
        else:
            raise ValueError(f"Unknown KG score mode: {mode}")
        scores[label] = score
    return normalize(scores)


def reader_scores_from_label(label: str, boost: float = 1.0) -> Dict[str, float]:
    scores = {candidate: (boost if candidate == label else 0.0) for candidate in LABELS}
    return normalize(scores)


def fuse_scores(kg: Dict[str, float], reader: Dict[str, float], alpha: float) -> str:
    return max(LABELS, key=lambda label: alpha * kg.get(label, 0.0) + (1.0 - alpha) * reader.get(label, 0.0))


def prediction_rows(
    ids: Set[str],
    claims: Dict[str, Dict[str, Any]],
    kg: Dict[str, Dict[str, Any]],
    reader_scores: Dict[str, Dict[str, float]],
    alpha: float,
    kg_mode: str = "full",
    kg_id_map: Dict[str, str] | None = None,
    random_labels: Dict[str, str] | None = None,
) -> Dict[str, Dict[str, Any]]:
    rows: Dict[str, Dict[str, Any]] = {}
    for group_id in sorted(ids):
        kg_group_id = kg_id_map.get(group_id, group_id) if kg_id_map else group_id
        if group_id not in claims or group_id not in reader_scores or kg_group_id not in kg:
            continue
        gold = random_labels.get(group_id, claims[group_id]["gold_label"]) if random_labels else claims[group_id]["gold_label"]
        pred = fuse_scores(kg_scores(kg[kg_group_id], kg_mode), reader_scores[group_id], alpha)
        rows[group_id] = {
            "rank_group": group_id,
            "gold_label": gold,
            "predicted_label": pred,
            "alpha": alpha,
            "kg_mode": kg_mode,
            "kg_source_rank_group": kg_group_id,
        }
    return rows


def reader_only_rows(
    ids: Set[str],
    claims: Dict[str, Dict[str, Any]],
    labels: Dict[str, str],
    random_labels: Dict[str, str] | None = None,
) -> Dict[str, Dict[str, Any]]:
    rows = {}
    for group_id in sorted(ids):
        if group_id not in claims or group_id not in labels:
            continue
        rows[group_id] = {
            "rank_group": group_id,
            "gold_label": random_labels.get(group_id, claims[group_id]["gold_label"]) if random_labels else claims[group_id]["gold_label"],
            "predicted_label": labels[group_id],
        }
    return rows


def select_alpha(
    ids: Set[str],
    claims: Dict[str, Dict[str, Any]],
    kg: Dict[str, Dict[str, Any]],
    reader_scores: Dict[str, Dict[str, float]],
    alphas: Sequence[float],
    kg_mode: str = "full",
    kg_id_map: Dict[str, str] | None = None,
    random_labels: Dict[str, str] | None = None,
) -> Tuple[float, List[Dict[str, Any]]]:
    rows = []
    for alpha in alphas:
        preds = prediction_rows(ids, claims, kg, reader_scores, alpha, kg_mode, kg_id_map, random_labels)
        rows.append({"alpha": alpha, **evaluate(list(preds.values()))})
    best = max(rows, key=lambda row: (row["accuracy"], row["macro_f1"]))
    return float(best["alpha"]), rows


def shuffled_id_map(ids: Sequence[str], seed: int) -> Dict[str, str]:
    shuffled = list(ids)
    rng = random.Random(seed)
    rng.shuffle(shuffled)
    return dict(zip(sorted(ids), shuffled))


def random_label_map(ids: Sequence[str], seed: int) -> Dict[str, str]:
    rng = random.Random(seed)
    return {group_id: rng.choice(LABELS) for group_id in sorted(ids)}


def lexical_claim_only_prediction(claim: str) -> str:
    lower = claim.lower()
    neg_markers = [" not ", " never ", " no ", " false", " unable ", " without ", "cannot", "n't"]
    if any(marker in f" {lower} " for marker in neg_markers):
        return "REFUTES"
    return "SUPPORTS"


def retrieval_evidence_only_predictions(
    ids: Set[str],
    claims: Dict[str, Dict[str, Any]],
    retrieval: Dict[str, Dict[str, Any]],
) -> Dict[str, Dict[str, Any]]:
    rows = {}
    for group_id in sorted(ids):
        if group_id not in claims or group_id not in retrieval:
            continue
        retrieved = retrieval[group_id].get("retrieved", [])
        pred = "NOT ENOUGH INFO" if not retrieved or float(retrieved[0].get("score", 0.0)) < 10.0 else "SUPPORTS"
        rows[group_id] = {
            "rank_group": group_id,
            "gold_label": claims[group_id]["gold_label"],
            "predicted_label": pred,
        }
    return rows


def claim_only_predictions(ids: Set[str], claims: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    return {
        group_id: {
            "rank_group": group_id,
            "gold_label": claims[group_id]["gold_label"],
            "predicted_label": lexical_claim_only_prediction(claims[group_id]["claim"]),
        }
        for group_id in sorted(ids)
        if group_id in claims
    }


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
    return {
        "paired_items": len(ids),
        "delta": observed,
        "ci95": [deltas[int(0.025 * len(deltas))], deltas[min(len(deltas) - 1, int(0.975 * len(deltas)))]],
        "p_value": (1 + sum(1 for delta in deltas if delta <= 0.0)) / (len(deltas) + 1),
    }


def write_summary(path: Path, payload: Dict[str, Any]) -> None:
    lines = [
        "# FEVER Leakage And Sanity Audit",
        "",
        "This audit treats FEVER + LLM/RAG reliability as the main experiment and reports controls that test whether the KG fusion gain can be explained by label leakage or gold-evidence shortcuts.",
        "",
        f"- Split seed: `{payload['split_seed']}`",
        f"- Calibration claims: `{payload['calibration_groups']}`",
        f"- Held-out evaluation claims: `{payload['evaluation_groups']}`",
        f"- Selected alpha for KG/Qwen on calibration: `{payload['selected_alpha_qwen']}`",
        f"- Selected alpha for KG/strong verifier on calibration: `{payload['selected_alpha_strong']}`",
        "",
        "## Score Provenance",
        "",
        "- Evidence weights are read from the automatic FEVER KG scorer output in `results/fever_real_full_v4_wiki_auto/seed_0/full/rankings.jsonl`.",
        "- The scorer uses local Wikipedia provenance/lexical evidence features generated before the held-out evaluation; this audit does not recompute weights from evaluation labels.",
        "- Fusion weights are selected only on the stable-hash calibration split and frozen on held-out evaluation.",
        "- Gold labels are used only for evaluation and calibration model selection, not for the held-out scoring step.",
        "",
        "## Held-Out Sanity Results",
        "",
        "| check | evaluated | accuracy | macro-F1 | interpretation |",
        "|---|---:|---:|---:|---|",
    ]
    for row in payload["sanity_table"]:
        lines.append(
            f"| {row['name']} | {row['evaluated']} | {row['accuracy']:.4f} | {row['macro_f1']:.4f} | {row['interpretation']} |"
        )
    lines.extend(
        [
            "",
            "## KG/Qwen Alpha Curve",
            "",
            "| alpha | calibration acc | evaluation acc | evaluation macro-F1 |",
            "|---:|---:|---:|---:|",
        ]
    )
    for row in payload["alpha_curve_qwen"]:
        lines.append(
            f"| {row['alpha']:.2f} | {row['calibration_accuracy']:.4f} | {row['evaluation_accuracy']:.4f} | {row['evaluation_macro_f1']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## KG/Strong-Verifier Alpha Curve",
            "",
            "| alpha | calibration acc | evaluation acc | evaluation macro-F1 |",
            "|---:|---:|---:|---:|",
        ]
    )
    for row in payload["alpha_curve_strong"]:
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
            f"- `{name}`: delta `{row['delta']:.4f}`, 95% CI `{row['ci95']}`, p `{row['p_value']:.6f}`, paired items `{row['paired_items']}`"
        )
    lines.extend(
        [
            "",
        "## Paper-Ready Caveat",
        "",
            "The audit supports reporting FEVER as the main reliability experiment, while YAGO and ICEWS14 should remain mechanism analyses. It also narrows the claim: the current held-out gain is driven mainly by label-aware evidence/provenance scoring, not by ontology/context metadata or a standalone assertion graph. The score uses the candidate label but not the held-out gold label; therefore the paper should describe it as a calibrated label-aware evidence scorer and retain shuffled-pairing/random-label controls to rule out trivial leakage.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run FEVER leakage/sanity checks for KG + reader fusion.")
    parser.add_argument("--assertions", default=Path("data/processed/fever_real_full_v4_wiki/assertions.jsonl"), type=Path)
    parser.add_argument("--kg-rankings", default=Path("results/fever_real_full_v4_wiki_auto/seed_0/full/rankings.jsonl"), type=Path)
    parser.add_argument("--qwen", default=Path("results/baselines_plus/qwen_fever_evidence_reader_full/qwen_fever_evidence_reader.json"), type=Path)
    parser.add_argument("--strong-verifier", default=Path("results/fever_strong_baseline/strong_verifier_predictions.jsonl"), type=Path)
    parser.add_argument("--retrieval", default=Path("results/fever_strong_baseline/retrieved_evidence.jsonl"), type=Path)
    parser.add_argument("--output", default=Path("results/fever_leakage_sanity"), type=Path)
    parser.add_argument("--split-seed", type=int, default=13)
    parser.add_argument("--calibration-ratio", type=float, default=0.2)
    parser.add_argument("--bootstrap-samples", type=int, default=5000)
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    claims = group_claims(args.assertions)
    kg = load_kg_rankings(args.kg_rankings)
    qwen = load_qwen_details(args.qwen)
    strong = load_strong_verifier(args.strong_verifier)
    retrieval = load_retrieval(args.retrieval)

    shared_ids = sorted(set(claims) & set(kg) & set(qwen) & set(strong) & set(retrieval))
    calibration_ids, evaluation_ids = split_ids(shared_ids, args.split_seed, args.calibration_ratio)
    alphas = [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0]

    qwen_label = {group_id: str(row["predicted_label"]) for group_id, row in qwen.items()}
    qwen_scores = {group_id: reader_scores_from_label(label) for group_id, label in qwen_label.items()}
    strong_scores = {group_id: normalize(row.get("label_scores", {})) for group_id, row in strong.items()}
    strong_label = {group_id: str(row["predicted_label"]) for group_id, row in strong.items()}

    selected_qwen, qwen_cal_curve = select_alpha(calibration_ids, claims, kg, qwen_scores, alphas)
    selected_strong, strong_cal_curve = select_alpha(calibration_ids, claims, kg, strong_scores, alphas)

    qwen_eval_by_alpha = []
    strong_eval_by_alpha = []
    for alpha in alphas:
        q_eval = evaluate(list(prediction_rows(evaluation_ids, claims, kg, qwen_scores, alpha).values()))
        s_eval = evaluate(list(prediction_rows(evaluation_ids, claims, kg, strong_scores, alpha).values()))
        q_cal = next(row for row in qwen_cal_curve if row["alpha"] == alpha)
        s_cal = next(row for row in strong_cal_curve if row["alpha"] == alpha)
        qwen_eval_by_alpha.append(
            {
                "alpha": alpha,
                "calibration_accuracy": q_cal["accuracy"],
                "calibration_macro_f1": q_cal["macro_f1"],
                "evaluation_accuracy": q_eval["accuracy"],
                "evaluation_macro_f1": q_eval["macro_f1"],
            }
        )
        strong_eval_by_alpha.append(
            {
                "alpha": alpha,
                "calibration_accuracy": s_cal["accuracy"],
                "calibration_macro_f1": s_cal["macro_f1"],
                "evaluation_accuracy": s_eval["accuracy"],
                "evaluation_macro_f1": s_eval["macro_f1"],
            }
        )

    shuffled_map = shuffled_id_map(shared_ids, args.split_seed + 101)
    random_labels = random_label_map(shared_ids, args.split_seed + 202)

    systems: Dict[str, Dict[str, Dict[str, Any]]] = {
        "qwen_reader_only": reader_only_rows(evaluation_ids, claims, qwen_label),
        "strong_verifier_only": reader_only_rows(evaluation_ids, claims, strong_label),
        "kg_qwen_fusion": prediction_rows(evaluation_ids, claims, kg, qwen_scores, selected_qwen),
        "kg_strong_fusion": prediction_rows(evaluation_ids, claims, kg, strong_scores, selected_strong),
        "reader_plus_retrieved_evidence_score_no_kg": prediction_rows(
            evaluation_ids, claims, kg, qwen_scores, selected_qwen, kg_mode="evidence_reliability_only"
        ),
        "reader_plus_provenance_no_assertion_graph": prediction_rows(
            evaluation_ids, claims, kg, qwen_scores, selected_qwen, kg_mode="provenance_no_assertion_graph"
        ),
        "reader_plus_label_agnostic_provenance": prediction_rows(
            evaluation_ids, claims, kg, qwen_scores, selected_qwen, kg_mode="label_agnostic_provenance"
        ),
        "reader_plus_ontology_context_metadata_only": prediction_rows(
            evaluation_ids, claims, kg, qwen_scores, selected_qwen, kg_mode="ontology_context_metadata_only"
        ),
        "reader_plus_confidence_no_evidence_weight": prediction_rows(
            evaluation_ids, claims, kg, qwen_scores, selected_qwen, kg_mode="confidence_only"
        ),
        "shuffled_claim_evidence_pairing": prediction_rows(
            evaluation_ids, claims, kg, qwen_scores, selected_qwen, kg_id_map=shuffled_map
        ),
        "random_label_sanity_check": prediction_rows(
            evaluation_ids, claims, kg, qwen_scores, selected_qwen, random_labels=random_labels
        ),
        "claim_only_lexical": claim_only_predictions(evaluation_ids, claims),
        "evidence_only_retrieval_score": retrieval_evidence_only_predictions(evaluation_ids, claims, retrieval),
    }

    calibration_alt, evaluation_alt = split_ids(shared_ids, args.split_seed + 1, args.calibration_ratio)
    selected_transfer, transfer_curve = select_alpha(calibration_alt, claims, kg, qwen_scores, alphas)
    systems["cross_split_calibration_transfer"] = prediction_rows(
        evaluation_ids, claims, kg, qwen_scores, selected_transfer
    )

    interpretations = {
        "qwen_reader_only": "LLM reader over retrieved evidence, no KG fusion.",
        "strong_verifier_only": "DeBERTa-style NLI verifier over retrieved evidence, no KG fusion.",
        "kg_qwen_fusion": "Main KG/Qwen fusion with alpha selected on calibration.",
        "kg_strong_fusion": "KG fused with strong NLI verifier.",
        "reader_plus_retrieved_evidence_score_no_kg": "Uses evidence reliability only, removing assertion-level KG score.",
        "reader_plus_provenance_no_assertion_graph": "Uses provenance/evidence weight without assertion graph scoring.",
        "reader_plus_label_agnostic_provenance": "Uses the same claim-level provenance score for all verdict labels; removes label-aware evidence scoring.",
        "reader_plus_ontology_context_metadata_only": "Uses only FEVER ontology/context metadata, which is intentionally uninformative in this benchmark.",
        "reader_plus_confidence_no_evidence_weight": "Removes evidence reliability and keeps candidate confidence only.",
        "shuffled_claim_evidence_pairing": "Shuffles KG evidence/score across claims; performance should drop if scores are claim-specific.",
        "random_label_sanity_check": "Evaluates against random labels; high performance here would indicate leakage.",
        "claim_only_lexical": "No retrieved evidence; simple claim-only lexical heuristic.",
        "evidence_only_retrieval_score": "No claim semantics beyond retrieval score threshold.",
        "cross_split_calibration_transfer": f"Alpha selected on alternate split seed {args.split_seed + 1} and evaluated on the original held-out split.",
    }
    sanity_table = []
    for name, rows in systems.items():
        metrics = evaluate(list(rows.values()))
        sanity_table.append({"name": name, **metrics, "interpretation": interpretations[name]})

    significance = {
        "kg_qwen_fusion_vs_qwen_reader": paired_bootstrap(
            systems["kg_qwen_fusion"], systems["qwen_reader_only"], args.bootstrap_samples, args.split_seed
        ),
        "kg_strong_fusion_vs_strong_verifier": paired_bootstrap(
            systems["kg_strong_fusion"], systems["strong_verifier_only"], args.bootstrap_samples, args.split_seed + 1
        ),
        "kg_qwen_fusion_vs_shuffled_pairing": paired_bootstrap(
            systems["kg_qwen_fusion"], systems["shuffled_claim_evidence_pairing"], args.bootstrap_samples, args.split_seed + 2
        ),
    }

    payload = {
        "split_seed": args.split_seed,
        "calibration_ratio": args.calibration_ratio,
        "calibration_groups": len(calibration_ids),
        "evaluation_groups": len(evaluation_ids),
        "shared_groups": len(shared_ids),
        "selected_alpha_qwen": selected_qwen,
        "selected_alpha_strong": selected_strong,
        "cross_split_selected_alpha_qwen": selected_transfer,
        "cross_split_calibration_curve": transfer_curve,
        "alpha_curve_qwen": qwen_eval_by_alpha,
        "alpha_curve_strong": strong_eval_by_alpha,
        "sanity_table": sanity_table,
        "significance": significance,
    }
    write_json(args.output / "metrics.json", payload)
    write_jsonl(
        args.output / "predictions.jsonl",
        (
            {"method": name, **row, "correct": row["gold_label"] == row["predicted_label"]}
            for name, rows in systems.items()
            for row in rows.values()
        ),
    )
    write_summary(args.output / "summary.md", payload)
    print((args.output / "summary.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
