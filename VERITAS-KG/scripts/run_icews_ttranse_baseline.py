from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple


Quad = Tuple[str, str, str, str]


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def group_records(records: Sequence[Dict[str, Any]], groups_limit: int) -> Dict[str, List[Dict[str, Any]]]:
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for record in records:
        group = record.get("rank_group")
        if not group:
            continue
        group_id = str(group)
        if group_id not in grouped and len(grouped) >= groups_limit:
            continue
        grouped[group_id].append(record)
    return grouped


def train_quads(records: Sequence[Dict[str, Any]], limit: int) -> List[Quad]:
    quads: List[Quad] = []
    for record in records:
        if record.get("split") != "train":
            continue
        quads.append(
            (
                str(record.get("raw_subject_id")),
                str(record.get("raw_relation_id")),
                str(record.get("raw_object_id")),
                str(record.get("raw_time_start")),
            )
        )
        if limit > 0 and len(quads) >= limit:
            break
    return quads


def rand_vec(rng: random.Random, dim: int) -> List[float]:
    return [rng.uniform(-0.1, 0.1) for _ in range(dim)]


def l2_score(
    ent: Dict[str, List[float]],
    rel: Dict[str, List[float]],
    tim: Dict[str, List[float]],
    quad: Quad,
) -> float:
    s, r, o, t = quad
    sv = ent[s]
    rv = rel[r]
    ov = ent[o]
    tv = tim[t]
    return sum((sv[i] + rv[i] + tv[i] - ov[i]) ** 2 for i in range(len(sv)))


def update(
    ent: Dict[str, List[float]],
    rel: Dict[str, List[float]],
    tim: Dict[str, List[float]],
    pos: Quad,
    neg: Quad,
    lr: float,
    margin: float,
) -> float:
    pos_score = l2_score(ent, rel, tim, pos)
    neg_score = l2_score(ent, rel, tim, neg)
    loss = margin + pos_score - neg_score
    if loss <= 0:
        return 0.0
    for quad, sign in [(pos, 1.0), (neg, -1.0)]:
        s, r, o, t = quad
        for i in range(len(ent[s])):
            diff = ent[s][i] + rel[r][i] + tim[t][i] - ent[o][i]
            grad = 2.0 * diff * sign
            ent[s][i] -= lr * grad
            rel[r][i] -= lr * grad
            tim[t][i] -= lr * grad
            ent[o][i] += lr * grad
    return loss


def corrupt(rng: random.Random, quad: Quad, entities: Sequence[str]) -> Quad:
    s, r, o, t = quad
    if rng.random() < 0.5:
        return rng.choice(entities), r, o, t
    return s, r, rng.choice(entities), t


def evaluate(
    grouped: Dict[str, List[Dict[str, Any]]],
    ent: Dict[str, List[float]],
    rel: Dict[str, List[float]],
    tim: Dict[str, List[float]],
) -> Dict[str, Any]:
    rows: List[Dict[str, Any]] = []
    for group_id, candidates in sorted(grouped.items()):
        gold = next((row for row in candidates if row.get("is_gold")), None)
        if gold is None:
            continue
        scored = []
        for row in candidates:
            quad = (
                str(row.get("raw_subject_id")),
                str(row.get("raw_relation_id")),
                str(row.get("raw_object_id")),
                str(row.get("raw_time_start")),
            )
            if quad[0] not in ent or quad[2] not in ent or quad[1] not in rel or quad[3] not in tim:
                score = float("inf")
            else:
                score = l2_score(ent, rel, tim, quad)
            scored.append((score, row))
        top = min(scored, key=lambda item: item[0])[1]
        rows.append(
            {
                "rank_group": group_id,
                "correct": bool(top.get("is_gold")),
                "top_label": top.get("candidate_label"),
                "gold_label": gold.get("gold_label"),
            }
        )
    accuracy = sum(1 for row in rows if row["correct"]) / len(rows) if rows else 0.0
    return {"evaluated": len(rows), "accuracy": accuracy, "details": rows[:100]}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a lightweight TTransE-style temporal KG baseline on ICEWS14 assertions.")
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--train-limit", type=int, default=50000)
    parser.add_argument("--groups-limit", type=int, default=10000)
    parser.add_argument("--dim", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--margin", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    records = load_jsonl(args.input)
    train = train_quads(records, args.train_limit)
    grouped = group_records(records, args.groups_limit)
    entities = sorted({q[0] for q in train} | {q[2] for q in train})
    relations = sorted({q[1] for q in train})
    times = sorted({q[3] for q in train})
    ent = {entity: rand_vec(rng, args.dim) for entity in entities}
    rel = {relation: rand_vec(rng, args.dim) for relation in relations}
    tim = {time: rand_vec(rng, args.dim) for time in times}

    losses: List[float] = []
    for _ in range(args.epochs):
        shuffled = list(train)
        rng.shuffle(shuffled)
        total = 0.0
        for quad in shuffled:
            total += update(ent, rel, tim, quad, corrupt(rng, quad, entities), args.lr, args.margin)
        losses.append(total / max(len(shuffled), 1))

    result = evaluate(grouped, ent, rel, tim)
    payload = {
        "baseline": "ttranse_style_temporal_embedding",
        "train_quads": len(train),
        "evaluated": result["evaluated"],
        "accuracy": result["accuracy"],
        "dim": args.dim,
        "epochs": args.epochs,
        "losses": losses,
        "details": result["details"],
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "baseline.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (args.output / "summary.md").write_text(
        "\n".join(
            [
                "# ICEWS14 TTransE-Style Baseline",
                "",
                f"- Train quads: {len(train)}",
                f"- Evaluated groups: {result['evaluated']}",
                f"- Accuracy: {result['accuracy']:.4f}",
                f"- Dim: {args.dim}",
                f"- Epochs: {args.epochs}",
                f"- Losses: {', '.join(f'{loss:.4f}' for loss in losses)}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    print((args.output / "summary.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
