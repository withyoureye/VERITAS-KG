from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Set, Tuple

from run_icews_ttranse_baseline import corrupt, l2_score, rand_vec, update


Quad = Tuple[str, str, str, str]


def read_quads(path: Path, limit: int) -> List[Quad]:
    quads: List[Quad] = []
    if not path.exists():
        return quads
    with path.open("r", encoding="utf-8") as handle:
        for idx, line in enumerate(handle):
            if limit > 0 and idx >= limit:
                break
            parts = line.strip().split()
            if len(parts) < 4:
                continue
            subj, rel, obj, time = parts[:4]
            quads.append((subj, rel, obj, time))
    return quads


def train_model(
    train: Sequence[Quad],
    entities: Sequence[str],
    relations: Sequence[str],
    times: Sequence[str],
    dim: int,
    epochs: int,
    lr: float,
    margin: float,
    seed: int,
) -> Tuple[Dict[str, List[float]], Dict[str, List[float]], Dict[str, List[float]], List[float]]:
    rng = random.Random(seed)
    ent = {entity: rand_vec(rng, dim) for entity in entities}
    rel = {relation: rand_vec(rng, dim) for relation in relations}
    tim = {time: rand_vec(rng, dim) for time in times}
    losses: List[float] = []
    for _ in range(epochs):
        shuffled = list(train)
        rng.shuffle(shuffled)
        total = 0.0
        for quad in shuffled:
            total += update(ent, rel, tim, quad, corrupt(rng, quad, entities), lr, margin)
        losses.append(total / max(len(shuffled), 1))
    return ent, rel, tim, losses


def rank_tail(
    query: Quad,
    candidates: Sequence[str],
    known: Set[Quad],
    ent: Dict[str, List[float]],
    rel: Dict[str, List[float]],
    tim: Dict[str, List[float]],
) -> int:
    subj, relation, obj, time = query
    scores = []
    for candidate in candidates:
        quad = (subj, relation, candidate, time)
        if candidate != obj and quad in known:
            continue
        scores.append((l2_score(ent, rel, tim, quad), candidate))
    scores.sort(key=lambda item: item[0])
    for idx, (_, candidate) in enumerate(scores, start=1):
        if candidate == obj:
            return idx
    return len(scores) + 1


def rank_head(
    query: Quad,
    candidates: Sequence[str],
    known: Set[Quad],
    ent: Dict[str, List[float]],
    rel: Dict[str, List[float]],
    tim: Dict[str, List[float]],
) -> int:
    subj, relation, obj, time = query
    scores = []
    for candidate in candidates:
        quad = (candidate, relation, obj, time)
        if candidate != subj and quad in known:
            continue
        scores.append((l2_score(ent, rel, tim, quad), candidate))
    scores.sort(key=lambda item: item[0])
    for idx, (_, candidate) in enumerate(scores, start=1):
        if candidate == subj:
            return idx
    return len(scores) + 1


def sampled_candidates(rng: random.Random, entities: Sequence[str], gold: str, size: int) -> List[str]:
    if size <= 0 or size >= len(entities):
        pool = list(entities)
    else:
        sample_size = min(size - 1, len(entities) - 1)
        pool = rng.sample([entity for entity in entities if entity != gold], sample_size)
        pool.append(gold)
    return pool


def metrics(ranks: Sequence[int]) -> Dict[str, float]:
    if not ranks:
        return {"mrr": 0.0, "hits1": 0.0, "hits3": 0.0, "hits10": 0.0}
    return {
        "mrr": sum(1.0 / rank for rank in ranks) / len(ranks),
        "hits1": sum(1 for rank in ranks if rank <= 1) / len(ranks),
        "hits3": sum(1 for rank in ranks if rank <= 3) / len(ranks),
        "hits10": sum(1 for rank in ranks if rank <= 10) / len(ranks),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run filtered ICEWS14 temporal link prediction with a TTransE-style model.")
    parser.add_argument("--data-root", default=Path("../data"), type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--train-limit", type=int, default=50000)
    parser.add_argument("--test-limit", type=int, default=2000)
    parser.add_argument("--candidate-size", type=int, default=1000)
    parser.add_argument("--dim", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--margin", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    root = args.data_root / "ICEWS14"
    train = read_quads(root / "train.txt", args.train_limit)
    test = read_quads(root / "test.txt", args.test_limit)
    all_known = set(read_quads(root / "train.txt", -1)) | set(read_quads(root / "test.txt", -1))
    entities = sorted({quad[0] for quad in train + test} | {quad[2] for quad in train + test}, key=lambda x: int(x))
    relations = sorted({quad[1] for quad in train + test}, key=lambda x: int(x))
    times = sorted({quad[3] for quad in train + test}, key=lambda x: int(float(x)))
    ent, rel, tim, losses = train_model(
        train,
        entities,
        relations,
        times,
        args.dim,
        args.epochs,
        args.lr,
        args.margin,
        args.seed,
    )

    rng = random.Random(args.seed)
    ranks: List[int] = []
    examples = []
    for quad in test:
        head_candidates = sampled_candidates(rng, entities, quad[0], args.candidate_size)
        tail_candidates = sampled_candidates(rng, entities, quad[2], args.candidate_size)
        head_rank = rank_head(quad, head_candidates, all_known, ent, rel, tim)
        tail_rank = rank_tail(quad, tail_candidates, all_known, ent, rel, tim)
        ranks.extend([head_rank, tail_rank])
        if len(examples) < 50:
            examples.append({"quad": quad, "head_rank": head_rank, "tail_rank": tail_rank})

    result = {
        "baseline": "filtered_ttranse_style_temporal_link_prediction",
        "train_quads": len(train),
        "test_quads": len(test),
        "candidate_size": args.candidate_size,
        "dim": args.dim,
        "epochs": args.epochs,
        "losses": losses,
        **metrics(ranks),
        "examples": examples,
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "baseline.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (args.output / "summary.md").write_text(
        "\n".join(
            [
                "# ICEWS14 Filtered TTransE-Style Link Prediction",
                "",
                f"- Train quads: {len(train)}",
                f"- Test quads: {len(test)}",
                f"- Candidate size: {args.candidate_size}",
                f"- MRR: {result['mrr']:.4f}",
                f"- Hits@1: {result['hits1']:.4f}",
                f"- Hits@3: {result['hits3']:.4f}",
                f"- Hits@10: {result['hits10']:.4f}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    print((args.output / "summary.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
