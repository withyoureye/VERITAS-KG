from __future__ import annotations

import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")

LABELS = {"SUPPORTS", "REFUTES", "NOT ENOUGH INFO"}
FEVER_NEGATION_TERMS = {"not", "no", "never", "false", "disassociated", "refused", "without"}

YAGO_SIGNATURES = {
    "wasBornIn": ("Person", "Location"),
    "diedIn": ("Person", "Location"),
    "worksAt": ("Person", "Organization"),
    "playsFor": ("Person", "Organization"),
    "hasWonPrize": ("Person", "Award"),
    "isMarriedTo": ("Person", "Person"),
    "owns": ("Person", "Entity"),
    "graduatedFrom": ("Person", "Organization"),
    "isAffiliatedTo": ("Person", "Organization"),
    "created": ("Person", "CreativeWork"),
    "isLocatedIn": ("Entity", "Location"),
    "isCitizenOf": ("Person", "Location"),
    "hasCapital": ("Country", "Location"),
    "participatedIn": ("Entity", "Event"),
    "hasOfficialLanguage": ("Country", "Language"),
    "directed": ("Person", "CreativeWork"),
    "actedIn": ("Person", "CreativeWork"),
    "wroteMusicFor": ("Person", "CreativeWork"),
    "hasGender": ("Person", "Gender"),
    "hasMusicalRole": ("Person", "Role"),
    "hasChild": ("Person", "Person"),
    "livesIn": ("Person", "Location"),
    "happenedIn": ("Event", "Location"),
    "isConnectedTo": ("Location", "Location"),
}


def tokenize(text: str) -> List[str]:
    return [token.lower() for token in TOKEN_RE.findall(text)]


def normalize_score(value: float) -> float:
    return max(0.0, min(1.0, value))


def lexical_overlap(a: str, b: str) -> float:
    a_tokens = set(tokenize(a))
    b_tokens = set(tokenize(b))
    if not a_tokens:
        return 0.0
    return len(a_tokens & b_tokens) / len(a_tokens)


def page_title_overlap(claim: str, evidence_lines: Sequence[str]) -> float:
    page_tokens = set()
    for line in evidence_lines:
        page = str(line).split("#", 1)[0]
        page_tokens.update(tokenize(page.replace("_", " ")))
    claim_tokens = set(tokenize(claim))
    if not claim_tokens:
        return 0.0
    return len(claim_tokens & page_tokens) / len(claim_tokens)


@dataclass
class IcewsStats:
    triple: Counter
    subj_rel: Counter
    rel_obj: Counter

    @classmethod
    def from_records(cls, records: Sequence[Dict[str, Any]]) -> "IcewsStats":
        triple: Counter = Counter()
        subj_rel: Counter = Counter()
        rel_obj: Counter = Counter()
        for record in records:
            if record.get("split") != "train":
                continue
            subj = str(record.get("raw_subject_id", record.get("subject", {}).get("id", "")))
            rel = str(record.get("raw_relation_id", record.get("relation", "")))
            obj = str(record.get("raw_object_id", record.get("object", {}).get("id", "")))
            triple[(subj, rel, obj)] += 1
            subj_rel[(subj, rel)] += 1
            rel_obj[(rel, obj)] += 1
        return cls(triple=triple, subj_rel=subj_rel, rel_obj=rel_obj)

    def max_triple(self) -> int:
        return max(self.triple.values()) if self.triple else 1

    def max_subj_rel(self) -> int:
        return max(self.subj_rel.values()) if self.subj_rel else 1

    def max_rel_obj(self) -> int:
        return max(self.rel_obj.values()) if self.rel_obj else 1


@dataclass
class YagoStats:
    relation_counts: Counter
    signature_counts: Counter

    @classmethod
    def from_records(cls, records: Sequence[Dict[str, Any]]) -> "YagoStats":
        relation_counts: Counter = Counter()
        signature_counts: Counter = Counter()
        for record in records:
            if record.get("split") != "train":
                continue
            relation = str(record.get("relation", ""))
            subj_type = str(record.get("subject", {}).get("type", "Entity"))
            obj_type = str(record.get("object", {}).get("type", "Entity"))
            relation_counts[relation] += 1
            signature_counts[(relation, subj_type, obj_type)] += 1
        return cls(relation_counts=relation_counts, signature_counts=signature_counts)

    def max_relation(self) -> int:
        return max(self.relation_counts.values()) if self.relation_counts else 1

    def max_signature(self) -> int:
        return max(self.signature_counts.values()) if self.signature_counts else 1


def score_fever(record: Dict[str, Any]) -> float:
    claim = str(record.get("claim_text", record.get("subject", {}).get("label", "")))
    snippet = str(record.get("evidence", {}).get("snippet", ""))
    label = str(record.get("candidate_label", "")).upper()
    source_type = str(record.get("evidence", {}).get("source_type", ""))
    evidence_lines = record.get("evidence_lines") or []
    if not isinstance(evidence_lines, list):
        evidence_lines = []
    overlap = max(lexical_overlap(claim, snippet), page_title_overlap(claim, evidence_lines))
    evidence_present = source_type != "claim" and "No evidence available" not in snippet and bool(evidence_lines)
    has_negation = any(token in tokenize(claim) for token in FEVER_NEGATION_TERMS)
    if label == "NOT ENOUGH INFO":
        base = 0.76 if not evidence_present else 0.24 + 0.18 * (1.0 - overlap)
    else:
        base = 0.30 + 0.36 * overlap
        if evidence_present:
            base += 0.12
        if label == "REFUTES" and has_negation:
            base += 0.14
        if label == "SUPPORTS" and has_negation:
            base -= 0.06
    return normalize_score(base)


def score_icews(record: Dict[str, Any], stats: IcewsStats) -> float:
    subj = str(record.get("raw_subject_id", record.get("subject", {}).get("id", "")))
    rel = str(record.get("raw_relation_id", record.get("relation", "")))
    obj = str(record.get("raw_object_id", record.get("object", {}).get("id", "")))
    triple_score = stats.triple[(subj, rel, obj)] / stats.max_triple()
    subj_rel_score = stats.subj_rel[(subj, rel)] / stats.max_subj_rel()
    rel_obj_score = stats.rel_obj[(rel, obj)] / stats.max_rel_obj()
    evidence = str(record.get("evidence", {}).get("snippet", ""))
    time_str = f"{record.get('raw_time_start', '')}..{record.get('raw_time_end', '')}"
    lexical = lexical_overlap(f"{subj} {rel} {obj} {time_str}", evidence)
    base = 0.12 + 0.52 * triple_score + 0.2 * subj_rel_score + 0.12 * rel_obj_score + 0.04 * lexical
    return normalize_score(base)


def score_yago(record: Dict[str, Any], stats: YagoStats) -> float:
    relation = str(record.get("relation", ""))
    subj_type = str(record.get("subject", {}).get("type", "Entity"))
    obj_type = str(record.get("object", {}).get("type", "Entity"))
    expected = YAGO_SIGNATURES.get(relation)
    compatible = 1.0
    if expected is not None:
        left, right = expected
        compatible = float((left == "Entity" or subj_type == left) and (right == "Entity" or obj_type == right))
    relation_score = stats.relation_counts[relation] / stats.max_relation()
    signature_score = stats.signature_counts[(relation, subj_type, obj_type)] / stats.max_signature()
    evidence = str(record.get("evidence", {}).get("snippet", ""))
    lexical = lexical_overlap(f"{relation} {subj_type} {obj_type}", evidence)
    base = 0.08 + 0.52 * compatible + 0.2 * signature_score + 0.16 * relation_score + 0.04 * lexical
    return normalize_score(base)


def apply_auto_evidence_scores(records: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    dataset = str(next((record.get("dataset") for record in records if record.get("dataset")), "")).upper()
    icews_stats = IcewsStats.from_records(records)
    yago_stats = YagoStats.from_records(records)
    scored: List[Dict[str, Any]] = []
    for record in records:
        copied = dict(record)
        evidence = dict(copied.get("evidence", {}))
        if dataset == "FEVER":
            evidence["weight"] = round(score_fever(copied), 6)
        elif dataset == "ICEWS14":
            evidence["weight"] = round(score_icews(copied, icews_stats), 6)
        elif dataset == "YAGO":
            evidence["weight"] = round(score_yago(copied, yago_stats), 6)
        copied["evidence"] = evidence
        copied["evidence_scorer"] = "auto"
        scored.append(copied)
    return scored
