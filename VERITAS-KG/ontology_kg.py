from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple


@dataclass(frozen=True)
class Context:
    """A semantic context for an assertion."""

    domain: str
    time_start: Optional[date] = None
    time_end: Optional[date] = None
    location: Optional[str] = None
    condition: Optional[str] = None

    def overlaps(self, other: "Context") -> bool:
        if self.domain != other.domain:
            return False

        if self.location and other.location and self.location != other.location:
            return False

        if self.condition and other.condition and self.condition != other.condition:
            return False

        return self._time_overlaps(other)

    def _time_overlaps(self, other: "Context") -> bool:
        if self.time_start is None or self.time_end is None:
            return True
        if other.time_start is None or other.time_end is None:
            return True
        return not (self.time_end < other.time_start or other.time_end < self.time_start)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "domain": self.domain,
            "time_start": self.time_start.isoformat() if self.time_start else None,
            "time_end": self.time_end.isoformat() if self.time_end else None,
            "location": self.location,
            "condition": self.condition,
        }


@dataclass(frozen=True)
class Evidence:
    """Source evidence supporting an assertion."""

    source_id: str
    source_type: str
    snippet: str
    weight: float = 1.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source_id": self.source_id,
            "source_type": self.source_type,
            "snippet": self.snippet,
            "weight": self.weight,
        }


@dataclass(frozen=True)
class Entity:
    """Ontology-level entity."""

    entity_id: str
    label: str
    entity_type: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.entity_id,
            "label": self.label,
            "type": self.entity_type,
        }


@dataclass(frozen=True)
class Assertion:
    """A reified knowledge statement with evidence and context."""

    subject: str
    relation: str
    obj: str
    context: Context
    evidence: Evidence
    confidence: float = 1.0
    polarity: bool = True
    metadata: Optional[Dict[str, Any]] = None

    def key(self) -> Tuple[str, str, str]:
        return self.subject, self.relation, self.obj

    def support_score(self, use_evidence_weight: bool = True) -> float:
        evidence_weight = self.evidence.weight if use_evidence_weight else 1.0
        return self.confidence * evidence_weight

    def to_dict(self, entities: Optional[Dict[str, Entity]] = None) -> Dict[str, Any]:
        subject = self.subject
        obj = self.obj
        if entities:
            subject = entities.get(self.subject, Entity(self.subject, self.subject, "Entity")).to_dict()
            obj = entities.get(self.obj, Entity(self.obj, self.obj, "Entity")).to_dict()
        return {
            "subject": subject,
            "relation": self.relation,
            "object": obj,
            "context": self.context.to_dict(),
            "evidence": self.evidence.to_dict(),
            "confidence": self.confidence,
            "polarity": self.polarity,
            "metadata": self.metadata or {},
        }


@dataclass(frozen=True)
class ImportReport:
    attempted: int
    imported: int
    rejected: int
    rejection_reasons: Dict[str, int]


class OntologyKnowledgeGraph:
    def __init__(
        self,
        enforce_ontology: bool = True,
        use_context: bool = True,
        use_evidence_weight: bool = True,
    ) -> None:
        self.entities: Dict[str, Entity] = {}
        self.assertions: List[Assertion] = []
        self.enforce_ontology = enforce_ontology
        self.use_context = use_context
        self.use_evidence_weight = use_evidence_weight
        self.allowed_relations: Dict[str, Set[Tuple[str, str]]] = {
            "works_for": {("Person", "Organization")},
            "worksAt": {("Person", "Organization")},
            "playsFor": {("Person", "Organization")},
            "wasBornIn": {("Person", "Location")},
            "diedIn": {("Person", "Location")},
            "hasWonPrize": {("Person", "Award")},
            "isMarriedTo": {("Person", "Person")},
            "owns": {("Person", "Entity"), ("Organization", "Entity")},
            "graduatedFrom": {("Person", "Organization")},
            "isAffiliatedTo": {("Person", "Organization")},
            "created": {("Person", "Artifact"), ("Organization", "Artifact")},
            "located_in": {("Organization", "Location"), ("Event", "Location")},
            "acquired": {("Organization", "Organization")},
            "part_of": {("Entity", "Entity")},
            "causes": {("Event", "Event"), ("Entity", "Event")},
            "has_verdict": {("Claim", "Verdict")},
            "isLocatedIn": {("Entity", "Location")},
            "isCitizenOf": {("Person", "Location")},
            "hasCapital": {("Country", "Location")},
            "participatedIn": {("Entity", "Event")},
            "hasOfficialLanguage": {("Country", "Language")},
            "hasGender": {("Person", "Gender")},
            "hasMusicalRole": {("Person", "Role")},
            "hasChild": {("Person", "Person")},
            "influences": {("Entity", "Entity")},
            "edited": {("Person", "CreativeWork")},
            "hasWebsite": {("Entity", "Website")},
            "livesIn": {("Person", "Location")},
            "happenedIn": {("Event", "Location")},
            "directed": {("Person", "CreativeWork")},
            "actedIn": {("Person", "CreativeWork")},
            "wroteMusicFor": {("Person", "CreativeWork")},
            "isConnectedTo": {("Location", "Location")},
            "dealsWith": {("Entity", "Entity")},
        }

    def add_entity(self, entity: Entity) -> None:
        self.entities[entity.entity_id] = entity

    def add_entities(self, entities: Iterable[Entity]) -> None:
        for entity in entities:
            self.add_entity(entity)

    def add_assertion(self, assertion: Assertion, validate: Optional[bool] = None) -> None:
        if self.enforce_ontology if validate is None else validate:
            self._validate_assertion(assertion)
        self.assertions.append(assertion)

    def import_assertions(
        self,
        assertions: Iterable[Assertion],
        strict: bool = False,
    ) -> ImportReport:
        attempted = 0
        imported = 0
        rejected = 0
        reasons: Dict[str, int] = {}
        for assertion in assertions:
            attempted += 1
            try:
                self.add_assertion(assertion)
                imported += 1
            except ValueError as exc:
                rejected += 1
                reason = str(exc)
                reasons[reason] = reasons.get(reason, 0) + 1
                if strict:
                    raise
        return ImportReport(attempted, imported, rejected, reasons)

    def _validate_assertion(self, assertion: Assertion) -> None:
        error = self.validation_error(assertion)
        if error is not None:
            raise ValueError(error)

    def validation_error(self, assertion: Assertion) -> Optional[str]:
        subj = self.entities.get(assertion.subject)
        obj = self.entities.get(assertion.obj)

        if subj is None:
            return f"Unknown subject entity: {assertion.subject}"
        if obj is None:
            return f"Unknown object entity: {assertion.obj}"

        allowed = self.allowed_relations.get(assertion.relation)
        if allowed is None:
            if assertion.relation.startswith("icews:R"):
                return None
            return None

        if not self._type_compatible(subj.entity_type, obj.entity_type, allowed):
            return (
                f"Relation {assertion.relation} not compatible with "
                f"({subj.entity_type}, {obj.entity_type})"
            )
        return None

    def is_valid_assertion(self, assertion: Assertion) -> bool:
        return self.validation_error(assertion) is None

    @staticmethod
    def _type_compatible(
        subj_type: str,
        obj_type: str,
        allowed_pairs: Set[Tuple[str, str]],
    ) -> bool:
        for left, right in allowed_pairs:
            if (left == subj_type or left == "Entity") and (
                right == obj_type or right == "Entity"
            ):
                return True
        return False

    def find_assertions(
        self,
        subject: Optional[str] = None,
        relation: Optional[str] = None,
        obj: Optional[str] = None,
        context: Optional[Context] = None,
        polarity: Optional[bool] = None,
    ) -> List[Assertion]:
        results: List[Assertion] = []
        for assertion in self.assertions:
            if subject is not None and assertion.subject != subject:
                continue
            if relation is not None and assertion.relation != relation:
                continue
            if obj is not None and assertion.obj != obj:
                continue
            if polarity is not None and assertion.polarity != polarity:
                continue
            if (
                self.use_context
                and context is not None
                and not assertion.context.overlaps(context)
            ):
                continue
            results.append(assertion)
        return results

    def detect_conflicts(self) -> List[Tuple[Assertion, Assertion]]:
        conflicts: List[Tuple[Assertion, Assertion]] = []
        grouped: Dict[Tuple[str, str, str], List[Assertion]] = {}
        for assertion in self.assertions:
            grouped.setdefault(assertion.key(), []).append(assertion)

        for assertions in grouped.values():
            positives = [assertion for assertion in assertions if assertion.polarity]
            negatives = [assertion for assertion in assertions if not assertion.polarity]
            if not positives or not negatives:
                continue
            for left in positives:
                for right in negatives:
                    if not self.use_context or left.context.overlaps(right.context):
                        conflicts.append((left, right))
        return conflicts

    def rank_assertions(self, assertions: Iterable[Assertion]) -> List[Assertion]:
        return sorted(
            assertions,
            key=lambda a: (a.support_score(self.use_evidence_weight), a.polarity),
            reverse=True,
        )

    def best_supported_assertion(
        self,
        subject: str,
        relation: str,
        obj: str,
        context: Optional[Context] = None,
    ) -> Optional[Assertion]:
        candidates = self.find_assertions(
            subject=subject,
            relation=relation,
            obj=obj,
            context=context,
        )
        ranked = self.rank_assertions(candidates)
        return ranked[0] if ranked else None

    def explain(self, assertion: Assertion) -> str:
        entity_s = self.entities[assertion.subject].label
        entity_o = self.entities[assertion.obj].label
        ctx = assertion.context
        parts = [f"{entity_s} --{assertion.relation}--> {entity_o}"]
        parts.append(f"domain={ctx.domain}")
        if ctx.time_start or ctx.time_end:
            parts.append(f"time={ctx.time_start}..{ctx.time_end}")
        if ctx.location:
            parts.append(f"location={ctx.location}")
        if ctx.condition:
            parts.append(f"condition={ctx.condition}")
        parts.append(f"evidence={assertion.evidence.source_id}")
        parts.append(f"confidence={assertion.confidence:.2f}")
        parts.append(f"support={assertion.support_score(self.use_evidence_weight):.2f}")
        return " | ".join(parts)

    def export_assertions(self) -> List[Dict[str, Any]]:
        return [assertion.to_dict(self.entities) for assertion in self.assertions]


def parse_date(value: Optional[str]) -> Optional[date]:
    if value in (None, "", "null"):
        return None
    return date.fromisoformat(str(value))


def context_from_dict(data: Dict[str, Any]) -> Context:
    return Context(
        domain=str(data.get("domain", "default")),
        time_start=parse_date(data.get("time_start")),
        time_end=parse_date(data.get("time_end")),
        location=data.get("location"),
        condition=data.get("condition"),
    )


def evidence_from_dict(data: Any) -> Evidence:
    if isinstance(data, list):
        data = max(data, key=lambda item: float(item.get("weight", 0.0))) if data else {}
    if not isinstance(data, dict):
        data = {}
    return Evidence(
        source_id=str(data.get("source_id", "")),
        source_type=str(data.get("source_type", "unknown")),
        snippet=str(data.get("snippet", "")),
        weight=float(data.get("weight", 0.0)),
    )


def entity_from_dict(data: Dict[str, Any]) -> Entity:
    return Entity(
        entity_id=str(data["id"]),
        label=str(data.get("label", data["id"])),
        entity_type=str(data.get("type", "Entity")),
    )


def assertion_from_record(record: Dict[str, Any]) -> Tuple[List[Entity], Assertion]:
    subject = entity_from_dict(record["subject"])
    obj = entity_from_dict(record["object"])
    metadata = {
        key: value
        for key, value in record.items()
        if key
        not in {
            "subject",
            "relation",
            "object",
            "context",
            "evidence",
            "confidence",
            "polarity",
        }
    }
    assertion = Assertion(
        subject=subject.entity_id,
        relation=str(record["relation"]),
        obj=obj.entity_id,
        context=context_from_dict(record.get("context", {})),
        evidence=evidence_from_dict(record.get("evidence", {})),
        confidence=float(record.get("confidence", 1.0)),
        polarity=bool(record.get("polarity", True)),
        metadata=metadata,
    )
    return [subject, obj], assertion


def build_graph_from_records(
    records: Iterable[Dict[str, Any]],
    enforce_ontology: bool = True,
    use_context: bool = True,
    use_evidence_weight: bool = True,
) -> Tuple[OntologyKnowledgeGraph, ImportReport]:
    kg = OntologyKnowledgeGraph(
        enforce_ontology=enforce_ontology,
        use_context=use_context,
        use_evidence_weight=use_evidence_weight,
    )
    assertions: List[Assertion] = []
    for record in records:
        entities, assertion = assertion_from_record(record)
        kg.add_entities(entities)
        assertions.append(assertion)
    report = kg.import_assertions(assertions)
    return kg, report


def build_demo_graph() -> OntologyKnowledgeGraph:
    kg = OntologyKnowledgeGraph()

    kg.add_entity(Entity("e1", "OpenAI", "Organization"))
    kg.add_entity(Entity("e2", "San Francisco", "Location"))
    kg.add_entity(Entity("e3", "GPT-5", "Event"))
    kg.add_entity(Entity("e4", "Alice", "Person"))
    kg.add_entity(Entity("e5", "Acme Corp", "Organization"))

    kg.add_assertion(
        Assertion(
            subject="e1",
            relation="located_in",
            obj="e2",
            context=Context(domain="company_profile"),
            evidence=Evidence("src_001", "web", "OpenAI is based in San Francisco.", 0.95),
            confidence=0.92,
        )
    )
    kg.add_assertion(
        Assertion(
            subject="e4",
            relation="works_for",
            obj="e1",
            context=Context(domain="employment", time_start=date(2024, 1, 1)),
            evidence=Evidence("src_002", "profile", "Alice works at OpenAI.", 0.9),
            confidence=0.88,
        )
    )
    kg.add_assertion(
        Assertion(
            subject="e4",
            relation="works_for",
            obj="e5",
            context=Context(domain="employment", time_start=date(2025, 1, 1)),
            evidence=Evidence("src_003", "profile", "Alice now works at Acme Corp.", 0.85),
            confidence=0.93,
        )
    )
    kg.add_assertion(
        Assertion(
            subject="e4",
            relation="works_for",
            obj="e1",
            context=Context(
                domain="employment",
                time_start=date(2024, 6, 1),
                location="San Francisco",
            ),
            evidence=Evidence(
                "src_005",
                "social_post",
                "Alice no longer works at OpenAI as of mid-2024.",
                0.45,
            ),
            confidence=0.51,
            polarity=False,
        )
    )
    kg.add_assertion(
        Assertion(
            subject="e3",
            relation="located_in",
            obj="e2",
            context=Context(domain="event_log"),
            evidence=Evidence("src_004", "log", "The event is hosted in San Francisco.", 0.7),
            confidence=0.65,
        )
    )

    return kg


def demo() -> None:
    kg = build_demo_graph()
    conflicts = kg.detect_conflicts()

    print("Assertions:")
    for assertion in kg.assertions:
        print(" -", kg.explain(assertion))

    print("\nConflicts:")
    if not conflicts:
        print(" - none")
    else:
        for left, right in conflicts:
            print(" -")
            print("   ", kg.explain(left))
            print("   ", kg.explain(right))

    print("\nQuery: works_for(Alice, ?)")
    query = kg.find_assertions(subject="e4", relation="works_for")
    for assertion in kg.rank_assertions(query):
        print(" -", kg.explain(assertion))

    print("\nBest supported assertion for Alice works_for OpenAI:")
    best = kg.best_supported_assertion("e4", "works_for", "e1")
    if best is None:
        print(" - none")
    else:
        print(" -", kg.explain(best))


if __name__ == "__main__":
    demo()
