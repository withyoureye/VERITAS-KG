from __future__ import annotations

import argparse
import json
import types
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple


PREPARED_SIGNATURES: Dict[str, Tuple[str, str]] = {
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


ONTOLOGY_DOMAIN_RANGE: Dict[str, Tuple[str, str]] = {
    **PREPARED_SIGNATURES,
    # Deliberately use a parent class here so that the closure ablation has
    # real work to do. Without subclass closure, CreativeWork objects for
    # created/directed/actedIn/wroteMusicFor are over-rejected.
    "created": ("Person", "Artifact"),
    "directed": ("Person", "Artifact"),
    "actedIn": ("Person", "Artifact"),
    "wroteMusicFor": ("Person", "Artifact"),
}


SUBCLASS_OF: Dict[str, str] = {
    "Award": "Entity",
    "CreativeWork": "Artifact",
    "Artifact": "Entity",
    "Country": "Location",
    "Event": "Entity",
    "Gender": "Entity",
    "Language": "Entity",
    "Location": "Entity",
    "Organization": "Entity",
    "Person": "Entity",
    "Role": "Entity",
    "Website": "Entity",
}

Signature = Tuple[str, str, str]


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def ancestors(entity_type: str) -> Set[str]:
    seen = {entity_type}
    current = entity_type
    while current in SUBCLASS_OF and SUBCLASS_OF[current] not in seen:
        current = SUBCLASS_OF[current]
        seen.add(current)
    return seen


def exact_type_match(actual: str, expected: str) -> bool:
    return expected == "Entity" or actual == expected


def closure_type_match(actual: str, expected: str) -> bool:
    return expected == "Entity" or expected in ancestors(actual)


def record_signature(record: Dict[str, Any]) -> Signature:
    return (
        str(record["subject"]["type"]),
        str(record["relation"]),
        str(record["object"]["type"]),
    )


def learned_train_signatures(records: Sequence[Dict[str, Any]]) -> Dict[str, Set[Tuple[str, str]]]:
    signatures: Dict[str, Set[Tuple[str, str]]] = defaultdict(set)
    for record in records:
        if record.get("split") != "train" or record.get("is_gold") is not True:
            continue
        signatures[str(record["relation"])].add(
            (str(record["subject"]["type"]), str(record["object"]["type"]))
        )
    return signatures


def accepts(
    record: Dict[str, Any],
    mode: str,
    train_signatures: Dict[str, Set[Tuple[str, str]]],
) -> bool:
    relation = str(record["relation"])
    subj_type = str(record["subject"]["type"])
    obj_type = str(record["object"]["type"])

    if mode == "relation_only":
        return relation in PREPARED_SIGNATURES

    if mode == "relation_signature_train":
        return (subj_type, obj_type) in train_signatures.get(relation, set())

    if mode == "domain_range_only":
        expected = PREPARED_SIGNATURES.get(relation)
        if expected is None:
            return True
        return exact_type_match(subj_type, expected[0]) and exact_type_match(obj_type, expected[1])

    if mode == "without_subclass_closure":
        expected = ONTOLOGY_DOMAIN_RANGE.get(relation)
        if expected is None:
            return True
        return exact_type_match(subj_type, expected[0]) and exact_type_match(obj_type, expected[1])

    if mode == "type_closure":
        expected = ONTOLOGY_DOMAIN_RANGE.get(relation)
        if expected is None:
            return True
        return closure_type_match(subj_type, expected[0]) and closure_type_match(obj_type, expected[1])

    raise ValueError(f"Unknown mode: {mode}")


def evaluate_mode(
    records: Sequence[Dict[str, Any]],
    mode: str,
    train_signatures: Dict[str, Set[Tuple[str, str]]],
) -> Dict[str, Any]:
    accepted = 0
    rejected = 0
    true_accept = 0
    true_reject = 0
    false_accept = 0
    false_reject = 0
    by_relation: Counter[str] = Counter()
    false_examples: List[Dict[str, Any]] = []

    for record in records:
        accepted_flag = accepts(record, mode, train_signatures)
        valid_gold = record.get("is_gold") is True
        accepted += int(accepted_flag)
        rejected += int(not accepted_flag)
        true_accept += int(accepted_flag and valid_gold)
        true_reject += int((not accepted_flag) and (not valid_gold))
        false_accept += int(accepted_flag and (not valid_gold))
        false_reject += int((not accepted_flag) and valid_gold)
        if accepted_flag and not valid_gold:
            by_relation[str(record["relation"])] += 1
        if len(false_examples) < 20 and ((accepted_flag and not valid_gold) or ((not accepted_flag) and valid_gold)):
            false_examples.append(
                {
                    "assertion_id": record.get("assertion_id"),
                    "relation": record.get("relation"),
                    "subject_type": record.get("subject", {}).get("type"),
                    "object_type": record.get("object", {}).get("type"),
                    "is_gold": record.get("is_gold"),
                    "accepted": accepted_flag,
                }
            )

    total = len(records)
    invalid_total = sum(1 for record in records if record.get("is_gold") is not True)
    valid_total = total - invalid_total
    validation_accuracy = (true_accept + true_reject) / total if total else 0.0
    false_accept_rate = false_accept / invalid_total if invalid_total else 0.0
    false_reject_rate = false_reject / valid_total if valid_total else 0.0
    invalid_accepted_rate = false_accept / accepted if accepted else 0.0

    return {
        "baseline": mode,
        "total_assertions": total,
        "accepted": accepted,
        "rejected": rejected,
        "valid_total": valid_total,
        "invalid_total": invalid_total,
        "true_accept": true_accept,
        "true_reject": true_reject,
        "false_accept": false_accept,
        "false_reject": false_reject,
        "validation_accuracy": validation_accuracy,
        "false_accept_rate": false_accept_rate,
        "false_reject_rate": false_reject_rate,
        "invalid_accepted_rate": invalid_accepted_rate,
        "false_accept_by_relation": dict(by_relation.most_common(20)),
        "sampled_errors": false_examples,
    }


def evaluate_cached_reasoner(
    records: Sequence[Dict[str, Any]],
    baseline: str,
    acceptance_by_signature: Dict[Signature, bool],
) -> Dict[str, Any]:
    accepted = 0
    rejected = 0
    true_accept = 0
    true_reject = 0
    false_accept = 0
    false_reject = 0
    by_relation: Counter[str] = Counter()
    false_examples: List[Dict[str, Any]] = []

    for record in records:
        signature = record_signature(record)
        accepted_flag = acceptance_by_signature.get(signature, True)
        valid_gold = record.get("is_gold") is True
        accepted += int(accepted_flag)
        rejected += int(not accepted_flag)
        true_accept += int(accepted_flag and valid_gold)
        true_reject += int((not accepted_flag) and (not valid_gold))
        false_accept += int(accepted_flag and (not valid_gold))
        false_reject += int((not accepted_flag) and valid_gold)
        if accepted_flag and not valid_gold:
            by_relation[str(record["relation"])] += 1
        if len(false_examples) < 20 and ((accepted_flag and not valid_gold) or ((not accepted_flag) and valid_gold)):
            false_examples.append(
                {
                    "assertion_id": record.get("assertion_id"),
                    "relation": record.get("relation"),
                    "subject_type": record.get("subject", {}).get("type"),
                    "object_type": record.get("object", {}).get("type"),
                    "is_gold": record.get("is_gold"),
                    "accepted": accepted_flag,
                }
            )

    total = len(records)
    invalid_total = sum(1 for record in records if record.get("is_gold") is not True)
    valid_total = total - invalid_total
    validation_accuracy = (true_accept + true_reject) / total if total else 0.0
    false_accept_rate = false_accept / invalid_total if invalid_total else 0.0
    false_reject_rate = false_reject / valid_total if valid_total else 0.0
    invalid_accepted_rate = false_accept / accepted if accepted else 0.0

    return {
        "baseline": baseline,
        "total_assertions": total,
        "accepted": accepted,
        "rejected": rejected,
        "valid_total": valid_total,
        "invalid_total": invalid_total,
        "true_accept": true_accept,
        "true_reject": true_reject,
        "false_accept": false_accept,
        "false_reject": false_reject,
        "validation_accuracy": validation_accuracy,
        "false_accept_rate": false_accept_rate,
        "false_reject_rate": false_reject_rate,
        "invalid_accepted_rate": invalid_accepted_rate,
        "false_accept_by_relation": dict(by_relation.most_common(20)),
        "sampled_errors": false_examples,
        "unique_type_relation_signatures": len(acceptance_by_signature),
    }


def audit_dataset(records: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    split_counts: Counter[str] = Counter()
    label_counts: Counter[str] = Counter()
    relation_counts: Counter[str] = Counter()
    valid_relation_counts: Counter[str] = Counter()
    invalid_relation_counts: Counter[str] = Counter()
    subject_type_counts: Counter[str] = Counter()
    object_type_counts: Counter[str] = Counter()
    corruption_counts: Counter[str] = Counter()
    signatures: Set[Signature] = set()

    for record in records:
        split = str(record.get("split", "unknown"))
        relation = str(record.get("relation", "unknown"))
        subj_type = str(record.get("subject", {}).get("type", "unknown"))
        obj_type = str(record.get("object", {}).get("type", "unknown"))
        is_valid = record.get("is_gold") is True

        split_counts[split] += 1
        label_counts["valid" if is_valid else "invalid"] += 1
        relation_counts[relation] += 1
        subject_type_counts[subj_type] += 1
        object_type_counts[obj_type] += 1
        signatures.add((subj_type, relation, obj_type))
        if is_valid:
            valid_relation_counts[relation] += 1
        else:
            invalid_relation_counts[relation] += 1
            corruption_counts[str(record.get("corruption", "unspecified"))] += 1

    return {
        "same_evaluation_set": True,
        "evaluation_protocol": (
            "All baselines consume the exact same loaded assertion records from the same JSONL file; "
            "no baseline-specific sampling or filtering is applied."
        ),
        "label_construction": {
            "valid_assertions": (
                "Valid labels are original YAGO triples whose relations are covered by the prepared "
                "domain/range signatures, plus the positive test candidates used in ranking groups."
            ),
            "invalid_assertions": (
                "Invalid labels are generated from real YAGO test triples by subject/object type "
                "corruption while keeping the relation fixed; this preserves the relation distribution "
                "inside the corrupted candidate set and avoids trivial relation-only detection."
            ),
            "corruption_strategy": (
                "Subject corruption replaces the subject with a Location-typed entity for relations "
                "expecting a Person-like subject; object corruption replaces the object with a "
                "Person-typed entity for relations expecting a non-Person object."
            ),
        },
        "total_assertions": len(records),
        "label_counts": dict(label_counts),
        "split_counts": dict(split_counts),
        "corruption_counts": dict(corruption_counts),
        "unique_relations": len(relation_counts),
        "unique_subject_types": len(subject_type_counts),
        "unique_object_types": len(object_type_counts),
        "unique_entity_types": len(set(subject_type_counts) | set(object_type_counts)),
        "unique_type_relation_signatures": len(signatures),
        "top_relations": dict(relation_counts.most_common(20)),
        "top_invalid_relations": dict(invalid_relation_counts.most_common(20)),
        "valid_relations": len(valid_relation_counts),
        "invalid_relations": len(invalid_relation_counts),
        "subject_type_counts": dict(subject_type_counts.most_common()),
        "object_type_counts": dict(object_type_counts.most_common()),
    }


def unique_signatures(records: Sequence[Dict[str, Any]]) -> List[Signature]:
    return sorted({record_signature(record) for record in records})


def shacl_domain_range_acceptance(records: Sequence[Dict[str, Any]]) -> Tuple[Optional[Dict[Signature, bool]], str]:
    try:
        from pyshacl import validate
        from rdflib import BNode, Graph, Literal, Namespace, RDF, RDFS, URIRef
        from rdflib.namespace import SH, XSD
    except Exception as exc:  # pragma: no cover - environment diagnostic
        return None, f"not_run_dependency_error: {type(exc).__name__}: {exc}"

    ex = Namespace("http://example.org/yago-import-validation/")

    def class_uri(name: str) -> URIRef:
        return URIRef(ex[f"class/{name}"])

    def property_uri(name: str) -> URIRef:
        return URIRef(ex[f"property/{name}"])

    data_graph = Graph()
    shapes_graph = Graph()
    node_to_signature: Dict[URIRef, Signature] = {}
    signatures = unique_signatures(records)

    for child, parent in SUBCLASS_OF.items():
        data_graph.add((class_uri(child), RDFS.subClassOf, class_uri(parent)))

    for index, signature in enumerate(signatures):
        subj_type, relation, obj_type = signature
        subject = URIRef(ex[f"assertion/{index}/subject"])
        obj = URIRef(ex[f"assertion/{index}/object"])
        node_to_signature[subject] = signature
        data_graph.add((subject, RDF.type, class_uri(subj_type)))
        data_graph.add((obj, RDF.type, class_uri(obj_type)))
        data_graph.add((subject, property_uri(relation), obj))

    for relation, (domain_type, range_type) in ONTOLOGY_DOMAIN_RANGE.items():
        shape = URIRef(ex[f"shape/{relation}"])
        prop_shape = BNode()
        prop = property_uri(relation)
        shapes_graph.add((shape, RDF.type, SH.NodeShape))
        shapes_graph.add((shape, SH.targetSubjectsOf, prop))
        shapes_graph.add((shape, SH["class"], class_uri(domain_type)))
        shapes_graph.add((shape, SH.property, prop_shape))
        shapes_graph.add((prop_shape, SH.path, prop))
        shapes_graph.add((prop_shape, SH.minCount, Literal(1, datatype=XSD.integer)))
        shapes_graph.add((prop_shape, SH["class"], class_uri(range_type)))

    _, results_graph, _ = validate(
        data_graph=data_graph,
        shacl_graph=shapes_graph,
        inference="rdfs",
        abort_on_first=False,
        allow_infos=True,
        allow_warnings=True,
    )

    acceptance = {signature: True for signature in signatures}
    for result in results_graph.subjects(RDF.type, SH.ValidationResult):
        focus_node = results_graph.value(result, SH.focusNode)
        if focus_node in node_to_signature:
            acceptance[node_to_signature[focus_node]] = False

    rejected = sum(1 for accepted in acceptance.values() if not accepted)
    return acceptance, f"run_pyshacl_rdfs_unique_signatures={len(signatures)} rejected_signatures={rejected}"


def owlready2_type_closure_acceptance(records: Sequence[Dict[str, Any]]) -> Tuple[Optional[Dict[Signature, bool]], str]:
    try:
        from owlready2 import Thing, get_ontology
    except Exception as exc:  # pragma: no cover - environment diagnostic
        return None, f"not_run_dependency_error: {type(exc).__name__}: {exc}"

    all_types = set(SUBCLASS_OF.keys()) | set(SUBCLASS_OF.values())
    for subj_type, _, obj_type in unique_signatures(records):
        all_types.add(subj_type)
        all_types.add(obj_type)

    ontology = get_ontology("http://example.org/yago_import_validation.owl#")
    with ontology:
        owl_classes = {name: types.new_class(name, (Thing,)) for name in sorted(all_types)}
        for child, parent in SUBCLASS_OF.items():
            if child in owl_classes and parent in owl_classes and owl_classes[parent] not in owl_classes[child].is_a:
                owl_classes[child].is_a.append(owl_classes[parent])

    def class_match(actual_type: str, expected_type: str) -> bool:
        if expected_type == "Entity":
            return True
        actual_class = owl_classes.get(actual_type)
        expected_class = owl_classes.get(expected_type)
        return bool(actual_class and expected_class and expected_class in actual_class.ancestors())

    acceptance: Dict[Signature, bool] = {}
    for signature in unique_signatures(records):
        subj_type, relation, obj_type = signature
        expected = ONTOLOGY_DOMAIN_RANGE.get(relation)
        if expected is None:
            acceptance[signature] = True
            continue
        acceptance[signature] = class_match(subj_type, expected[0]) and class_match(obj_type, expected[1])

    rejected = sum(1 for accepted in acceptance.values() if not accepted)
    return acceptance, f"run_owlready2_class_hierarchy_unique_signatures={len(acceptance)} rejected_signatures={rejected}"


def run_optional_reasoner_baselines(
    records: Sequence[Dict[str, Any]],
    skip_reasoners: bool,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    status = optional_reasoner_status()
    if skip_reasoners:
        status["shacl_baseline"] = "skipped_by_user"
        status["owl_reasoner_baseline"] = "skipped_by_user"
        return [], status

    results: List[Dict[str, Any]] = []
    if status.get("pyshacl") == "available" and status.get("rdflib") == "available":
        shacl_acceptance, shacl_status = shacl_domain_range_acceptance(records)
        status["shacl_baseline"] = shacl_status
        if shacl_acceptance is not None:
            results.append(evaluate_cached_reasoner(records, "shacl_domain_range_rdfs", shacl_acceptance))
    else:
        status["shacl_baseline"] = "not_run_pyshacl_or_rdflib_unavailable"

    if status.get("owlready2") == "available":
        owl_acceptance, owl_status = owlready2_type_closure_acceptance(records)
        status["owl_reasoner_baseline"] = owl_status
        if owl_acceptance is not None:
            results.append(evaluate_cached_reasoner(records, "owlready2_type_closure", owl_acceptance))
    else:
        status["owl_reasoner_baseline"] = "not_run_owlready2_unavailable"

    return results, status


def optional_reasoner_status() -> Dict[str, Any]:
    status: Dict[str, Any] = {}
    for package in ["pyshacl", "owlready2", "rdflib"]:
        try:
            __import__(package)
            status[package] = "available"
        except Exception as exc:  # pragma: no cover - environment diagnostic
            status[package] = f"unavailable: {type(exc).__name__}"
    if status.get("pyshacl") != "available":
        status["shacl_baseline"] = "not_run_pyshacl_unavailable"
    if status.get("owlready2") != "available":
        status["owl_reasoner_baseline"] = "not_run_owlready2_unavailable"
    return status


def write_summary(path: Path, payload: Dict[str, Any]) -> None:
    audit = payload["dataset_audit"]
    lines = [
        "# YAGO Ontology-Constrained Import Validation",
        "",
        "This is an import validation diagnostic, not a standalone strong reasoning benchmark.",
        "",
        "All YAGO baselines are evaluated on the same labeled import-validation assertion set; the same JSONL file is loaded once and no baseline-specific sampling or filtering is applied.",
        "",
        f"- Assertions: `{payload['total_assertions']}`",
        f"- Valid labels: `{payload['valid_total']}`",
        f"- Invalid labels: `{payload['invalid_total']}`",
        f"- Input: `{payload['input']}`",
        f"- Unique relations: `{audit['unique_relations']}`",
        f"- Unique entity types: `{audit['unique_entity_types']}`",
        f"- Unique subject/relation/object type signatures: `{audit['unique_type_relation_signatures']}`",
        "",
        "| baseline | accepted | rejected | validation_acc | false_accept_rate | false_reject_rate | invalid_accepted_rate |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in payload["results"]:
        lines.append(
            "| {baseline} | {accepted} | {rejected} | {validation_accuracy:.4f} | "
            "{false_accept_rate:.4f} | {false_reject_rate:.4f} | {invalid_accepted_rate:.4f} |".format(**row)
        )
    lines.extend(
        [
            "",
            "## Dataset And Label Construction Audit",
            "",
            audit["label_construction"]["valid_assertions"],
            "",
            audit["label_construction"]["invalid_assertions"],
            "",
            audit["label_construction"]["corruption_strategy"],
            "",
            "| label | count |",
            "|---|---:|",
        ]
    )
    for label, count in sorted(audit["label_counts"].items()):
        lines.append(f"| {label} | {count} |")
    lines.extend(
        [
            "",
            "| split | count |",
            "|---|---:|",
        ]
    )
    for split, count in sorted(audit["split_counts"].items()):
        lines.append(f"| {split} | {count} |")
    lines.extend(
        [
            "",
            "| invalid corruption | count |",
            "|---|---:|",
        ]
    )
    for corruption, count in sorted(audit["corruption_counts"].items()):
        lines.append(f"| {corruption} | {count} |")
    lines.extend(
        [
            "",
            "Top invalid-label relations, useful for checking that invalid examples are not concentrated in one trivial relation:",
            "",
            "| relation | invalid count |",
            "|---|---:|",
        ]
    )
    for relation, count in list(audit["top_invalid_relations"].items())[:10]:
        lines.append(f"| {relation} | {count} |")
    lines.extend(
        [
            "",
            "## Optional SHACL/OWL Status",
            "",
        ]
    )
    for key, value in payload["reasoner_status"].items():
        lines.append(f"- `{key}`: `{value}`")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run YAGO import validation baselines on the same assertion file.")
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--skip-reasoners", action="store_true", help="Skip pySHACL/owlready2 baselines even if installed.")
    args = parser.parse_args()

    records = load_jsonl(args.input)
    train_signatures = learned_train_signatures(records)
    modes = [
        "relation_only",
        "relation_signature_train",
        "domain_range_only",
        "without_subclass_closure",
        "type_closure",
    ]
    results = [evaluate_mode(records, mode, train_signatures) for mode in modes]
    reasoner_results, reasoner_status = run_optional_reasoner_baselines(records, args.skip_reasoners)
    results.extend(reasoner_results)
    valid_total = sum(1 for record in records if record.get("is_gold") is True)
    dataset_audit = audit_dataset(records)
    payload = {
        "input": str(args.input),
        "total_assertions": len(records),
        "valid_total": valid_total,
        "invalid_total": len(records) - valid_total,
        "modes": modes + [row["baseline"] for row in reasoner_results],
        "results": results,
        "reasoner_status": reasoner_status,
        "dataset_audit": dataset_audit,
    }
    args.output.mkdir(parents=True, exist_ok=True)
    write_json(args.output / "validation_baselines.json", payload)
    write_json(args.output / "dataset_audit.json", dataset_audit)
    write_summary(args.output / "summary.md", payload)
    print((args.output / "summary.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
