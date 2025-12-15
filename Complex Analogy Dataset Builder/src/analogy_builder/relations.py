from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path
from typing import Any

import yaml
from tqdm import tqdm

from .config import AppConfig
from .io_utils import (
    append_jsonl,
    load_concept_pairs,
    load_existing_ids,
    read_jsonl,
    relation_slug,
    write_jsonl,
)
from .models import (
    CanonicalRelation,
    CanonicalRelationRecord,
    MinedRelation,
    MinedRelationRecord,
    RelationExtraction,
    WikiEvidenceRecord,
)
from .openai_service import OpenAIService


OPINION_TERMS = {
    "best",
    "better",
    "beautiful",
    "important",
    "greatest",
    "favourite",
    "favorite",
    "prestigious",
    "superior",
    "inferior",
}
TRANSIENT_TERMS = {
    "current",
    "currently",
    "recent",
    "recently",
    "now",
    "incumbent",
    "ongoing",
    "temporary",
    "latest",
}
AMBIGUOUS_RELATIONS = {
    "related_to",
    "associated_with",
    "connected_to",
    "linked_to",
    "represents",
    "influences",
    "leads",
    "has_connection_with",
}


def load_relation_aliases(path: Path | None) -> dict[str, str]:
    if path is None or not path.exists():
        return {}
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    aliases: dict[str, str] = {}
    for canonical, variants in raw.items():
        canonical_slug = relation_slug(str(canonical))
        aliases[canonical_slug] = canonical_slug
        if isinstance(variants, list):
            for variant in variants:
                aliases[relation_slug(str(variant))] = canonical_slug
        else:
            aliases[relation_slug(str(variants))] = canonical_slug
    return aliases


def normalize_relation(value: str, aliases: dict[str, str]) -> str:
    slug = relation_slug(value)
    return aliases.get(slug, slug)


def _build_relation_prompt(pair: Any, evidence: WikiEvidenceRecord, char_limit: int) -> tuple[str, str]:
    context_lines: list[str] = []
    total = 0
    for sentence in evidence.evidence_sentences:
        line = f"[{sentence.sentence_id}] {sentence.text}"
        if total + len(line) > char_limit:
            break
        context_lines.append(line)
        total += len(line)
    context = "\n".join(context_lines)
    system = (
        "You extract stable, factual relationships between two concepts from supplied "
        "Wikipedia evidence. Return only relationships supported by the evidence. "
        "Use concise snake_case predicates, keep the direction concept_1 -> concept_2, "
        "avoid opinions, vague associations, duplicates, and time-sensitive facts."
    )
    user = f"""
Concept 1: {pair.concept_1}
Concept 2: {pair.concept_2}
Known base relation: {pair.base_relation}

Extract every distinct relationship between Concept 1 and Concept 2 that is supported by the context.
For each relation, include the evidence sentence IDs that support it and a short evidence quote.
The predicate should form an RDF-like triple:
({pair.concept_1}, predicate, {pair.concept_2})

Wikipedia context:
{context}
""".strip()
    return system, user


def mine_all_relations(config: AppConfig, resume: bool = True) -> Path:
    pairs = load_concept_pairs(config.paths.input_csv)
    evidence_path = config.paths.work_dir / "wikipedia_evidence.jsonl"
    evidence_map = {
        record.pair_id: record
        for record in read_jsonl(evidence_path, WikiEvidenceRecord)
    }
    output = config.paths.work_dir / "raw_relations.jsonl"
    if not resume and output.exists():
        output.unlink()
    completed = load_existing_ids(output) if resume else set()
    aliases = load_relation_aliases(config.paths.relation_aliases_file)
    service = OpenAIService(config)

    for pair in tqdm(pairs, desc="Mining relations"):
        if pair.pair_id in completed:
            continue
        evidence = evidence_map.get(pair.pair_id)
        if evidence is None:
            append_jsonl(
                output,
                MinedRelationRecord(
                    **pair.model_dump(),
                    error="No Wikipedia evidence record found",
                ),
            )
            continue
        if (
            not evidence.passes_minimum
            and not config.wikipedia.allow_pairs_below_minimum
        ):
            append_jsonl(
                output,
                MinedRelationRecord(
                    **pair.model_dump(),
                    error=(
                        f"Only {len(evidence.evidence_sentences)} co-occurrence sentences; "
                        f"minimum is {config.wikipedia.min_sentences}"
                    ),
                ),
            )
            continue
        try:
            system, user = _build_relation_prompt(
                pair,
                evidence,
                config.openai.relation_context_char_limit,
            )
            parsed = service.parse(
                config.openai.relation_model,
                system,
                user,
                RelationExtraction,
            )
            valid_ids = {item.sentence_id for item in evidence.evidence_sentences}
            mined: list[MinedRelation] = []
            for item in parsed.relations:
                normalized = normalize_relation(item.predicate, aliases)
                if not normalized:
                    continue
                evidence_ids = sorted(
                    set(identifier for identifier in item.evidence_sentence_ids if identifier in valid_ids)
                )
                mined.append(
                    MinedRelation(
                        raw_predicate=item.predicate,
                        normalized_predicate=normalized,
                        evidence_sentence_ids=evidence_ids,
                        evidence_quote=item.evidence_quote,
                        explanation=item.explanation,
                    )
                )

            base_normalized = normalize_relation(pair.base_relation, aliases)
            if base_normalized not in {item.normalized_predicate for item in mined}:
                mined.append(
                    MinedRelation(
                        raw_predicate=pair.base_relation,
                        normalized_predicate=base_normalized,
                        source="input_base_relation",
                        explanation="Injected from the supplied base-relation label.",
                    )
                )
            append_jsonl(
                output,
                MinedRelationRecord(**pair.model_dump(), relations=mined),
            )
        except Exception as exc:  # noqa: BLE001
            append_jsonl(
                output,
                MinedRelationRecord(
                    **pair.model_dump(),
                    error=f"{type(exc).__name__}: {exc}",
                ),
            )
    return output


def suggest_filter_decision(
    relation: MinedRelation,
    seen_normalized: set[str],
    config: AppConfig,
) -> tuple[str, str, str]:
    predicate = relation.normalized_predicate
    tokens = set(predicate.split("_"))
    if relation.source == "input_base_relation" and config.relation_filter.auto_accept_base_relation:
        return "accept", "base_relation", "Supplied base relation"
    if not predicate or len(predicate) < 2:
        return "reject", "malformed", "Empty or malformed predicate"
    if predicate in seen_normalized:
        return "reject", "duplicate", "Duplicate canonical predicate within this pair"
    if predicate in AMBIGUOUS_RELATIONS:
        return "review", "ambiguous", "Vague relation label"
    if tokens & OPINION_TERMS:
        return "reject", "opinion_based", "Contains subjective language"
    if tokens & TRANSIENT_TERMS:
        return "reject", "transient", "Contains time-sensitive language"
    if not relation.evidence_sentence_ids:
        if config.relation_filter.auto_reject_unsupported:
            return "reject", "unsupported", "No valid evidence sentence ID"
        return "review", "unsupported", "No valid evidence sentence ID"
    return "accept", "valid", "Passes automatic checks"


def prepare_relation_review(config: AppConfig) -> Path:
    raw_path = config.paths.work_dir / "raw_relations.jsonl"
    evidence_path = config.paths.work_dir / "wikipedia_evidence.jsonl"
    evidence_map = {
        record.pair_id: record
        for record in read_jsonl(evidence_path, WikiEvidenceRecord)
    }
    output = config.paths.review_dir / "relation_review.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "pair_id",
        "base_relation",
        "concept_1",
        "concept_2",
        "raw_predicate",
        "normalized_predicate",
        "evidence_sentence_ids",
        "evidence_text",
        "source",
        "suggested_decision",
        "suggested_category",
        "suggested_reason",
        "final_decision",
        "final_relation",
        "reviewer_notes",
    ]
    with output.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in read_jsonl(raw_path, MinedRelationRecord):
            evidence = evidence_map.get(record.pair_id)
            by_id = (
                {item.sentence_id: item.text for item in evidence.evidence_sentences}
                if evidence
                else {}
            )
            seen: set[str] = set()
            for relation in record.relations:
                decision, category, reason = suggest_filter_decision(relation, seen, config)
                if decision != "reject" or category != "duplicate":
                    seen.add(relation.normalized_predicate)
                evidence_text = " || ".join(
                    by_id.get(identifier, f"[missing sentence {identifier}]")
                    for identifier in relation.evidence_sentence_ids
                )
                writer.writerow(
                    {
                        "pair_id": record.pair_id,
                        "base_relation": record.base_relation,
                        "concept_1": record.concept_1,
                        "concept_2": record.concept_2,
                        "raw_predicate": relation.raw_predicate,
                        "normalized_predicate": relation.normalized_predicate,
                        "evidence_sentence_ids": ",".join(
                            str(value) for value in relation.evidence_sentence_ids
                        ),
                        "evidence_text": evidence_text,
                        "source": relation.source,
                        "suggested_decision": decision,
                        "suggested_category": category,
                        "suggested_reason": reason,
                        "final_decision": "",
                        "final_relation": relation.normalized_predicate,
                        "reviewer_notes": "",
                    }
                )
    return output


def finalize_relations(config: AppConfig, use_suggestions: bool = False) -> Path:
    review_path = config.paths.review_dir / "relation_review.csv"
    if not review_path.exists():
        raise FileNotFoundError(
            f"Review file not found: {review_path}. Run prepare-review first."
        )
    grouped: dict[str, dict[str, Any]] = defaultdict(
        lambda: {"relations": [], "metadata": None}
    )
    unresolved = 0
    with review_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            decision = (row.get("final_decision") or "").strip().casefold()
            if not decision and use_suggestions:
                decision = (row.get("suggested_decision") or "").strip().casefold()
            if decision not in {"accept", "reject"}:
                unresolved += 1
                continue
            pair_id = row["pair_id"]
            grouped[pair_id]["metadata"] = {
                "pair_id": pair_id,
                "base_relation": row["base_relation"],
                "concept_1": row["concept_1"],
                "concept_2": row["concept_2"],
            }
            if decision == "reject":
                continue
            relation = relation_slug(row.get("final_relation") or row["normalized_predicate"])
            ids = [
                int(value)
                for value in (row.get("evidence_sentence_ids") or "").split(",")
                if value.strip().isdigit()
            ]
            grouped[pair_id]["relations"].append(
                CanonicalRelation(
                    relation=relation,
                    evidence_sentence_ids=ids,
                    source=row.get("source") or "reviewed",
                    notes=row.get("reviewer_notes") or None,
                )
            )
    if unresolved and not use_suggestions:
        raise ValueError(
            f"{unresolved} review rows do not have final_decision=accept/reject. "
            "Complete the CSV or rerun with --use-suggestions."
        )

    records: list[CanonicalRelationRecord] = []
    for pair_id, data in grouped.items():
        metadata = data["metadata"]
        if metadata is None:
            continue
        unique: dict[str, CanonicalRelation] = {}
        for relation in data["relations"]:
            unique.setdefault(relation.relation, relation)
        if unique:
            records.append(
                CanonicalRelationRecord(
                    **metadata,
                    relations=list(unique.values()),
                )
            )
    output = config.paths.work_dir / "canonical_relations.jsonl"
    write_jsonl(output, records)
    return output
