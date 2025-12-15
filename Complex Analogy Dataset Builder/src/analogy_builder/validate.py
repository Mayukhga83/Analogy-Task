from __future__ import annotations

import json
from collections import Counter

import numpy as np

from .config import AppConfig
from .io_utils import load_concept_pairs, read_jsonl
from .models import QuestionRecord, RelationRankingRecord, WikiEvidenceRecord


def validate_input(config: AppConfig) -> dict:
    pairs = load_concept_pairs(config.paths.input_csv)
    counts = Counter(pair.base_relation for pair in pairs)
    undersized = {
        relation: count
        for relation, count in counts.items()
        if count < config.questions.target_count + 1
    }
    return {
        "pair_count": len(pairs),
        "base_relation_count": len(counts),
        "pairs_per_base_relation": dict(sorted(counts.items())),
        "groups_too_small_for_four_targets": undersized,
    }


def validate_outputs(config: AppConfig) -> dict:
    pairs = load_concept_pairs(config.paths.input_csv)
    pair_map = {pair.pair_id: pair for pair in pairs}
    evidence = list(
        read_jsonl(
            config.paths.work_dir / "wikipedia_evidence.jsonl",
            WikiEvidenceRecord,
        )
    )
    rankings = list(
        read_jsonl(
            config.paths.work_dir / "relation_rankings.jsonl",
            RelationRankingRecord,
        )
    )
    questions = list(
        read_jsonl(
            config.paths.output_dir / "analogy_questions.jsonl",
            QuestionRecord,
        )
    )

    errors: list[str] = []
    warnings: list[str] = []
    seen_questions: set[str] = set()

    for question in questions:
        if question.question_id in seen_questions:
            errors.append(f"Duplicate question_id: {question.question_id}")
        seen_questions.add(question.question_id)
        if len(question.targets) != config.questions.target_count:
            errors.append(
                f"{question.question_id}: expected {config.questions.target_count} targets"
            )
        target_ids = [target.pair_id for target in question.targets]
        if len(target_ids) != len(set(target_ids)):
            errors.append(f"{question.question_id}: duplicate target pair")
        if question.stem.pair_id in target_ids:
            errors.append(f"{question.question_id}: stem appears among targets")
        for target in question.targets:
            source = pair_map.get(target.pair_id)
            if source is None:
                errors.append(f"{question.question_id}: unknown target {target.pair_id}")
            elif source.base_relation != question.base_relation:
                errors.append(
                    f"{question.question_id}: target {target.pair_id} has different base relation"
                )
        for measure, scores in question.scores.items():
            if len(scores) != len(question.targets):
                errors.append(
                    f"{question.question_id}: {measure} score length mismatch"
                )
            answer = question.answers.get(measure)
            if answer is None or not (0 <= answer < len(question.targets)):
                errors.append(f"{question.question_id}: invalid answer for {measure}")

    relation_counts = [
        len(record.relations)
        for record in rankings
        if record.relations and not record.error
    ]
    context_counts = [len(record.evidence_sentences) for record in evidence]
    measure_agreement = Counter()
    for question in questions:
        values = list(question.answers.values())
        if len(set(values)) == 1:
            measure_agreement["all_three_agree"] += 1
        elif len(set(values)) == 2:
            measure_agreement["two_agree"] += 1
        else:
            measure_agreement["all_differ"] += 1

    failed_evidence = [record.pair_id for record in evidence if not record.passes_minimum]
    failed_rankings = [record.pair_id for record in rankings if record.error or not record.relations]
    if failed_evidence:
        warnings.append(f"{len(failed_evidence)} pairs failed the Wikipedia minimum")
    if failed_rankings:
        warnings.append(f"{len(failed_rankings)} pairs have no usable relation ranking")

    report = {
        "status": "ok" if not errors else "failed",
        "errors": errors,
        "warnings": warnings,
        "pair_count": len(pairs),
        "evidence_record_count": len(evidence),
        "ranking_record_count": len(rankings),
        "question_count": len(questions),
        "distinct_question_ids": len(seen_questions),
        "relation_statistics": {
            "mean": float(np.mean(relation_counts)) if relation_counts else 0.0,
            "median": float(np.median(relation_counts)) if relation_counts else 0.0,
            "minimum": min(relation_counts) if relation_counts else 0,
            "maximum": max(relation_counts) if relation_counts else 0,
        },
        "context_sentence_statistics": {
            "mean": float(np.mean(context_counts)) if context_counts else 0.0,
            "median": float(np.median(context_counts)) if context_counts else 0.0,
            "minimum": min(context_counts) if context_counts else 0,
            "maximum": max(context_counts) if context_counts else 0,
        },
        "semantic_measure_agreement": dict(measure_agreement),
    }
    output = config.paths.output_dir / "validation_report.json"
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return report
