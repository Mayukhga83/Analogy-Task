from __future__ import annotations

import random
from collections import defaultdict
from pathlib import Path

from tqdm import tqdm

from .config import AppConfig
from .embeddings import EmbeddingCache, cache_path
from .io_utils import (
    load_concept_pairs,
    read_jsonl,
    stable_question_id,
    write_jsonl,
)
from .measures import (
    argmax_with_ties,
    context_embedding_similarity,
    prototypical_similarity,
    ranked_relational_overlap,
)
from .models import (
    ConceptPair,
    QuestionRecord,
    RelationRankingRecord,
    TargetPair,
    WikiEvidenceRecord,
)
from .openai_service import OpenAIService


def _pair_text(pair: ConceptPair) -> str:
    return f"{pair.concept_1} : {pair.concept_2}"


def _stem_relation_text(pair: ConceptPair) -> str:
    return f"{pair.concept_1} : {pair.concept_2} | base relation: {pair.base_relation}"


def _ranking_scores(record: RelationRankingRecord) -> dict[str, float]:
    return {item.relation: item.score for item in record.relations}


def _build_stem_schedule(
    eligible_stems: list[ConceptPair],
    questions_per_stem: int,
    total_questions: int | None,
    rng: random.Random,
) -> list[ConceptPair]:
    base = [pair for pair in eligible_stems for _ in range(questions_per_stem)]
    if total_questions is None:
        rng.shuffle(base)
        return base
    if total_questions <= len(base):
        rng.shuffle(base)
        return base[:total_questions]
    schedule = base[:]
    while len(schedule) < total_questions:
        schedule.append(rng.choice(eligible_stems))
    rng.shuffle(schedule)
    return schedule


def generate_questions(config: AppConfig) -> Path:
    pairs = load_concept_pairs(config.paths.input_csv)
    evidence_map = {
        record.pair_id: record
        for record in read_jsonl(
            config.paths.work_dir / "wikipedia_evidence.jsonl",
            WikiEvidenceRecord,
        )
    }
    ranking_map = {
        record.pair_id: record
        for record in read_jsonl(
            config.paths.work_dir / "relation_rankings.jsonl",
            RelationRankingRecord,
        )
        if record.relations and not record.error
    }

    grouped: dict[str, list[ConceptPair]] = defaultdict(list)
    for pair in pairs:
        evidence = evidence_map.get(pair.pair_id)
        has_context = bool(
            evidence
            and (
                evidence.passes_minimum
                or config.wikipedia.allow_pairs_below_minimum
            )
        )
        if pair.pair_id not in ranking_map:
            continue
        if config.questions.require_context_pass and not has_context:
            continue
        grouped[pair.base_relation].append(pair)

    eligible_stems = [
        pair
        for group in grouped.values()
        if len(group) >= config.questions.target_count + 1
        for pair in group
    ]
    if not eligible_stems:
        raise ValueError(
            "No eligible stems. Each base-relation group needs at least "
            f"{config.questions.target_count + 1} pairs with rankings and context."
        )

    rng = random.Random(config.questions.seed)
    schedule = _build_stem_schedule(
        eligible_stems,
        config.questions.questions_per_stem,
        config.questions.total_questions,
        rng,
    )

    # Warm all pair and stem+relation representations before scoring questions.
    service = OpenAIService(config)
    cache = EmbeddingCache(cache_path(config))
    all_pair_texts = [_pair_text(pair) for pair in eligible_stems]
    all_stem_texts = [_stem_relation_text(pair) for pair in eligible_stems]
    all_context_texts = [
        sentence.text
        for pair in eligible_stems
        for sentence in evidence_map[pair.pair_id].evidence_sentences
    ]
    cache.ensure(
        all_pair_texts + all_stem_texts + all_context_texts,
        config.openai.embedding_model,
        service,
        config.embedding.batch_size,
    )

    used_target_sets: dict[str, set[tuple[str, ...]]] = defaultdict(set)
    occurrence_counter: dict[str, int] = defaultdict(int)
    questions: list[QuestionRecord] = []

    try:
        for stem in tqdm(schedule, desc="Generating analogy questions"):
            stem_relations = set(_ranking_scores(ranking_map[stem.pair_id]))
            pool = [
                pair
                for pair in grouped[stem.base_relation]
                if pair.pair_id != stem.pair_id
                and stem_relations.intersection(_ranking_scores(ranking_map[pair.pair_id]))
            ]
            if len(pool) < config.questions.target_count:
                continue
            selected: list[ConceptPair] | None = None
            target_key: tuple[str, ...] | None = None
            for _ in range(config.questions.max_sampling_attempts_per_question):
                candidate = rng.sample(pool, config.questions.target_count)
                key = tuple(sorted(item.pair_id for item in candidate))
                if key not in used_target_sets[stem.pair_id]:
                    selected = candidate
                    target_key = key
                    break
            if selected is None or target_key is None:
                continue
            used_target_sets[stem.pair_id].add(target_key)
            rng.shuffle(selected)

            stem_rank = _ranking_scores(ranking_map[stem.pair_id])
            stem_evidence = evidence_map[stem.pair_id]
            stem_context_vectors = [
                cache.get(config.openai.embedding_model, sentence.text)
                for sentence in stem_evidence.evidence_sentences
            ]
            stem_context_vectors = [vector for vector in stem_context_vectors if vector is not None]
            stem_proto_vector = cache.get(
                config.openai.embedding_model,
                _stem_relation_text(stem),
            )
            if stem_proto_vector is None:
                raise RuntimeError("Missing cached stem prototypicality embedding")

            relation_scores: list[float] = []
            context_scores: list[float] = []
            prototype_scores: list[float] = []

            for target in selected:
                target_rank = _ranking_scores(ranking_map[target.pair_id])
                relation_scores.append(
                    ranked_relational_overlap(stem_rank, target_rank)
                )

                target_evidence = evidence_map[target.pair_id]
                target_vectors = [
                    cache.get(config.openai.embedding_model, sentence.text)
                    for sentence in target_evidence.evidence_sentences
                ]
                target_vectors = [vector for vector in target_vectors if vector is not None]
                context_scores.append(
                    context_embedding_similarity(
                        stem_context_vectors,
                        target_vectors,
                        threshold=config.embedding.context_similarity_threshold,
                    )
                )

                target_pair_vector = cache.get(
                    config.openai.embedding_model,
                    _pair_text(target),
                )
                if target_pair_vector is None:
                    raise RuntimeError("Missing cached target-pair embedding")
                prototype_scores.append(
                    prototypical_similarity(stem_proto_vector, target_pair_vector)
                )

            all_scores = {
                "ranked_relational_overlap": relation_scores,
                "context_embedding_similarity": context_scores,
                "prototypicality": prototype_scores,
            }
            answers: dict[str, int] = {}
            ties: dict[str, list[int]] = {}
            answer_pair_ids: dict[str, str] = {}
            for measure, values in all_scores.items():
                answer, tied = argmax_with_ties(
                    values,
                    config.questions.tie_tolerance,
                )
                answers[measure] = answer
                ties[measure] = tied
                answer_pair_ids[measure] = selected[answer].pair_id

            occurrence = occurrence_counter[stem.pair_id]
            occurrence_counter[stem.pair_id] += 1
            question_id = stable_question_id(
                stem.pair_id,
                [item.pair_id for item in selected],
                occurrence,
            )
            questions.append(
                QuestionRecord(
                    question_id=question_id,
                    base_relation=stem.base_relation,
                    stem=TargetPair(
                        pair_id=stem.pair_id,
                        concept_1=stem.concept_1,
                        concept_2=stem.concept_2,
                    ),
                    targets=[
                        TargetPair(
                            pair_id=target.pair_id,
                            concept_1=target.concept_1,
                            concept_2=target.concept_2,
                        )
                        for target in selected
                    ],
                    scores=all_scores,
                    answers=answers,
                    answer_pair_ids=answer_pair_ids,
                    ties=ties,
                )
            )
    finally:
        cache.close()

    output = config.paths.output_dir / "analogy_questions.jsonl"
    write_jsonl(output, questions)
    if (
        config.questions.total_questions is not None
        and len(questions) < config.questions.total_questions
    ):
        warning_path = config.paths.output_dir / "generation_warning.txt"
        warning_path.write_text(
            f"Requested {config.questions.total_questions} questions but generated "
            f"{len(questions)}. Some relation groups or stems did not have enough "
            "unique, relation-overlapping target sets.\n",
            encoding="utf-8",
        )
    return output
