from __future__ import annotations

import math
import random
from collections import Counter
from itertools import combinations
from pathlib import Path

from tqdm import tqdm

from .config import AppConfig
from .io_utils import append_jsonl, load_existing_ids, read_jsonl, relation_slug
from .models import (
    CanonicalRelationRecord,
    MaxDiffChoice,
    RankedRelation,
    RelationRankingRecord,
)
from .openai_service import OpenAIService


def build_balanced_blocks(
    relations: list[str],
    subset_size: int = 4,
    target_appearances: int = 4,
    seed: int = 42,
) -> list[list[str]]:
    """Create approximately balanced Max-Diff blocks.

    For 2-3 relations, two-item blocks are used. For four or more relations,
    the configured subset size is used. Each relation appears approximately
    target_appearances times. The greedy design also discourages repeating the
    same relation pairs and exact blocks.
    """
    unique = list(dict.fromkeys(relations))
    n = len(unique)
    if n <= 1:
        return [unique] if unique else []

    k = min(subset_size, n)
    if n in (2, 3):
        k = 2

    target_blocks = max(1, math.ceil(n * target_appearances / k))
    rng = random.Random(seed)
    appearances: Counter[str] = Counter()
    pair_counts: Counter[tuple[str, str]] = Counter()
    used_blocks: Counter[tuple[str, ...]] = Counter()
    blocks: list[list[str]] = []

    # For very small spaces, cycling through combinations gives excellent balance.
    all_combos = list(combinations(unique, k)) if n <= 10 else []

    for block_index in range(target_blocks):
        if all_combos:
            scored: list[tuple[tuple[float, ...], tuple[str, ...]]] = []
            for combo in all_combos:
                sorted_combo = tuple(sorted(combo))
                future_counts = [appearances[item] + 1 for item in combo]
                pair_penalty = sum(
                    pair_counts[tuple(sorted(pair))] for pair in combinations(combo, 2)
                )
                score = (
                    max(future_counts),
                    sum(future_counts),
                    pair_penalty,
                    used_blocks[sorted_combo],
                    rng.random(),
                )
                scored.append((score, combo))
            _, selected_tuple = min(scored, key=lambda item: item[0])
            selected = list(selected_tuple)
        else:
            # Greedy fallback for larger relation inventories.
            candidates = unique[:]
            rng.shuffle(candidates)
            candidates.sort(key=lambda item: (appearances[item], rng.random()))
            selected = [candidates[0]]
            while len(selected) < k:
                remaining = [item for item in unique if item not in selected]
                remaining.sort(
                    key=lambda item: (
                        appearances[item],
                        sum(
                            pair_counts[tuple(sorted((item, chosen)))]
                            for chosen in selected
                        ),
                        rng.random(),
                    )
                )
                selected.append(remaining[0])

        rng.shuffle(selected)
        blocks.append(selected)
        for relation in selected:
            appearances[relation] += 1
        for left, right in combinations(selected, 2):
            pair_counts[tuple(sorted((left, right)))] += 1
        used_blocks[tuple(sorted(selected))] += 1

    return blocks


def _maxdiff_prompt(
    concept_1: str,
    concept_2: str,
    relations: list[str],
) -> tuple[str, str]:
    system = (
        "You are performing a Maximum Difference (best-worst) judgment. "
        "Choose only from the supplied relation labels. Judge how central each "
        "relation is to the specific concept pair, not how generally important the "
        "relation is in the world."
    )
    relation_lines = "\n".join(f"- {relation}" for relation in relations)
    user = f"""
Concept pair: {concept_1} -> {concept_2}

Relations:
{relation_lines}

Select:
1. The relation that is MOST relevant or defining for this concept pair.
2. The relation that is LEAST relevant or defining for this concept pair.

Return the relation labels exactly as written.
""".strip()
    return system, user


def _resolve_choice(value: str, available: list[str]) -> str:
    if value in available:
        return value
    slug = relation_slug(value)
    by_slug = {relation_slug(item): item for item in available}
    if slug in by_slug:
        return by_slug[slug]
    raise ValueError(f"Model selected '{value}', which is not in {available}")


def rank_all_relations(config: AppConfig, resume: bool = True) -> Path:
    canonical_path = config.paths.work_dir / "canonical_relations.jsonl"
    output = config.paths.work_dir / "relation_rankings.jsonl"
    if not resume and output.exists():
        output.unlink()
    completed = load_existing_ids(output) if resume else set()
    service = OpenAIService(config)

    records = list(read_jsonl(canonical_path, CanonicalRelationRecord))
    for record in tqdm(records, desc="Ranking relations with Max-Diff"):
        if record.pair_id in completed:
            continue
        relations = [item.relation for item in record.relations]
        if not relations:
            append_jsonl(
                output,
                RelationRankingRecord(
                    pair_id=record.pair_id,
                    base_relation=record.base_relation,
                    concept_1=record.concept_1,
                    concept_2=record.concept_2,
                    relations=[],
                    error="No accepted relations",
                ),
            )
            continue

        if len(relations) == 1:
            single = RankedRelation(
                relation=relations[0],
                appearances=1,
                most_count=1,
                least_count=0,
                p_most=1.0,
                p_least=0.0,
                score=1.0 + config.maxdiff.score_offset,
                rank=1,
            )
            append_jsonl(
                output,
                RelationRankingRecord(
                    pair_id=record.pair_id,
                    base_relation=record.base_relation,
                    concept_1=record.concept_1,
                    concept_2=record.concept_2,
                    relations=[single],
                    blocks=[
                        {
                            "block_id": 0,
                            "relations": relations,
                            "most_relevant": relations[0],
                            "least_relevant": None,
                            "rationale": "Only one relation was available.",
                        }
                    ],
                ),
            )
            continue

        try:
            blocks = build_balanced_blocks(
                relations,
                subset_size=config.maxdiff.subset_size,
                target_appearances=config.maxdiff.target_appearances_per_relation,
                seed=config.maxdiff.seed + int(record.pair_id[-4:], 16)
                if all(ch in "0123456789abcdef" for ch in record.pair_id[-4:].casefold())
                else config.maxdiff.seed,
            )
            appearances: Counter[str] = Counter()
            most_counts: Counter[str] = Counter()
            least_counts: Counter[str] = Counter()
            block_results: list[dict] = []

            for block_id, block in enumerate(blocks):
                system, user = _maxdiff_prompt(
                    record.concept_1,
                    record.concept_2,
                    block,
                )
                parsed = service.parse(
                    config.openai.ranking_model,
                    system,
                    user,
                    MaxDiffChoice,
                )
                most = _resolve_choice(parsed.most_relevant, block)
                least = _resolve_choice(parsed.least_relevant, block)
                if most == least:
                    raise ValueError("Max-Diff most and least choices are identical")
                for relation in block:
                    appearances[relation] += 1
                most_counts[most] += 1
                least_counts[least] += 1
                block_results.append(
                    {
                        "block_id": block_id,
                        "relations": block,
                        "most_relevant": most,
                        "least_relevant": least,
                        "rationale": parsed.rationale,
                    }
                )

            raw_rows: list[dict] = []
            for relation in relations:
                count = appearances[relation]
                p_most = most_counts[relation] / count if count else 0.0
                p_least = least_counts[relation] / count if count else 0.0
                score = p_most - p_least + config.maxdiff.score_offset
                raw_rows.append(
                    {
                        "relation": relation,
                        "appearances": count,
                        "most_count": most_counts[relation],
                        "least_count": least_counts[relation],
                        "p_most": p_most,
                        "p_least": p_least,
                        "score": score,
                    }
                )
            raw_rows.sort(key=lambda item: (-item["score"], item["relation"]))
            ranked = [
                RankedRelation(**row, rank=index)
                for index, row in enumerate(raw_rows, start=1)
            ]
            append_jsonl(
                output,
                RelationRankingRecord(
                    pair_id=record.pair_id,
                    base_relation=record.base_relation,
                    concept_1=record.concept_1,
                    concept_2=record.concept_2,
                    relations=ranked,
                    blocks=block_results,
                ),
            )
        except Exception as exc:  # noqa: BLE001
            append_jsonl(
                output,
                RelationRankingRecord(
                    pair_id=record.pair_id,
                    base_relation=record.base_relation,
                    concept_1=record.concept_1,
                    concept_2=record.concept_2,
                    relations=[],
                    error=f"{type(exc).__name__}: {exc}",
                ),
            )
    return output
