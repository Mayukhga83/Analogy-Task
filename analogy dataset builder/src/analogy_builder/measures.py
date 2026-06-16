from __future__ import annotations

import numpy as np


def cosine_similarity(left: np.ndarray, right: np.ndarray) -> float:
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    if denominator == 0.0:
        return 0.0
    return float(np.dot(left, right) / denominator)


def ranked_relational_overlap(
    stem_scores: dict[str, float],
    target_scores: dict[str, float],
) -> float:
    shared = set(stem_scores) & set(target_scores)
    return float(
        sum(stem_scores[relation] * target_scores[relation] for relation in shared)
    )


def context_embedding_similarity(
    stem_vectors: list[np.ndarray],
    target_vectors: list[np.ndarray],
    threshold: float = 0.5,
) -> float:
    if not stem_vectors or not target_vectors:
        return 0.0
    retained: list[float] = []
    for stem_vector in stem_vectors:
        for target_vector in target_vectors:
            score = cosine_similarity(stem_vector, target_vector)
            if score > threshold:
                retained.append(score)
    if not retained:
        return 0.0
    # The paper describes summing similarities above the threshold and
    # normalizing by the number that pass it, which is their arithmetic mean.
    return float(np.mean(retained))


def prototypical_similarity(
    stem_with_relation_vector: np.ndarray,
    target_pair_vector: np.ndarray,
) -> float:
    return cosine_similarity(stem_with_relation_vector, target_pair_vector)


def argmax_with_ties(values: list[float], tolerance: float) -> tuple[int, list[int]]:
    if not values:
        raise ValueError("Cannot choose an answer from an empty score list")
    maximum = max(values)
    ties = [index for index, value in enumerate(values) if abs(value - maximum) <= tolerance]
    return ties[0], ties
