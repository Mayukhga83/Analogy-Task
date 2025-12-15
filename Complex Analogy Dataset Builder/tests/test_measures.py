import numpy as np

from analogy_builder.measures import (
    argmax_with_ties,
    context_embedding_similarity,
    ranked_relational_overlap,
)


def test_ranked_relational_overlap_uses_shared_relations_only():
    stem = {"capital": 2.0, "largest_city": 1.2}
    target = {"capital": 1.5, "located_in": 0.9}
    assert ranked_relational_overlap(stem, target) == 3.0


def test_context_similarity_threshold_and_mean():
    stem = [np.array([1.0, 0.0])]
    target = [np.array([1.0, 0.0]), np.array([0.0, 1.0])]
    assert context_embedding_similarity(stem, target, threshold=0.5) == 1.0


def test_argmax_reports_ties():
    answer, ties = argmax_with_ties([0.1, 0.5, 0.5], tolerance=1e-12)
    assert answer == 1
    assert ties == [1, 2]
