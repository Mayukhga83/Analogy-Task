from collections import Counter

from analogy_builder.maxdiff import build_balanced_blocks


def test_balanced_blocks_have_valid_size_and_coverage():
    relations = [f"r{i}" for i in range(6)]
    blocks = build_balanced_blocks(
        relations,
        subset_size=4,
        target_appearances=4,
        seed=7,
    )
    assert all(len(block) == 4 for block in blocks)
    assert all(len(block) == len(set(block)) for block in blocks)
    counts = Counter(item for block in blocks for item in block)
    assert set(counts) == set(relations)
    assert max(counts.values()) - min(counts.values()) <= 1


def test_three_relations_use_two_item_blocks():
    blocks = build_balanced_blocks(["a", "b", "c"], seed=1)
    assert blocks
    assert all(len(block) == 2 for block in blocks)
