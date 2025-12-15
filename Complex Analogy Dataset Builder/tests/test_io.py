from pathlib import Path

from analogy_builder.io_utils import load_concept_pairs, relation_slug


def test_relation_slug_normalizes_camel_case_and_spaces():
    assert relation_slug("Located In") == "located_in"
    assert relation_slug("isLocatedIn") == "is_located_in"


def test_load_pairs_deduplicates_exact_rows(tmp_path: Path):
    path = tmp_path / "pairs.csv"
    path.write_text(
        "base_relation,concept_1,concept_2\n"
        "capital,Paris,France\n"
        "capital,Paris,France\n",
        encoding="utf-8",
    )
    pairs = load_concept_pairs(path)
    assert len(pairs) == 1
