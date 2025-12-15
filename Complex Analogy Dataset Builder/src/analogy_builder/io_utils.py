from __future__ import annotations

import csv
import hashlib
import json
import re
import tempfile
import unicodedata
from pathlib import Path
from typing import Any, Iterable, Iterator, TypeVar

from pydantic import BaseModel

from .models import ConceptPair

T = TypeVar("T", bound=BaseModel)


def normalize_space(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


def relation_slug(value: str) -> str:
    value = normalize_space(value)
    value = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", value)
    value = unicodedata.normalize("NFKD", value)
    value = "".join(ch for ch in value if not unicodedata.combining(ch))
    value = value.casefold()
    value = re.sub(r"[^a-z0-9]+", "_", value)
    return re.sub(r"_+", "_", value).strip("_")


def stable_pair_id(base_relation: str, concept_1: str, concept_2: str) -> str:
    payload = "\0".join(
        [relation_slug(base_relation), concept_1.casefold().strip(), concept_2.casefold().strip()]
    )
    return f"pair_{hashlib.sha1(payload.encode('utf-8')).hexdigest()[:12]}"


def stable_question_id(stem_id: str, target_ids: list[str], occurrence: int) -> str:
    payload = "\0".join([stem_id, *target_ids, str(occurrence)])
    return f"q_{hashlib.sha1(payload.encode('utf-8')).hexdigest()[:14]}"


def load_concept_pairs(path: Path) -> list[ConceptPair]:
    if not path.exists():
        raise FileNotFoundError(f"Input concept-pair file not found: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"base_relation", "concept_1", "concept_2"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(
                f"Input CSV is missing columns: {sorted(missing)}. "
                "Required columns: base_relation, concept_1, concept_2."
            )
        records: list[ConceptPair] = []
        seen_ids: set[str] = set()
        seen_triples: set[tuple[str, str, str]] = set()
        for row_number, row in enumerate(reader, start=2):
            base_relation = normalize_space(row.get("base_relation", ""))
            concept_1 = normalize_space(row.get("concept_1", ""))
            concept_2 = normalize_space(row.get("concept_2", ""))
            if not all((base_relation, concept_1, concept_2)):
                raise ValueError(f"Blank required value in row {row_number}")
            triple = (
                relation_slug(base_relation),
                concept_1.casefold(),
                concept_2.casefold(),
            )
            if triple in seen_triples:
                continue
            seen_triples.add(triple)
            pair_id = normalize_space(row.get("pair_id", "")) or stable_pair_id(
                base_relation, concept_1, concept_2
            )
            if pair_id in seen_ids:
                raise ValueError(f"Duplicate pair_id '{pair_id}' in row {row_number}")
            seen_ids.add(pair_id)
            records.append(
                ConceptPair(
                    pair_id=pair_id,
                    base_relation=relation_slug(base_relation),
                    concept_1=concept_1,
                    concept_2=concept_2,
                )
            )
    if not records:
        raise ValueError("Input CSV contains no usable concept pairs")
    return records


def read_jsonl(path: Path, model: type[T] | None = None) -> Iterator[T | dict[str, Any]]:
    if not path.exists():
        return iter(())

    def generator() -> Iterator[T | dict[str, Any]]:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSONL in {path}, line {line_number}: {exc}") from exc
                yield model.model_validate(obj) if model else obj

    return generator()


def write_jsonl(path: Path, records: Iterable[BaseModel | dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, delete=False, newline="\n"
    ) as handle:
        temp_path = Path(handle.name)
        for record in records:
            if isinstance(record, BaseModel):
                payload = record.model_dump(mode="json")
            else:
                payload = record
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
    temp_path.replace(path)


def append_jsonl(path: Path, record: BaseModel | dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = record.model_dump(mode="json") if isinstance(record, BaseModel) else record
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def load_existing_ids(path: Path, id_field: str = "pair_id") -> set[str]:
    if not path.exists():
        return set()
    return {
        str(record[id_field])
        for record in read_jsonl(path)
        if isinstance(record, dict) and id_field in record
    }


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, delete=False
    ) as handle:
        temp_path = Path(handle.name)
        handle.write(text)
    temp_path.replace(path)
