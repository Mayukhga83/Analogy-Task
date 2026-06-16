from __future__ import annotations

import hashlib
import sqlite3
from pathlib import Path

import numpy as np
from tqdm import tqdm

from .config import AppConfig
from .io_utils import read_jsonl
from .models import WikiEvidenceRecord
from .openai_service import OpenAIService


class EmbeddingCache:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self.connection = sqlite3.connect(path)
        self.connection.execute(
            """
            CREATE TABLE IF NOT EXISTS embeddings (
                cache_key TEXT PRIMARY KEY,
                model TEXT NOT NULL,
                text_value TEXT NOT NULL,
                dimensions INTEGER NOT NULL,
                vector BLOB NOT NULL
            )
            """
        )
        self.connection.commit()

    @staticmethod
    def key(model: str, text: str) -> str:
        payload = f"{model}\0{text}".encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    def get(self, model: str, text: str) -> np.ndarray | None:
        key = self.key(model, text)
        row = self.connection.execute(
            "SELECT dimensions, vector FROM embeddings WHERE cache_key = ?",
            (key,),
        ).fetchone()
        if row is None:
            return None
        dimensions, blob = row
        vector = np.frombuffer(blob, dtype=np.float32)
        if len(vector) != dimensions:
            raise ValueError(f"Corrupt embedding cache row for key {key}")
        return vector.copy()

    def put(self, model: str, text: str, vector: np.ndarray) -> None:
        vector = np.asarray(vector, dtype=np.float32)
        self.connection.execute(
            """
            INSERT OR REPLACE INTO embeddings
                (cache_key, model, text_value, dimensions, vector)
            VALUES (?, ?, ?, ?, ?)
            """,
            (self.key(model, text), model, text, len(vector), vector.tobytes()),
        )

    def commit(self) -> None:
        self.connection.commit()

    def close(self) -> None:
        self.connection.commit()
        self.connection.close()

    def ensure(
        self,
        texts: list[str],
        model: str,
        service: OpenAIService,
        batch_size: int,
    ) -> None:
        missing = list(dict.fromkeys(text for text in texts if self.get(model, text) is None))
        for start in tqdm(
            range(0, len(missing), batch_size),
            desc="Embedding batches",
            disable=not missing,
        ):
            batch = missing[start : start + batch_size]
            vectors = service.embed(batch, model)
            if len(vectors) != len(batch):
                raise RuntimeError("Embedding API returned a different number of vectors")
            for text, vector in zip(batch, vectors, strict=True):
                self.put(model, text, np.asarray(vector, dtype=np.float32))
            self.commit()


def cache_path(config: AppConfig) -> Path:
    return config.paths.cache_dir / "embeddings.sqlite3"


def warm_context_embeddings(config: AppConfig) -> Path:
    evidence_path = config.paths.work_dir / "wikipedia_evidence.jsonl"
    records = list(read_jsonl(evidence_path, WikiEvidenceRecord))
    texts = [
        sentence.text
        for record in records
        for sentence in record.evidence_sentences
        if record.passes_minimum or config.wikipedia.allow_pairs_below_minimum
    ]
    service = OpenAIService(config)
    cache = EmbeddingCache(cache_path(config))
    try:
        cache.ensure(
            texts,
            config.openai.embedding_model,
            service,
            config.embedding.batch_size,
        )
    finally:
        cache.close()
    return cache_path(config)
