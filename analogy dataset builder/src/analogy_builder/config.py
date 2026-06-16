from __future__ import annotations

from pathlib import Path
from typing import Optional

import yaml
from pydantic import BaseModel, Field, model_validator


class PathConfig(BaseModel):
    input_csv: Path = Path("data/input/concept_pairs.csv")
    work_dir: Path = Path("data/work")
    review_dir: Path = Path("data/review")
    output_dir: Path = Path("data/output")
    cache_dir: Path = Path("data/cache")
    relation_aliases_file: Optional[Path] = Path("relation_aliases.yaml")


class WikipediaConfig(BaseModel):
    language: str = "en"
    min_sentences: int = 5
    search_results: int = 8
    max_pages: int = 10
    request_delay_seconds: float = 0.15
    request_timeout_seconds: float = 30.0
    max_sentence_chars: int = 900
    min_sentence_chars: int = 25
    allow_pairs_below_minimum: bool = False
    user_agent: str = (
        "AnalogyDatasetBuilder/0.1 "
        "(research code; set your contact email in config.yaml)"
    )


class OpenAIConfig(BaseModel):
    relation_model: str = "gpt-4.1"
    ranking_model: str = "gpt-4.1"
    embedding_model: str = "text-embedding-3-large"
    max_retries: int = 5
    relation_context_char_limit: int = 24_000


class EmbeddingConfig(BaseModel):
    batch_size: int = 64
    context_similarity_threshold: float = 0.5


class MaxDiffConfig(BaseModel):
    subset_size: int = 4
    target_appearances_per_relation: int = 4
    score_offset: float = 1.1
    seed: int = 42


class QuestionConfig(BaseModel):
    target_count: int = 4
    questions_per_stem: int = 1
    total_questions: Optional[int] = None
    seed: int = 42
    require_context_pass: bool = True
    tie_tolerance: float = 1e-10
    max_sampling_attempts_per_question: int = 200


class RelationFilterConfig(BaseModel):
    auto_accept_base_relation: bool = True
    auto_reject_unsupported: bool = False


class AppConfig(BaseModel):
    paths: PathConfig = Field(default_factory=PathConfig)
    wikipedia: WikipediaConfig = Field(default_factory=WikipediaConfig)
    openai: OpenAIConfig = Field(default_factory=OpenAIConfig)
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    maxdiff: MaxDiffConfig = Field(default_factory=MaxDiffConfig)
    questions: QuestionConfig = Field(default_factory=QuestionConfig)
    relation_filter: RelationFilterConfig = Field(default_factory=RelationFilterConfig)
    config_path: Optional[Path] = None

    @model_validator(mode="after")
    def resolve_paths(self) -> "AppConfig":
        base = self.config_path.parent.resolve() if self.config_path else Path.cwd()
        for name in (
            "input_csv",
            "work_dir",
            "review_dir",
            "output_dir",
            "cache_dir",
            "relation_aliases_file",
        ):
            value = getattr(self.paths, name)
            if value is not None and not value.is_absolute():
                setattr(self.paths, name, (base / value).resolve())
        return self

    def ensure_directories(self) -> None:
        for path in (
            self.paths.work_dir,
            self.paths.review_dir,
            self.paths.output_dir,
            self.paths.cache_dir,
        ):
            path.mkdir(parents=True, exist_ok=True)


DEFAULT_CONFIG_PATH = Path("config.yaml")


def load_config(path: Path | str = DEFAULT_CONFIG_PATH) -> AppConfig:
    config_path = Path(path).resolve()
    if not config_path.exists():
        raise FileNotFoundError(
            f"Configuration file not found: {config_path}. "
            "Copy config.example.yaml to config.yaml first."
        )
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    raw["config_path"] = config_path
    config = AppConfig.model_validate(raw)
    config.ensure_directories()
    return config
