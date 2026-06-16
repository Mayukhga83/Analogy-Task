from __future__ import annotations

from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field


class ConceptPair(BaseModel):
    pair_id: str
    base_relation: str
    concept_1: str
    concept_2: str


class EvidenceSentence(BaseModel):
    sentence_id: int
    text: str
    page_title: str
    page_url: str


class WikiEvidenceRecord(BaseModel):
    pair_id: str
    base_relation: str
    concept_1: str
    concept_2: str
    resolved_titles: list[str] = Field(default_factory=list)
    pages_considered: list[str] = Field(default_factory=list)
    evidence_sentences: list[EvidenceSentence] = Field(default_factory=list)
    passes_minimum: bool = False
    error: Optional[str] = None


class ExtractedRelation(BaseModel):
    predicate: str
    evidence_sentence_ids: list[int] = Field(default_factory=list)
    evidence_quote: Optional[str] = None
    explanation: Optional[str] = None


class RelationExtraction(BaseModel):
    relations: list[ExtractedRelation] = Field(default_factory=list)


class MinedRelation(BaseModel):
    raw_predicate: str
    normalized_predicate: str
    evidence_sentence_ids: list[int] = Field(default_factory=list)
    evidence_quote: Optional[str] = None
    explanation: Optional[str] = None
    source: str = "openai"


class MinedRelationRecord(BaseModel):
    pair_id: str
    base_relation: str
    concept_1: str
    concept_2: str
    relations: list[MinedRelation] = Field(default_factory=list)
    error: Optional[str] = None


class CanonicalRelation(BaseModel):
    relation: str
    evidence_sentence_ids: list[int] = Field(default_factory=list)
    source: str
    notes: Optional[str] = None


class CanonicalRelationRecord(BaseModel):
    pair_id: str
    base_relation: str
    concept_1: str
    concept_2: str
    relations: list[CanonicalRelation] = Field(default_factory=list)


class MaxDiffChoice(BaseModel):
    most_relevant: str
    least_relevant: str
    rationale: Optional[str] = None


class RankedRelation(BaseModel):
    relation: str
    appearances: int
    most_count: int
    least_count: int
    p_most: float
    p_least: float
    score: float
    rank: int


class RelationRankingRecord(BaseModel):
    pair_id: str
    base_relation: str
    concept_1: str
    concept_2: str
    relations: list[RankedRelation]
    blocks: list[dict[str, Any]] = Field(default_factory=list)
    error: Optional[str] = None


class ReviewDecision(str, Enum):
    accept = "accept"
    reject = "reject"
    review = "review"


class ReviewCategory(str, Enum):
    valid = "valid"
    duplicate = "duplicate"
    ambiguous = "ambiguous"
    opinion_based = "opinion_based"
    transient = "transient"
    unsupported = "unsupported"
    malformed = "malformed"
    base_relation = "base_relation"


class TargetPair(BaseModel):
    pair_id: str
    concept_1: str
    concept_2: str


class QuestionRecord(BaseModel):
    question_id: str
    base_relation: str
    stem: TargetPair
    targets: list[TargetPair]
    scores: dict[str, list[float]]
    answers: dict[str, int]
    answer_pair_ids: dict[str, str]
    ties: dict[str, list[int]]
