from __future__ import annotations

import html
import re
import time
from pathlib import Path
from typing import Iterable
from urllib.parse import quote

import requests
from tqdm import tqdm

from .config import AppConfig
from .io_utils import append_jsonl, load_concept_pairs, load_existing_ids, normalize_space
from .models import ConceptPair, EvidenceSentence, WikiEvidenceRecord


_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'“‘(])")


def _clean_wiki_text(text: str) -> str:
    text = html.unescape(text)
    text = re.sub(r"\[[0-9]+\]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def split_sentences(text: str, min_chars: int, max_chars: int) -> list[str]:
    cleaned = _clean_wiki_text(text)
    candidates = _SENTENCE_SPLIT.split(cleaned)
    result: list[str] = []
    for sentence in candidates:
        sentence = normalize_space(sentence)
        if min_chars <= len(sentence) <= max_chars:
            result.append(sentence)
    return result


def alias_variants(label: str) -> set[str]:
    label = normalize_space(label)
    variants = {
        label.casefold(),
        label.replace("_", " ").casefold(),
        re.sub(r"\s*\([^)]*\)\s*", "", label).strip().casefold(),
    }
    if label.startswith("The "):
        variants.add(label[4:].casefold())
    return {value for value in variants if value}


def contains_any(sentence: str, aliases: Iterable[str]) -> bool:
    folded = sentence.casefold()
    return any(alias in folded for alias in aliases)


class MediaWikiClient:
    def __init__(self, config: AppConfig):
        self.config = config.wikipedia
        self.api_url = f"https://{self.config.language}.wikipedia.org/w/api.php"
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": self.config.user_agent})
        self._last_request = 0.0

    def _get(self, params: dict) -> dict:
        elapsed = time.time() - self._last_request
        if elapsed < self.config.request_delay_seconds:
            time.sleep(self.config.request_delay_seconds - elapsed)
        response = self.session.get(
            self.api_url,
            params=params,
            timeout=self.config.request_timeout_seconds,
        )
        self._last_request = time.time()
        response.raise_for_status()
        return response.json()

    def search_titles(self, query: str, limit: int) -> list[str]:
        payload = self._get(
            {
                "action": "query",
                "list": "search",
                "srsearch": query,
                "srlimit": limit,
                "format": "json",
                "utf8": 1,
            }
        )
        return [entry["title"] for entry in payload.get("query", {}).get("search", [])]

    def resolve_title(self, query: str) -> str | None:
        titles = self.search_titles(query, 1)
        return titles[0] if titles else None

    def get_extract(self, title: str) -> tuple[str, str] | None:
        payload = self._get(
            {
                "action": "query",
                "prop": "extracts",
                "explaintext": 1,
                "redirects": 1,
                "titles": title,
                "format": "json",
                "utf8": 1,
            }
        )
        pages = payload.get("query", {}).get("pages", {})
        for page in pages.values():
            if "missing" in page:
                return None
            resolved_title = page.get("title", title)
            extract = page.get("extract", "")
            if extract:
                return resolved_title, extract
        return None

    def page_url(self, title: str) -> str:
        safe_title = quote(title.replace(" ", "_"))
        return f"https://{self.config.language}.wikipedia.org/wiki/{safe_title}"


def retrieve_pair_evidence(pair: ConceptPair, client: MediaWikiClient) -> WikiEvidenceRecord:
    try:
        title_1 = client.resolve_title(pair.concept_1)
        title_2 = client.resolve_title(pair.concept_2)
        candidate_titles: list[str] = []
        for title in (title_1, title_2):
            if title and title not in candidate_titles:
                candidate_titles.append(title)

        search_queries = [
            f'"{pair.concept_1}" "{pair.concept_2}"',
            f"{pair.concept_1} {pair.concept_2}",
        ]
        for query in search_queries:
            for title in client.search_titles(query, client.config.search_results):
                if title not in candidate_titles:
                    candidate_titles.append(title)
                if len(candidate_titles) >= client.config.max_pages:
                    break
            if len(candidate_titles) >= client.config.max_pages:
                break

        aliases_1 = alias_variants(pair.concept_1)
        aliases_2 = alias_variants(pair.concept_2)
        for title in (title_1, title_2):
            if title:
                if title == title_1:
                    aliases_1 |= alias_variants(title)
                if title == title_2:
                    aliases_2 |= alias_variants(title)

        evidence: list[EvidenceSentence] = []
        seen_sentences: set[str] = set()
        pages_considered: list[str] = []
        resolved_titles: list[str] = [title for title in (title_1, title_2) if title]

        for title in candidate_titles[: client.config.max_pages]:
            extracted = client.get_extract(title)
            if not extracted:
                continue
            resolved_title, text = extracted
            pages_considered.append(resolved_title)
            for sentence in split_sentences(
                text,
                min_chars=client.config.min_sentence_chars,
                max_chars=client.config.max_sentence_chars,
            ):
                if not (contains_any(sentence, aliases_1) and contains_any(sentence, aliases_2)):
                    continue
                key = sentence.casefold()
                if key in seen_sentences:
                    continue
                seen_sentences.add(key)
                evidence.append(
                    EvidenceSentence(
                        sentence_id=len(evidence),
                        text=sentence,
                        page_title=resolved_title,
                        page_url=client.page_url(resolved_title),
                    )
                )

        return WikiEvidenceRecord(
            pair_id=pair.pair_id,
            base_relation=pair.base_relation,
            concept_1=pair.concept_1,
            concept_2=pair.concept_2,
            resolved_titles=resolved_titles,
            pages_considered=pages_considered,
            evidence_sentences=evidence,
            passes_minimum=len(evidence) >= client.config.min_sentences,
        )
    except Exception as exc:  # noqa: BLE001 - persist per-pair failures and continue
        return WikiEvidenceRecord(
            pair_id=pair.pair_id,
            base_relation=pair.base_relation,
            concept_1=pair.concept_1,
            concept_2=pair.concept_2,
            error=f"{type(exc).__name__}: {exc}",
        )


def retrieve_all_wikipedia(config: AppConfig, resume: bool = True) -> Path:
    pairs = load_concept_pairs(config.paths.input_csv)
    output = config.paths.work_dir / "wikipedia_evidence.jsonl"
    completed = load_existing_ids(output) if resume else set()
    if not resume and output.exists():
        output.unlink()
    client = MediaWikiClient(config)

    for pair in tqdm(pairs, desc="Retrieving Wikipedia evidence"):
        if pair.pair_id in completed:
            continue
        record = retrieve_pair_evidence(pair, client)
        append_jsonl(output, record)
    return output
