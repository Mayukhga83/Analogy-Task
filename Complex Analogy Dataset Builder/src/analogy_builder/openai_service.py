from __future__ import annotations

import os
from typing import TypeVar

from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential_jitter

from .config import AppConfig

T = TypeVar("T", bound=BaseModel)


class OpenAIService:
    def __init__(self, config: AppConfig):
        load_dotenv()
        if not os.getenv("OPENAI_API_KEY"):
            raise RuntimeError(
                "OPENAI_API_KEY is not set. Add it to a .env file or your environment."
            )
        self.config = config
        self.client = OpenAI()

    def parse(self, model: str, system: str, user: str, schema: type[T]) -> T:
        attempts = self.config.openai.max_retries

        @retry(
            stop=stop_after_attempt(attempts),
            wait=wait_exponential_jitter(initial=1, max=30),
            retry=retry_if_exception_type(Exception),
            reraise=True,
        )
        def _call() -> T:
            response = self.client.responses.parse(
                model=model,
                input=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                text_format=schema,
            )
            parsed = response.output_parsed
            if parsed is None:
                raise RuntimeError("The model returned no parsed structured output")
            return parsed

        return _call()

    def embed(self, texts: list[str], model: str) -> list[list[float]]:
        if not texts:
            return []
        attempts = self.config.openai.max_retries

        @retry(
            stop=stop_after_attempt(attempts),
            wait=wait_exponential_jitter(initial=1, max=30),
            retry=retry_if_exception_type(Exception),
            reraise=True,
        )
        def _call() -> list[list[float]]:
            response = self.client.embeddings.create(
                model=model,
                input=texts,
                encoding_format="float",
            )
            ordered = sorted(response.data, key=lambda item: item.index)
            return [item.embedding for item in ordered]

        return _call()
