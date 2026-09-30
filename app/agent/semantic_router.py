from __future__ import annotations

import asyncio
import os
import re
from dataclasses import dataclass
from functools import lru_cache
from typing import Any


class SemanticRouterError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class SemanticRoute:
    scope: str
    source: str
    language: str
    score: float
    label: str


@lru_cache(maxsize=1)
def _load_classifier(model_name: str) -> Any:
    from transformers import pipeline

    return pipeline(
        "zero-shot-classification",
        model=model_name,
        device=-1,
    )


def _pick_language(text: str) -> str:
    patterns = (
        ("Hindi/Devanagari", r"[\u0900-\u097F]"),
        ("Kannada", r"[\u0C80-\u0CFF]"),
        ("Telugu", r"[\u0C00-\u0C7F]"),
        ("Tamil", r"[\u0B80-\u0BFF]"),
        ("Malayalam", r"[\u0D00-\u0D7F]"),
        ("Bengali", r"[\u0980-\u09FF]"),
        ("Gujarati", r"[\u0A80-\u0AFF]"),
        ("Gurmukhi", r"[\u0A00-\u0A7F]"),
    )

    for language, pattern in patterns:
        if re.search(pattern, text or ""):
            return language

    return "same-as-user"


class LocalSemanticRouter:
    """
    Free/local semantic router.

    No company names, language phrases, slang dictionaries, or question
    templates are hard-coded. The active company is supplied dynamically.
    """

    def __init__(self, model_name: str | None = None) -> None:
        self.model_name = (
            model_name or os.getenv("SEMANTIC_ROUTER_MODEL", "")
        ).strip()

        if not self.model_name:
            raise SemanticRouterError(
                "SEMANTIC_ROUTER_MODEL is not configured in .env"
            )

    async def classify(
        self,
        question: str,
        company_name: str = "",
        recent_context: str = "",
    ) -> SemanticRoute:
        question = str(question or "").strip()

        if not question:
            return SemanticRoute(
                scope="unclear",
                source="company",
                language="same-as-user",
                score=0.0,
                label="empty",
            )

        context = str(recent_context or "")
        if len(context) > 600:
            context = context[-600:]

        sequence = (
            f"Active company: {company_name}\n"
            f"User request: {question}\n"
            f"Recent context: {context}"
        )

        candidate_labels = [
            "personal/unrelated",
            "business/company knowledge",
            "current/public web information",
            "both company knowledge and current/public web information",
        ]

        try:
            classifier = _load_classifier(self.model_name)

            result = await asyncio.to_thread(
                classifier,
                sequence,
                candidate_labels=candidate_labels,
                multi_label=False,
                hypothesis_template="This is {}.",
                truncation=True,
            )

            labels = result.get("labels") or []
            scores = result.get("scores") or []

            label = str(labels[0]) if labels else "unclear"
            score = float(scores[0]) if scores else 0.0

            normalized = label.lower()

            if normalized == "personal/unrelated":
                scope = "out_of_scope"
                source = "company"
            elif normalized.startswith("both"):
                scope = "business"
                source = "both"
            elif normalized == "current/public web information":
                scope = "business"
                source = "web"
            else:
                scope = "business"
                source = "company"

            return SemanticRoute(
                scope=scope,
                source=source,
                language=_pick_language(question),
                score=score,
                label=label,
            )

        except SemanticRouterError:
            raise
        except Exception as exc:
            raise SemanticRouterError(
                f"local semantic router unavailable: {type(exc).__name__}"
            ) from exc