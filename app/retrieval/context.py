"""Build safe, structured context from retrieved evidence."""

from __future__ import annotations

import re
from dataclasses import dataclass, replace
from datetime import date
from typing import Sequence


@dataclass(frozen=True)
class ContextResult:
    """Structured context returned to the LLM/application."""

    text: str
    chunks: list
    stale_ids: list[str] | None = None
    truncated: bool = False


def _safe_attribute(value: object) -> str:
    """Make a value safe for use inside a double-quoted attribute.

    Double quotes are converted to single quotes so normal metadata such as
    Title "quoted" remains readable while not breaking the attribute.
    """
    return str(value).replace('"', "'")


def _neutralize_evidence_delimiters(content: str) -> str:
    """Prevent source content from creating or closing evidence blocks."""

    content = re.sub(
        r"</evidence\s*>",
        "&lt;/evidence&gt;",
        content,
        flags=re.IGNORECASE,
    )

    content = re.sub(
        r"<evidence\b",
        "&lt;evidence",
        content,
        flags=re.IGNORECASE,
    )

    return content


def _estimate_tokens(text: str) -> int:
    """Estimate token usage conservatively."""
    if not text:
        return 0

    return max(1, (len(text) + 3) // 4)


def _attach_metadata(chunk: object, citation_id: str) -> object:
    """Attach a citation ID to a RetrievedChunk."""

    model_copy = getattr(chunk, "model_copy", None)

    if callable(model_copy):
        try:
            return model_copy(
                update={"citation_id": citation_id}
            )
        except Exception:
            pass

    try:
        return replace(
            chunk,
            citation_id=citation_id,
        )
    except (TypeError, ValueError):
        pass

    try:
        setattr(chunk, "citation_id", citation_id)
        return chunk
    except Exception:
        return chunk


def build_context(
    chunks: Sequence,
    *,
    today: date | None = None,
    token_budget: int | None = None,
    stale_after_days: int | None = None,
) -> ContextResult:
    """Convert retrieved chunks into numbered evidence blocks."""

    chunk_list = list(chunks)

    if not chunk_list:
        return ContextResult(
            text="",
            chunks=[],
            stale_ids=[],
            truncated=False,
        )

    if today is None:
        today = date.today()

    selected_chunks: list = []
    blocks: list[str] = []
    stale_ids: list[str] = []

    for index, chunk in enumerate(chunk_list, start=1):
        citation_id = f"E{index}"

        content = str(
            getattr(chunk, "content", "") or ""
        )

        content = _neutralize_evidence_delimiters(content)

        document_type = getattr(
            chunk,
            "document_type",
            None,
        )

        published_on = getattr(
            chunk,
            "published_on",
            None,
        )

        # IMPORTANT:
        # RetrievedChunk uses source_url, not url.
        source_url = getattr(
            chunk,
            "source_url",
            None,
        )

        title = getattr(
            chunk,
            "title",
            None,
        )

        metadata: list[str] = []

        if document_type:
            metadata.append(
                f'type="{_safe_attribute(document_type)}"'
            )

        if published_on:
            metadata.append(
                f'date="{_safe_attribute(published_on)}"'
            )

        if source_url:
            metadata.append(
                f'url="{_safe_attribute(source_url)}"'
            )

        if title:
            metadata.append(
                f'title="{_safe_attribute(title)}"'
            )

        metadata_text = ""

        if metadata:
            metadata_text = " " + " ".join(metadata)

        block = (
            f'<evidence id="{citation_id}"{metadata_text}>\n'
            f"{content}\n"
            f"</evidence>"
        )

        # Check whether this evidence is stale.
        if (
            stale_after_days is not None
            and published_on is not None
        ):
            try:
                age_days = (
                    today - published_on
                ).days

                if age_days > stale_after_days:
                    stale_ids.append(citation_id)

            except (TypeError, ValueError):
                pass

        # Enforce approximate token budget.
        if token_budget is not None:
            current_text = "\n\n".join(blocks)

            candidate_text = (
                block
                if not current_text
                else current_text + "\n\n" + block
            )

            if _estimate_tokens(candidate_text) > token_budget:
                break

        selected_chunks.append(
            _attach_metadata(
                chunk,
                citation_id,
            )
        )

        blocks.append(block)

    truncated = (
        len(selected_chunks) < len(chunk_list)
    )

    # Only return stale IDs for chunks that were actually included.
    included_ids = {
        getattr(chunk, "citation_id", None)
        for chunk in selected_chunks
    }

    stale_ids = [
        citation_id
        for citation_id in stale_ids
        if citation_id in included_ids
    ]

    return ContextResult(
        text="\n\n".join(blocks),
        chunks=selected_chunks,
        stale_ids=stale_ids,
        truncated=truncated,
    )