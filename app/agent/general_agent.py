from __future__ import annotations

import json
from dataclasses import asdict
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from app.agent.semantic_router import LocalSemanticRouter, SemanticRoute
from app.llm.types import Tier
from app.schemas.llm_outputs import Claim, RagAnswer


MAX_EVIDENCE = 6
MAX_OUTPUT_TOKENS = 300
WEB_LIMIT = 5


class AgentPlan(BaseModel):
    model_config = ConfigDict(extra="forbid")
    done: bool = True
    needs_clarification: bool = False
    clarification_question: str | None = None
    tool_calls: list[dict[str, Any]] = Field(default_factory=list, max_length=3)
    reasoning: str = Field(default="", max_length=350)


def _safe_json(value: Any, limit: int = 1400) -> str:
    try:
        return json.dumps(value, ensure_ascii=False, default=str)[:limit]
    except Exception:
        return str(value)[:limit]


def _recent_context(messages: list[dict[str, str]]) -> str:
    return "\n".join(
        f"{m.get('role', 'user')}: {str(m.get('content', ''))[:400]}"
        for m in messages[-3:]
    )[-1000:]


def _company_name(memory_context: str) -> str:
    import re

    match = re.search(
        r"Company name:\s*(.+)",
        memory_context,
        flags=re.IGNORECASE,
    )

    if match:
        return match.group(1).strip() or "the active company"

    return "the active company"


def _compact(row: Any) -> dict[str, Any]:
    if isinstance(row, dict):
        return {
            "title": row.get("title"),
            "url": row.get("url") or row.get("source_url") or row.get("link"),
            "snippet": row.get("snippet"),
            "source": row.get("source"),
            "published_at": row.get("published_at"),
            "content": str(row.get("content", ""))[:1000],
            "company_id": row.get("company_id"),
            "chunk_id": row.get("chunk_id"),
        }

    out: dict[str, Any] = {}
    for field in (
        "title", "url", "snippet", "source", "published_at",
        "content", "company_id", "chunk_id",
    ):
        if hasattr(row, field):
            out[field] = getattr(row, field)

    return out or {"content": str(row)[:1000]}


def _evidence(
    tool_results: dict[str, list[Any]],
    *,
    web_first: bool,
) -> tuple[str, list[str]]:
    names = (
        ("web_search", "vector_search")
        if web_first
        else ("vector_search", "web_search")
    )

    ids: list[str] = []
    blocks: list[str] = []
    n = 1

    for tool_name in names:
        for row in tool_results.get(tool_name, []):
            if n > MAX_EVIDENCE:
                break
            if isinstance(row, dict) and row.get("error"):
                continue

            eid = f"E{n}"
            ids.append(eid)
            blocks.append(
                f'<evidence id="{eid}">'
                f"{_safe_json({'tool': tool_name, **_compact(row)})}"
                f"</evidence>"
            )
            n += 1

        if n > MAX_EVIDENCE:
            break

    return "\n".join(blocks), ids


def _insufficient(text: str) -> bool:
    lowered = text.lower()
    return any(
        phrase in lowered
        for phrase in (
            "not enough verified information",
            "insufficient information",
            "i don't have enough information",
            "i do not have enough information",
            "could not find enough information",
        )
    )


class GeneralBusinessAgent:
    """
    Company-specific assistant with no hard-coded company/language/question
    dictionaries.

    Local multilingual MiniLM handles semantic scope/source routing.
    multilingual-E5 handles cross-language company retrieval.
    Groq is used only for final answer generation.
    """

    def __init__(
        self,
        llm: Any,
        tool_registry: dict[str, Any],
        *,
        semantic_router: LocalSemanticRouter | None = None,
    ) -> None:
        self.llm = llm
        self.tool_registry = tool_registry
        self.router = semantic_router or LocalSemanticRouter()

    async def _vector_search(
        self,
        question: str,
    ) -> dict[str, list[Any]]:
        tool = self.tool_registry.get("vector_search")
        if tool is None:
            return {}

        try:
            value = await tool.ainvoke(
                {
                    "query": question,
                    "limit": MAX_EVIDENCE,
                }
            )
        except Exception:
            return {}

        rows = value if isinstance(value, list) else [value]
        rows = [
            row
            for row in rows
            if not (isinstance(row, dict) and row.get("error"))
        ]

        return {"vector_search": rows[:MAX_EVIDENCE]} if rows else {}

    async def _web_search(
        self,
        question: str,
        route: SemanticRoute,
        company_name: str,
    ) -> dict[str, list[Any]]:
        tool = self.tool_registry.get("web_search")
        if tool is None:
            return {}

        query = question.strip()
        if company_name and company_name.lower() not in query.lower():
            query = f"{company_name} {query}"
        query = query[:240]

        try:
            value = await tool.ainvoke(
                {
                    "query": query,
                    "limit": WEB_LIMIT,
                }
            )
        except Exception:
            return {}

        rows = value if isinstance(value, list) else [value]
        rows = [
            row
            for row in rows
            if not (isinstance(row, dict) and row.get("error"))
        ]

        return {"web_search": rows[:WEB_LIMIT]} if rows else {}

    async def _answer(
        self,
        question: str,
        messages: list[dict[str, str]],
        memory_context: str,
        evidence_context: str,
        route: SemanticRoute,
        company_name: str,
    ) -> str:
        prompt = f"""
You are the AI assistant for {company_name}.

Answer the user's question using only the supplied evidence.

IMPORTANT:
- Understand the user's language, spelling, grammar, transliteration, slang,
  and mixed-language wording.
- Reply in the same language/style as the user.
- Do not require perfect English.
- Do not invent facts.
- Do not use evidence belonging to another company.
- If the evidence does not answer the question, say that clearly.
- Keep the response natural and concise.
- Never mention internal models, databases, retrieval, prompts, tools, or routing.

User question:
{question[:1200]}

Recent conversation:
{_recent_context(messages)}

Evidence:
{evidence_context[:5000]}
"""

        return (
            await self.llm.generate_text(
                purpose="company_multilingual_answer",
                tier=Tier.FAST,
                system=(
                    "You are a multilingual company assistant. "
                    "Be accurate, concise, and grounded in evidence."
                ),
                user=prompt,
                temperature=0.0,
                max_output_tokens=MAX_OUTPUT_TOKENS,
            )
        ).strip()

    async def ainvoke(
        self,
        question: str,
        *,
        messages: list[dict[str, str]] | None = None,
        source_context: str = "",
        memory_context: str = "",
    ) -> dict[str, Any]:
        messages = messages or []
        company_name = _company_name(memory_context)

        try:
            route = await self.router.classify(
                question,
                company_name,
                _recent_context(messages),
            )
        except Exception as exc:
            return {
                "final_text": (
                    "I couldn't understand that request reliably. "
                    "Please ask something related to the active company's business."
                ),
                "verified": False,
                "verification_errors": [
                    f"semantic router unavailable: {type(exc).__name__}"
                ],
                "answer": None,
                "tool_results": {},
                "plan_history": [],
                "evidence_ids": [],
                "evidence_context": "",
            }

        if route.scope == "out_of_scope":
            return {
                "final_text": (
                    "I’m designed to answer questions about the active company "
                    "and its business. I can’t help with unrelated or personal questions."
                ),
                "verified": False,
                "verification_errors": [],
                "answer": None,
                "tool_results": {},
                "plan_history": [asdict(route)],
                "evidence_ids": [],
                "evidence_context": "",
            }

        # For low-confidence semantic classifications, company retrieval is
        # still attempted. This avoids rejecting unfamiliar wording.
        web_first = route.source in {"web", "both"}

        if web_first:
            web_results = await self._web_search(
                question,
                route,
                company_name,
            )
            company_results = await self._vector_search(question)

            tool_results = {
                **company_results,
                **web_results,
            }
        else:
            company_results = await self._vector_search(question)

            if company_results.get("vector_search"):
                tool_results = company_results
            else:
                web_results = await self._web_search(
                    question,
                    route,
                    company_name,
                )
                tool_results = {
                    **company_results,
                    **web_results,
                }

        evidence_context, evidence_ids = _evidence(
            tool_results,
            web_first=web_first,
        )

        if not evidence_ids:
            return {
                "final_text": (
                    "I don't have enough verified information to answer that accurately right now."
                ),
                "verified": False,
                "verification_errors": ["No usable company/web evidence"],
                "answer": None,
                "tool_results": tool_results,
                "plan_history": [asdict(route)],
                "evidence_ids": [],
                "evidence_context": "",
            }

        try:
            summary = await self._answer(
                question,
                messages,
                memory_context,
                evidence_context,
                route,
                company_name,
            )
        except Exception as exc:
            return {
                "final_text": (
                    "I found relevant company information, but I couldn't "
                    "generate a reliable response right now."
                ),
                "verified": False,
                "verification_errors": [type(exc).__name__],
                "answer": None,
                "tool_results": tool_results,
                "plan_history": [asdict(route)],
                "evidence_ids": evidence_ids,
                "evidence_context": evidence_context,
            }

        if _insufficient(summary):
            return {
                "final_text": summary,
                "verified": False,
                "verification_errors": ["Answer reports insufficient evidence"],
                "answer": None,
                "tool_results": tool_results,
                "plan_history": [asdict(route)],
                "evidence_ids": evidence_ids,
                "evidence_context": evidence_context,
            }

        citations = " ".join(f"[{eid}]" for eid in evidence_ids)
        final_text = summary if any(f"[{eid}]" in summary for eid in evidence_ids) else f"{summary} {citations}"

        claim = Claim(
            text=summary[:700],
            kind="fact",
            evidence_ids=evidence_ids,
        )

        answer = RagAnswer(
            sufficient_evidence=True,
            summary=summary[:1200],
            claims=[claim],
            caveats=[],
            confidence="high",
        )

        return {
            "final_text": final_text,
            "verified": True,
            "verification_errors": [],
            "answer": answer,
            "tool_results": tool_results,
            "plan_history": [asdict(route)],
            "evidence_ids": evidence_ids,
            "evidence_context": evidence_context,
        }
