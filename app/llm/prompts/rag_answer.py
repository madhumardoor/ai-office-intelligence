"""Grounded answer generation prompt."""

from __future__ import annotations

from app.llm.prompts.base import GROUNDING_RULES, fence


NAME = "rag_answer"
VERSION = "1.0"


SYSTEM = f"""\
You answer questions about companies and their potential need for office space,
using only supplied evidence.

{GROUNDING_RULES}

OUTPUT REQUIREMENTS:
- Every claim must list the evidence ids that support it (evidence_ids).
- A statement you cannot support with at least one evidence id must not appear.
- kind="fact": stated directly in evidence.
- kind="signal": an observation that may indicate expansion.
- kind="inference": your interpretation. Phrase inferences with hedging such as
  "may" or "could".
- Never claim a company definitely needs office space. It is an evidence-based
  potential only.
- If evidence does not answer the question, set sufficient_evidence=false,
  leave claims empty, and explain what is missing in summary.
- confidence="high" only with multiple independent, recent, consistent sources.
- Use "low" when evidence is thin or stale.
- Add caveats for stale evidence, single-source claims, or conflicts.
"""


def render_user(
    question: str,
    context: str,
    structured_facts: str = "",
) -> str:
    """Render question plus trusted structured facts and retrieved evidence."""

    facts = (
        "\nSTRUCTURED DATABASE FACTS "
        "(trusted, from internal tables):\n"
        f"{structured_facts}\n"
        if structured_facts.strip()
        else ""
    )

    return (
        f"QUESTION:\n{fence('question', question.strip()[:1500])}"
        f"{facts}\n"
        f"EVIDENCE:\n{context or '(no evidence retrieved)'}"
    )