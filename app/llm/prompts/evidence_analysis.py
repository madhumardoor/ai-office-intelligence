"""FACT -> SIGNAL -> INFERENCE evidence-analysis prompt."""

from __future__ import annotations

from app.llm.prompts.base import GROUNDING_RULES, fence


NAME = "evidence_analysis"
VERSION = "1.0"


SYSTEM = f"""\
You explain why a company may or may not be a potential office-space
expansion lead as an explicit chain:

FACTS
what the evidence directly states

-> SIGNALS
what those facts may indicate

-> INFERENCE
a hedged conclusion

{GROUNDING_RULES}

Rules:
- facts must contain only direct statements from evidence
- signals are observations that follow from facts
- inference is at most one conclusion
- inference must be hedged
- inference must cite the evidence it rests on
- set inference_is_confirmed_fact=false always
- include conflicting_evidence when sources disagree
- include limitations for stale data, single-source evidence,
  missing data, or other material limitations
- if evidence is too thin to support an inference, leave inference null
  and explain why in limitations
- the score breakdown, when supplied, is computed by deterministic code;
  explain it but never change or recompute it
"""


def render_user(
    company: str,
    context: str,
    score_breakdown: str = "",
) -> str:
    """Render a company and its evidence for analysis."""

    score = (
        f"\nSCORE BREAKDOWN (deterministic, trusted):\n"
        f"{score_breakdown}\n"
        if score_breakdown.strip()
        else ""
    )

    return (
        f"COMPANY: {fence('company', company.strip()[:300])}"
        f"{score}\n"
        f"EVIDENCE:\n{context or '(no evidence)'}"
    )