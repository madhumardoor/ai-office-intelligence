from __future__ import annotations

from app.agent.state import AgentState
from app.schemas.llm_outputs import RagAnswer


def _format_claim(text: str, evidence_ids: list[str]) -> str:
    citations = " ".join(f"[{evidence_id}]" for evidence_id in evidence_ids)
    return f"- {text} {citations}".strip()


def build_final_response(
    state: AgentState,
) -> AgentState:
    """
    Convert the verified RagAnswer into the user-facing response.

    No LLM call is made here. This prevents post-verification
    hallucinations or unsupported rewrites.
    """

    answer = state.get("answer")

    if answer is None:
        raise ValueError("agent state is missing answer")

    if not isinstance(answer, RagAnswer):
        raise TypeError("agent state answer must be RagAnswer")

    verified = state.get("verified", False)
    verification_errors = state.get("verification_errors", [])

    # Never present an unverified generated answer as established fact.
    if not verified:
        lines = [
            "I could not fully verify the generated answer against "
            "the retrieved evidence."
        ]

        if verification_errors:
            lines.append("")
            lines.append("Verification issues:")
            for error in verification_errors[:6]:
                lines.append(f"- {error}")

        lines.append("")
        lines.append(
            "Please treat the retrieved evidence as unconfirmed until "
            "the cited information is validated."
        )

        return {
            **state,
            "final_text": "\n".join(lines),
        }

    lines: list[str] = []

    if answer.summary.strip():
        lines.append(answer.summary.strip())

    if answer.claims:
        lines.append("")
        lines.append("Evidence:")
        for claim in answer.claims:
            lines.append(
                _format_claim(
                    claim.text,
                    claim.evidence_ids,
                )
            )

    if answer.caveats:
        lines.append("")
        lines.append("Caveats:")
        for caveat in answer.caveats:
            lines.append(f"- {caveat}")

    if not lines:
        lines.append(
            "The available evidence is insufficient to answer the question."
        )

    return {
        **state,
        "final_text": "\n".join(lines),
    }