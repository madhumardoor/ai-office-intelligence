from __future__ import annotations

import json
import re
from typing import Any

from app.agent.state import AgentState
from app.schemas.llm_outputs import Claim, RagAnswer


_NUMBER_RE = re.compile(
    r"(?<![\w.])-?\d+(?:\.\d+)?%?(?![\w.])"
)

_QUOTED_RE = re.compile(
    r'"([^"]+)"|“([^”]+)”|\'([^\']+)\''
)


def _parse_evidence(context: str) -> dict[str, str]:
    """
    Convert the deterministic evidence packet into:

        E1 -> serialized source text
        E2 -> serialized source text
        ...

    Invalid evidence lines are ignored here and reported by the
    caller through missing evidence references.
    """

    evidence: dict[str, str] = {}

    for line in context.splitlines():
        line = line.strip()

        if not line:
            continue

        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue

        evidence_id = payload.get("evidence_id")

        if not isinstance(evidence_id, str):
            continue

        evidence[evidence_id] = json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
        )

    return evidence


def _extract_numbers(text: str) -> list[str]:
    return _NUMBER_RE.findall(text)


def _extract_quotes(text: str) -> list[str]:
    values: list[str] = []

    for match in _QUOTED_RE.finditer(text):
        for group in match.groups():
            if group:
                values.append(group)
                break

    return values


def _verify_claim(
    claim: Claim,
    evidence: dict[str, str],
) -> list[str]:
    errors: list[str] = []

    for evidence_id in claim.evidence_ids:
        source = evidence.get(evidence_id)

        if source is None:
            errors.append(
                f"claim cites missing evidence ID: {evidence_id}"
            )
            continue

        # Explicit quoted text must occur verbatim in the cited source.
        for quote in _extract_quotes(claim.text):
            if quote not in source:
                errors.append(
                    f"quoted text from {evidence_id!r} is not present "
                    f"in cited evidence: {quote!r}"
                )

        # Every number asserted by the claim must occur in its cited source.
        for number in _extract_numbers(claim.text):
            if number not in source:
                errors.append(
                    f"number {number!r} in claim is not present "
                    f"in cited evidence {evidence_id}"
                )

    return errors


def verify_answer(
    state: AgentState,
) -> AgentState:
    """
    Deterministically verify the structured RagAnswer against evidence.

    Checks:
    1. Every cited evidence ID exists.
    2. Explicit quoted text occurs in the cited evidence.
    3. Numbers asserted by claims occur in the cited evidence.

    No LLM is used here.
    """

    answer = state.get("answer")

    if answer is None:
        raise ValueError("agent state is missing answer")

    if not isinstance(answer, RagAnswer):
        raise TypeError("agent state answer must be RagAnswer")

    evidence_context = state.get("evidence_context", "")
    evidence_ids = set(state.get("evidence_ids", []))

    errors: list[str] = []

    parsed_evidence = _parse_evidence(evidence_context)

    # Cross-check the application-generated ID list too.
    for evidence_id in parsed_evidence:
        if evidence_id not in evidence_ids:
            errors.append(
                f"evidence context contains undeclared ID: {evidence_id}"
            )

    for claim in answer.claims:
        errors.extend(
            _verify_claim(
                claim,
                parsed_evidence,
            )
        )

    # A claim cannot cite evidence if there is no evidence packet.
    if answer.claims and not parsed_evidence:
        errors.append(
            "answer contains claims but no evidence is available"
        )

    verified = not errors

    return {
        **state,
        "verification_errors": errors,
        "verified": verified,
    }