from __future__ import annotations

from app.agent.state import AgentState
from app.llm.client import LLMClient
from app.llm.prompts import classification
from app.llm.types import Tier
from app.schemas.llm_outputs import RouteDecision


async def classify_query(
    state: AgentState,
    llm: LLMClient,
) -> AgentState:
    """
    Classify the user's question into a bounded RouteDecision.

    The LLM only produces the typed routing decision. It does not
    execute tools or access the database.
    """

    question = state.get("question", "").strip()

    if not question:
        raise ValueError("agent state is missing a question")

    route = await llm.generate_structured(
        purpose="routing",
        tier=Tier.FAST,
        system=classification.SYSTEM,
        user=classification.render_user(question),
        schema=RouteDecision,
    )

    return {
        **state,
        "route": route,
    }