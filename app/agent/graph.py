from __future__ import annotations

from typing import Any

from langgraph.graph import END, START, StateGraph

from app.agent.evidence import build_evidence_context
from app.agent.executor import execute_tools
from app.agent.finalizer import build_final_response
from app.agent.reasoner import reason_over_evidence
from app.agent.router import classify_query
from app.agent.state import AgentState
from app.agent.verifier import verify_answer
from app.llm.client import LLMClient


def build_agent_graph(
    llm: LLMClient,
    tool_registry: dict[str, Any],
):
    """
    Phase 7 agent graph.

    START
      ↓
    router
      ↓
    executor
      ↓
    evidence
      ↓
    reasoner
      ↓
    verifier
      ↓
    finalizer
      ↓
    END
    """

    async def router_node(state: AgentState) -> AgentState:
        return await classify_query(state, llm)

    async def executor_node(state: AgentState) -> AgentState:
        return await execute_tools(state, tool_registry)

    def evidence_node(state: AgentState) -> AgentState:
        return build_evidence_context(state)

    async def reasoner_node(state: AgentState) -> AgentState:
        return await reason_over_evidence(state, llm)

    def verifier_node(state: AgentState) -> AgentState:
        return verify_answer(state)

    def finalizer_node(state: AgentState) -> AgentState:
        return build_final_response(state)

    graph = StateGraph(AgentState)

    graph.add_node("router", router_node)
    graph.add_node("executor", executor_node)
    graph.add_node("evidence", evidence_node)
    graph.add_node("reasoner", reasoner_node)
    graph.add_node("verifier", verifier_node)
    graph.add_node("finalizer", finalizer_node)

    graph.add_edge(START, "router")
    graph.add_edge("router", "executor")
    graph.add_edge("executor", "evidence")
    graph.add_edge("evidence", "reasoner")
    graph.add_edge("reasoner", "verifier")
    graph.add_edge("verifier", "finalizer")
    graph.add_edge("finalizer", END)

    return graph.compile()