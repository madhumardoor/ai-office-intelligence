from __future__ import annotations

import re
from typing import Any

from app.agent.state import AgentState


MAX_TOOL_RESULTS = 50


class ToolExecutionError(RuntimeError):
    """Raised when a selected tool cannot be safely invoked."""


def _first_company(state: AgentState) -> str | None:
    route = state["route"]

    if route.company_names:
        return route.company_names[0]

    return None


def _extract_arithmetic_expression(
    question: str,
) -> str | None:
    """
    Extract a simple arithmetic expression from natural-language text.

    Example:
        "What is 125 + 75?" -> "125 + 75"
    """

    matches = re.findall(
        r"(?<!\w)(?:\d+(?:\.\d+)?(?:\s*[+\-*/%]\s*\d+(?:\.\d+)?)+)(?!\w)",
        question,
    )

    if not matches:
        return None

    return matches[0].strip()


def _extract_coordinates(
    question: str,
) -> tuple[float, float] | None:
    """
    Extract latitude/longitude when explicitly present in the question.

    Example:
        "Find offices near 12.9716, 77.5946"
    """

    match = re.search(
        r"(?<!\d)(-?\d{1,2}(?:\.\d+)?)\s*,\s*"
        r"(-?\d{1,3}(?:\.\d+)?)(?!\d)",
        question,
    )

    if not match:
        return None

    latitude = float(match.group(1))
    longitude = float(match.group(2))

    if not -90 <= latitude <= 90:
        return None

    if not -180 <= longitude <= 180:
        return None

    return latitude, longitude


def _build_tool_kwargs(
    tool_name: str,
    state: AgentState,
) -> dict[str, Any] | None:
    """
    Convert the validated RouteDecision into the strict arguments
    expected by one Phase 6 StructuredTool.

    Returning None means the selected tool does not have enough
    information in the current state to execute safely.
    """

    route = state["route"]
    question = state["question"]
    company = _first_company(state)

    # ---------------------------------------------------------------
    # Company search
    # ---------------------------------------------------------------
    #
    # Important:
    # If the user asks:
    #   "Which companies are listed in Bengaluru?"
    #
    # company is None, so query MUST remain None.
    # The city filter then returns all matching Bengaluru companies.
    #
    # If the user asks:
    #   "Is Nova Labs listed in Bengaluru?"
    #
    # company becomes "Nova Labs", so the tool searches for that
    # specific company and applies the Bengaluru filter.
    #
    if tool_name == "company_search":
        return {
            "query": company,
            "city": route.city,
            "limit": 20,
        }

    # ---------------------------------------------------------------
    # Tenant search
    # ---------------------------------------------------------------
    #
    # Do not send the entire natural-language question as a text
    # search. Only search by a company name when one was explicitly
    # extracted by the router.
    #
    if tool_name == "tenant_search":
        return {
            "query": company,
            "limit": 20,
        }

    # ---------------------------------------------------------------
    # Coworking search
    # ---------------------------------------------------------------

    if tool_name == "coworking_search":
        return {
            "operator_name": company,
            "city": route.city,
            "area": route.area,
            "limit": 20,
        }

    # ---------------------------------------------------------------
    # Property search
    # ---------------------------------------------------------------

    if tool_name == "property_search":
        return {
            "query": company,
            "city": route.city,
            "area": route.area,
            "limit": 20,
        }

    # ---------------------------------------------------------------
    # PostGIS search
    # ---------------------------------------------------------------

    if tool_name == "postgis_search":
        coordinates = _extract_coordinates(question)

        if coordinates is None:
            return None

        if route.radius_km is None:
            return None

        latitude, longitude = coordinates

        return {
            "latitude": latitude,
            "longitude": longitude,
            "radius_km": route.radius_km,
            "city": route.city,
            "limit": 20,
        }

    # ---------------------------------------------------------------
    # Vector search
    # ---------------------------------------------------------------

    if tool_name == "vector_search":
        return {
            "query": question,
            "limit": 8,
        }

    # ---------------------------------------------------------------
    # Signal lookup
    # ---------------------------------------------------------------

    if tool_name == "signal_lookup":
        return {
            "city": route.city,
            "limit": 20,
        }

    # ---------------------------------------------------------------
    # Calculator
    # ---------------------------------------------------------------

    if tool_name == "calculator":
        expression = _extract_arithmetic_expression(
            question
        )

        if expression is None:
            return None

        return {
            "expression": expression,
        }

    # ---------------------------------------------------------------
    # Web search
    # ---------------------------------------------------------------

    if tool_name == "web_search":
        return {
            "query": question,
            "limit": 10,
        }

    # ---------------------------------------------------------------
    # News search
    # ---------------------------------------------------------------

    if tool_name == "news_search":
        return {
            "query": question,
            "company": company,
            "city": route.city,
            "limit": 10,
        }

    raise ToolExecutionError(
        f"unsupported tool requested by route: {tool_name}"
    )


def _normalize_result(
    value: Any,
) -> list[dict[str, Any]]:
    """
    Normalize StructuredTool output into the AgentState shape.
    """

    if value is None:
        return []

    if isinstance(value, list):
        result: list[dict[str, Any]] = []

        for item in value[:MAX_TOOL_RESULTS]:
            if isinstance(item, dict):
                result.append(item)
            else:
                result.append(
                    {
                        "value": item
                    }
                )

        return result

    if isinstance(value, dict):
        return [value]

    return [
        {
            "value": value
        }
    ]


async def execute_tools(
    state: AgentState,
    tool_registry: dict[str, Any],
) -> AgentState:
    """
    Execute only tools explicitly approved by RouteDecision.

    Security properties:
    - no arbitrary tool names
    - registry lookup is mandatory
    - tool arguments are constructed locally
    - StructuredTool performs final Pydantic validation
    - tools without sufficient information are skipped safely
    """

    route = state.get("route")

    if route is None:
        raise ValueError(
            "agent state is missing route"
        )

    if route.needs_clarification:
        return {
            **state,
            "tool_results": {},
        }

    results: dict[str, list[dict[str, Any]]] = {}

    for tool_name in route.tools_required:

        tool = tool_registry.get(
            tool_name
        )

        if tool is None:
            raise ToolExecutionError(
                f"tool is not available in registry: {tool_name}"
            )

        kwargs = _build_tool_kwargs(
            tool_name,
            state,
        )

        if kwargs is None:
            results[tool_name] = [
                {
                    "status": "skipped",
                    "reason": (
                        "insufficient information "
                        "for safe execution"
                    ),
                }
            ]
            continue

        value = await tool.ainvoke(
            kwargs
        )

        results[tool_name] = _normalize_result(
            value
        )

    return {
        **state,
        "tool_results": results,
    }