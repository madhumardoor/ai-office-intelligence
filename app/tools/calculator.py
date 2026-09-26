from __future__ import annotations

import ast
import operator
from typing import Union

from app.tools.schemas.inputs import CalculatorInput


Number = Union[int, float]

_BIN_OPS: dict[type[ast.operator], object] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Mod: operator.mod,
}

_UNARY_OPS: dict[type[ast.unaryop], object] = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}


class CalculatorError(ValueError):
    """Raised when an arithmetic expression is unsafe or invalid."""


def _evaluate(node: ast.AST) -> Number:
    if isinstance(node, ast.Expression):
        return _evaluate(node.body)

    if isinstance(node, ast.Constant):
        if isinstance(node.value, bool):
            raise CalculatorError("Boolean values are not allowed.")

        if isinstance(node.value, (int, float)):
            return node.value

        raise CalculatorError("Only numeric constants are allowed.")

    if isinstance(node, ast.BinOp):
        operation = _BIN_OPS.get(type(node.op))
        if operation is None:
            raise CalculatorError(
                f"Operator {type(node.op).__name__} is not allowed."
            )

        left = _evaluate(node.left)
        right = _evaluate(node.right)

        try:
            result = operation(left, right)  # type: ignore[operator]
        except ZeroDivisionError as exc:
            raise CalculatorError("Division by zero is not allowed.") from exc
        except (OverflowError, ValueError) as exc:
            raise CalculatorError("Invalid arithmetic expression.") from exc

        if isinstance(result, float) and (
            result != result or result in (float("inf"), float("-inf"))
        ):
            raise CalculatorError("Result is not a finite number.")

        return result

    if isinstance(node, ast.UnaryOp):
        operation = _UNARY_OPS.get(type(node.op))
        if operation is None:
            raise CalculatorError(
                f"Unary operator {type(node.op).__name__} is not allowed."
            )

        operand = _evaluate(node.operand)

        try:
            result = operation(operand)  # type: ignore[operator]
        except (OverflowError, ValueError) as exc:
            raise CalculatorError("Invalid arithmetic expression.") from exc

        if isinstance(result, float) and (
            result != result or result in (float("inf"), float("-inf"))
        ):
            raise CalculatorError("Result is not a finite number.")

        return result

    raise CalculatorError(
        f"Expression node {type(node).__name__} is not allowed."
    )


def calculate(request: CalculatorInput) -> Number:
    expression = request.expression.strip()

    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError as exc:
        raise CalculatorError("Invalid arithmetic expression.") from exc

    return _evaluate(tree)