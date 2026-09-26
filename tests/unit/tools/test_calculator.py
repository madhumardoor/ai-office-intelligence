from __future__ import annotations

import ast
import pytest
from pydantic import ValidationError

from app.tools.calculator import CalculatorError, _evaluate, calculate
from app.tools.schemas.inputs import CalculatorInput


def test_calculator_handles_basic_arithmetic():
    request = CalculatorInput(expression="10 * (5 + 2)")

    assert calculate(request) == 70


def test_calculator_handles_division_and_modulo():
    assert calculate(CalculatorInput(expression="20 / 4")) == 5
    assert calculate(CalculatorInput(expression="17 % 5")) == 2


def test_calculator_handles_unary_operators():
    assert calculate(CalculatorInput(expression="-10 + 3")) == -7
    assert calculate(CalculatorInput(expression="+10")) == 10


def test_calculator_rejects_division_by_zero():
    request = CalculatorInput(expression="1 / 0")

    with pytest.raises(CalculatorError):
        calculate(request)


def test_calculator_rejects_power_operator():
    request = CalculatorInput(expression="2 ** 10")

    with pytest.raises(CalculatorError):
        calculate(request)


@pytest.mark.parametrize(
    "expression",
    [
        "abs(5)",
        "__import__('os')",
        "open('file.txt')",
        "().__class__",
        "[1, 2, 3]",
        '{"a": 1}',
    ],
)
def test_calculator_input_rejects_non_arithmetic_expressions(expression: str):
    with pytest.raises(ValidationError):
        CalculatorInput(expression=expression)


def test_calculator_rejects_unsupported_ast_nodes():
    tree = ast.parse("1", mode="eval")

    class UnsupportedNode(ast.AST):
        pass

    with pytest.raises(CalculatorError):
        _evaluate(UnsupportedNode())


def test_calculator_does_not_use_python_eval(monkeypatch):
    def fail_eval(*args, **kwargs):
        raise AssertionError("eval() must not be used")

    monkeypatch.setattr("builtins.eval", fail_eval)

    request = CalculatorInput(expression="10 * (5 + 2)")

    assert calculate(request) == 70