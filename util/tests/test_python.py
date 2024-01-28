from typing import Any

import pytest
from fess38.util.python import evaluate_expression, execute_expression
from fess38.util.typing import PyTree


@pytest.mark.parametrize(
    ("expression", "record", "expected"),
    [
        ('record["b"] = 2', {"a": 1}, {"a": 1, "b": 2}),
        ('del record["a"]', {"a": 1}, {}),
        ('record.clear(); record["b"] = 2', {"a": 1}, {"b": 2}),
    ],
)
def test_execute_expression(expression: str, record: PyTree, expected: Any):
    execute_expression(expression, locals={"record": record})
    assert record == expected


@pytest.mark.parametrize(
    ("expression", "record", "expected"),
    [
        ('record["a"] > 0', {"a": 1}, True),
        ('"b" in record', {"a": 1}, False),
    ],
)
def test_evaluate_expression(expression: str, record: PyTree, expected: Any):
    assert evaluate_expression(expression, locals={"record": record}) == expected
