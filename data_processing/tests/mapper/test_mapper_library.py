from typing import Any

import pytest
from fess38.data_processing.operation.mapper.mapper_library import execute_expressions
from fess38.util.typing import PyTree


@pytest.mark.parametrize(
    ("record", "expressions", "expected"),
    [
        ({"a": 1}, ['record["b"] = 2', 'record["c"] = 3'], {"a": 1, "b": 2, "c": 3}),
        ({"a": 1}, ['del record["a"]', 'record["b"] = 0'], {"b": 0}),
    ],
)
def test_execute_expression(record: PyTree, expressions: list[str], expected: Any):
    assert execute_expressions(record, expressions) == expected
