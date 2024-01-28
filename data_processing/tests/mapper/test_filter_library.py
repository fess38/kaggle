import pytest
from fess38.data_processing.operation.mapper.filter_library import evaluate_expression
from fess38.util.typing import PyTree


@pytest.mark.parametrize(
    ("record", "expression", "expected"),
    [
        ({"a": 1}, 'record["a"] > 0', True),
        ({"a": 1}, '"b" in record', False),
    ],
)
def test_execute_expression(record: PyTree, expression: str, expected: bool):
    assert evaluate_expression(record, expression) == expected
