from fess38.util.python import evaluate_expression as evaluate_expression_impl
from fess38.util.pytree import PyTreePath, format_path, get_field_by_path
from fess38.util.registry import Registry

filter_library = Registry("filter_library")


@filter_library("evaluate_expression")
def evaluate_expression(record: dict, expression: str) -> dict:
    return evaluate_expression_impl(expression, locals={"record": record})


@filter_library("drop_empty")
def drop_empty(record: dict, path: PyTreePath) -> bool:
    value = get_field_by_path(path, record)
    if not isinstance(value, str):
        raise ValueError(f"Field {format_path(path)} is not a string")

    return value is not None and value != ""
