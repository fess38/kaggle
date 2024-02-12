from fess38.util.python import evaluate_expression as evaluate_expression_impl
from fess38.util.registry import Registry

filter_library = Registry("filter_library")


@filter_library("evaluate_expression")
def evaluate_expression(record: dict, expression: str) -> dict:
    return evaluate_expression_impl(expression, locals={"record": record})
