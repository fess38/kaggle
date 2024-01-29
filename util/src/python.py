import threading
from typing import Any, Callable

_THREAD_LOCAL = threading.local()
_THREAD_LOCAL.compile_cache = {}


def run_expression(
    expression: str,
    globals: dict[str, Any] | None,
    locals: dict[str, Any] | None,
    run_fn: Callable,
):
    if expression not in _THREAD_LOCAL.compile_cache:
        compiled_expression = compile(expression, "<string>", run_fn.__name__)
        _THREAD_LOCAL.compile_cache[expression] = compiled_expression

    compiled_expression = _THREAD_LOCAL.compile_cache.get(expression)
    return run_fn(compiled_expression, globals, locals)


def evaluate_expression(
    expression: str,
    globals: dict[str, Any] | None = None,
    locals: dict[str, Any] | None = None,
):
    return run_expression(expression, globals, locals, eval)


def execute_expression(
    expression: str,
    globals: dict[str, Any] | None = None,
    locals: dict[str, Any] | None = None,
):
    return run_expression(expression, globals, locals, exec)
