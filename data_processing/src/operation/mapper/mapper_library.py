from typing import Any

from fess38.util.pytree import set_field_by_path
from fess38.util.registry import Registry
from fess38.util.typing import PyTreePath

mapper_library = Registry("mapper_library")


@mapper_library("add_constant_field")
def add_constant_field(record: dict, path: PyTreePath, value: Any) -> dict:
    set_field_by_path(record, path, value)
    return record
