from typing import Any, Literal

from fess38.utils.pytree import PyTreePath, format_path, get_field_by_path

from . import filter_library
from .base import FilterConfigBase


class DropEmptyFilterConfig(FilterConfigBase):
    type: Literal["drop_empty"] = "drop_empty"
    path: PyTreePath


@filter_library("drop_empty")
def drop_empty(record: Any, path: PyTreePath) -> bool:
    field = get_field_by_path(path, record)
    if not isinstance(field, str):
        raise ValueError(f"Field {format_path(path)} is not a string.")
    return field != ""
