import re
from typing import Any, Literal

from fess38.utils.pytree import PyTreePath, format_path, get_field_by_path

from . import filter_library
from .base import FilterConfigBase


class DropIfRegexpMatchFilterConfig(FilterConfigBase):
    type: Literal["drop_if_regexp_match"] = "drop_if_regexp_match"
    regexp: str
    path: PyTreePath


@filter_library("drop_if_regexp_match")
def drop_if_regexp_match(record: Any, regexp: str, path: PyTreePath) -> bool:
    field = get_field_by_path(path, record)
    if not isinstance(field, str):
        raise ValueError(f"Field {format_path(path)} is not a string.")
    return re.match(regexp, field) is None
