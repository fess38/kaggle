from typing import Any, Literal
from urllib.parse import urlparse

from fess38.utils.pytree import PyTreePath, format_path, get_field_by_path

from . import filter_library
from .base import FilterConfigBase


class DropDomainsFilterConfig(FilterConfigBase):
    type: Literal["drop_domains"] = "drop_domains"
    domains: list[str]
    url_path: PyTreePath
    include_subdomains: bool = True


@filter_library("drop_domains")
def drop_domains(
    record: Any,
    domains: list[str],
    url_path: PyTreePath,
    include_subdomains: bool,
) -> bool:
    url = get_field_by_path(url_path, record)
    if not isinstance(url, str):
        raise ValueError(f"Field {format_path(url_path)} is not a string.")
    parsed_url = urlparse(url)
    domain = parsed_url.netloc
    if include_subdomains and any(domain.endswith(f".{d}") for d in domains):
        return False
    return domain not in domains
