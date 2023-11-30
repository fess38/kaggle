from urllib.parse import ParseResult, urlparse

import fsspec

from .typing import PathLike


def fs_for_path(path: PathLike) -> fsspec.AbstractFileSystem:
    path = str(path)
    parsed_location: ParseResult = urlparse(path)
    fs_class = parsed_location.scheme if parsed_location.scheme is not None else "file"
    return fsspec.filesystem(fs_class)


def open_file(path: PathLike, mode: str) -> fsspec.core.OpenFile:
    return fs_for_path(path).open(path, mode)
