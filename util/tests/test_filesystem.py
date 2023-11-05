from pathlib import Path

from fess38.util.filesystem import fs_for_path
from fsspec.implementations.local import LocalFileSystem


def test_fs_for_path_local(tmp_path: Path):
    assert type(fs_for_path(tmp_path)) == LocalFileSystem
