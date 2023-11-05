from typing import Any

from .typing import PyTreePath


def normalize_path(path: PyTreePath) -> PyTreePath:
    return (path,) if isinstance(path, str) else path


def format_path(path: PyTreePath) -> str:
    path = normalize_path(path)
    return "=>".join([f"'{field}'" for field in path])


def get_field_by_path(tree: Any, path: PyTreePath) -> Any:
    path = normalize_path(path)

    for i in range(len(path)):
        field = path[i]
        if field not in tree:
            raise ValueError(f"Path {format_path(path[:i + 1])} not found in the tree {tree}.")
        tree = tree[field]

    return tree
