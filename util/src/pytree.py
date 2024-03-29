from typing import Any

from .typing import PyTree, PyTreePath


def normalize_path(path: PyTreePath) -> PyTreePath:
    return tuple(path.split(".")) if isinstance(path, str) else path


def format_path(path: PyTreePath) -> str:
    path = normalize_path(path)
    return "=>".join([f"'{field}'" for field in path])


def get_field_by_path_safe(tree: PyTree, path: PyTreePath) -> tuple[bool, Any]:
    for path_part in normalize_path(path):
        if not isinstance(tree, dict) or path_part not in tree:
            return False, None
        tree = tree[path_part]

    return True, tree


def get_field_by_path(tree: PyTree, path: PyTreePath) -> Any:
    path_exists, value = get_field_by_path_safe(tree, path)
    if not path_exists:
        raise ValueError(f"Path {format_path(path)} not found in the tree {tree}")

    return value


def set_field_by_path(tree: PyTree, path: PyTreePath, value: Any):
    path_parts = normalize_path(path)
    subtree = tree

    for path_part in path_parts[:-1]:
        if path_part not in subtree:
            subtree[path_part] = {}
        subtree = subtree[path_part]

    subtree[path_parts[-1]] = value


def delete_field_by_path(tree: PyTree, path: PyTreePath):
    path_parts = normalize_path(path)
    subtree = get_field_by_path_safe(tree, path_parts[:-1])[1]

    if isinstance(subtree, dict) and path_parts[-1] in subtree:
        del subtree[path_parts[-1]]
