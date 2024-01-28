from typing import Any

import pytest
from fess38.util.pytree import (
    delete_field_by_path,
    get_field_by_path,
    get_field_by_path_safe,
    normalize_path,
    set_field_by_path,
)
from fess38.util.typing import PyTree, PyTreePath


@pytest.mark.parametrize(
    ("path", "expected"),
    [
        ("foo", ("foo",)),
        (("foo", "bar"), ("foo", "bar")),
        ("foo.bar", ("foo", "bar")),
    ],
)
def test_normalize_path(path: PyTreePath, expected: tuple[str]):
    assert normalize_path(path) == expected


@pytest.mark.parametrize(
    ("tree", "path", "expected"),
    [
        ({"a": {"b": 1}}, "a.b", (True, 1)),
        ({"a": {"b": {}}}, "a.b", (True, {})),
        ({"a": {"b": 1}}, "a.b.c", (False, None)),
        ({"a": {"b": 1}}, "c", (False, None)),
    ],
)
def test_get_field_by_path_safe(
    tree: PyTree, path: PyTreePath, expected: tuple[bool, Any]
):
    assert get_field_by_path_safe(tree, path) == expected


@pytest.mark.parametrize(
    ("tree", "path", "is_raise", "expected"),
    [
        ({"a": {"b": 1}}, "a.b", False, 1),
        ({"a": {"b": {}}}, "a.b", False, {}),
        ({"a": {"b": 1}}, "a.b.c", True, None),
        ({"a": {"b": 1}}, "c", True, None),
    ],
)
def test_get_field_by_path(
    tree: PyTree, path: PyTreePath, is_raise: bool, expected: Any
):
    if is_raise:
        with pytest.raises(ValueError):
            get_field_by_path(tree, path)
        return

    assert get_field_by_path(tree, path) == expected


@pytest.mark.parametrize(
    ("tree", "path", "value"),
    [
        ({"a": {"b": 1}}, "a.c", "q"),
        ({"a": {"b": {}}}, "a.b.c", 1),
        ({"a": {"b": 1}}, "a.c.b", "z"),
        ({"a": {"b": 1}}, "c", False),
    ],
)
def test_set_field_by_path(tree: PyTree, path: PyTreePath, value: Any):
    set_field_by_path(tree, path, value)
    assert value == get_field_by_path(tree, path)


@pytest.mark.parametrize(
    ("tree", "path"),
    [
        ({"a": {"b": 1}}, "a"),
        ({"a": {"b": 1}}, "a.b"),
        ({"a": {"b": 1}}, "a.c"),
        ({"a": {"b": {}}}, "a.b.c"),
        ({"a": {"b": 1}}, "a.c.b"),
        ({"a": {"b": 1}}, "c"),
    ],
)
def test_delete_field_by_path(tree: PyTree, path: PyTreePath):
    delete_field_by_path(tree, path)
    assert not get_field_by_path_safe(tree, path)[0]
