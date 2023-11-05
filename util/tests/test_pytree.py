import pytest
from fess38.util.pytree import format_path, get_field_by_path, normalize_path


def test_normalize_path():
    assert normalize_path("foo") == ("foo",)
    assert normalize_path(("foo", "bar")) == ("foo", "bar")


def test_format_path():
    assert format_path("foo") == "'foo'"
    assert format_path(("foo", "bar")) == "'foo'=>'bar'"


def test_get_field_by_path():
    assert get_field_by_path({"foo": {"bar": 123, "a": "b"}}, ("foo", "bar")) == 123
    assert get_field_by_path({"foo": {"a": "b"}}, "foo") == {"a": "b"}


def test_get_field_by_path_raises():
    with pytest.raises(ValueError):
        get_field_by_path({"foo": {"a": "b"}}, "b")
