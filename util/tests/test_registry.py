import pytest
from fess38.util.registry import Registry


def create_registry() -> Registry:
    registry = Registry("foo")

    @registry("a")
    def a():
        ...

    @registry("b")
    def b():
        ...

    @registry("c")
    def c():
        ...

    return registry, [a, b, c]


def test_registry_contains():
    registry, _ = create_registry()
    assert "a" in registry
    assert "z" not in registry


def test_registry_getitem():
    registry, functions = create_registry()
    assert registry["a"] == functions[0]
    assert registry["a", False] == functions[0]
    assert registry[functions[1], True] == "b"


def test_registry_keys():
    registry, _ = create_registry()
    assert list(registry.keys()) == ["a", "b", "c"]


def test_registry_values():
    registry, functions = create_registry()
    assert list(registry.values()) == functions


def test_registry_unknown_key():
    registry, _ = create_registry()
    with pytest.raises(ValueError):
        registry["d"]

    with pytest.raises(ValueError):
        registry["d", False]


def test_registry_unknown_value():
    registry, _ = create_registry()
    with pytest.raises(ValueError):
        registry[dict, True]


def test_registry_already_registeted_name():
    registry = Registry("foo")
    with pytest.raises(ValueError):

        @registry("a")
        def a():
            ...

        @registry("a")
        def b():
            ...


def test_registry_already_registeted_value():
    registry = Registry("foo")
    with pytest.raises(ValueError):

        @registry("a")
        @registry("b")
        def a():
            ...
