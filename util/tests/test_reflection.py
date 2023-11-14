from fess38.util.reflection import find_class, find_module


def test_find_module():
    class_name = "fess38.util.registry.Registry"
    module = find_module(class_name)
    assert module.__name__ == "fess38.util.registry"


def test_find_class():
    class_name = "fess38.util.registry.Registry"
    class_ = find_class(class_name)
    assert class_.__name__ == "Registry"
