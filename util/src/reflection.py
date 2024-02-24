import importlib
from functools import cache
from types import ModuleType


@cache
def find_module(class_name: str) -> ModuleType:
    return importlib.import_module(".".join(class_name.split(".")[:-1]))


@cache
def find_class(class_name: str) -> type:
    return getattr(find_module(class_name), class_name.split(".")[-1])


def full_path(class_: type) -> str:
    return f"{class_.__module__}.{class_.__qualname__}"
