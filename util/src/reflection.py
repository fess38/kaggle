import importlib
from types import ModuleType


def find_module(class_name: str) -> ModuleType:
    return importlib.import_module(".".join(class_name.split(".")[:-1]))


def find_class(class_name: str) -> type:
    return getattr(find_module(class_name), class_name.split(".")[-1])
