from collections.abc import Callable
from typing import Any, Iterable


class Registry:
    def __init__(self, name):
        self._name = name
        self._mapping: dict[str, Any] = {}
        self._inverse_mapping: dict[Any, str] = {}

    def __call__(self, key: str) -> Callable[[Any], Any]:
        def _wrapper(what: Any) -> Any:
            if key in self._mapping:
                raise ValueError(
                    f"Name '{key}' is already registered in registry {self._name}."
                )

            if what in self._inverse_mapping:
                raise ValueError(
                    f"Value '{what}' is already registered in registry {self._name}."
                )

            self._mapping[key] = what
            self._inverse_mapping[what] = key

            return what

        return _wrapper

    def __contains__(self, key: str) -> bool:
        return key in self._mapping

    def __getitem__(self, item: str | tuple[str, bool]) -> Any:
        key_or_value, inverse = item if isinstance(item, tuple) else (item, False)
        return (self._get_by_value if inverse else self._get_by_key)(key_or_value)

    def keys(self) -> Iterable[str]:
        return self._mapping.keys()

    def values(self) -> Iterable[Any]:
        return self._inverse_mapping.keys()

    def _get_by_key(self, key: str) -> Any:
        if key not in self._mapping:
            raise ValueError(
                f"Name '{key}' is not registered in registry {self._name}."
            )

        return self._mapping[key]

    def _get_by_value(self, value: Any) -> Any:
        if value not in self._inverse_mapping:
            raise ValueError(
                f"Name '{value}' is not registered in registry {self._name}."
            )

        return self._inverse_mapping[value]
