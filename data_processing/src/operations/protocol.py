from typing import Any, Iterable, Protocol

from ..io.record import OutputIterable


class ProduceFn(Protocol):
    def __call__(self) -> Iterable[Any]:
        ...


class MapFn(Protocol):
    def __call__(self, record: Any, role: str | None) -> OutputIterable:
        ...


class MapReduceMapFn(Protocol):
    def __call__(self, record: Any, role: str | None) -> tuple[Any, Iterable[Any]]:
        ...


class MapReduceReduceFn(Protocol):
    def __call__(self, key: Any, records: Iterable[Any]) -> OutputIterable:
        ...
