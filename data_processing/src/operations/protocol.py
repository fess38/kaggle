from typing import Any, Iterable, Protocol

from ..io.record import OutputIterable


class ConsumeFn(Protocol):
    def __call__(self, record: Any, role: str | None):
        ...


class ConsumeAggregatorFn(Protocol):
    def __call__(self, records: Iterable[Any], role: str | None) -> None:
        ...


class ProduceFn(Protocol):
    def __call__(self) -> Iterable[Any]:
        ...


class MapFn(Protocol):
    def __call__(self, record: Any, role: str | None) -> OutputIterable:
        ...


class MapAggregatorFn(Protocol):
    def __call__(self, records: Iterable[Any], role: str | None) -> OutputIterable:
        ...


class MapReduceMapFn(Protocol):
    def __call__(self, record: Any, role: str | None) -> tuple[Any, Iterable[Any]]:
        ...


class MapReduceMapAggregatorFn(Protocol):
    def __call__(
        self,
        records: Iterable[Any],
        role: str | None,
    ) -> tuple[Any, Iterable[Any]]:
        ...


class MapReduceReduceFn(Protocol):
    def __call__(self, key: Any, records: Iterable[Any]) -> OutputIterable:
        ...


class MapReduceReduceAggregatorFn(Protocol):
    def __call__(
        self,
        record_groups: Iterable[tuple[Any, Iterable[Any]]],
    ) -> OutputIterable:
        ...
