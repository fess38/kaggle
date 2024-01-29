from typing import Any, Iterable, Protocol

from fess38.util.typing import PyTree

from ..io.record import OutputIterable


class RunFn(Protocol):
    def __call__(self):
        ...


class ConsumeFn(Protocol):
    def __call__(self, record: PyTree, role: str | None):
        ...


class ConsumeAggregatorFn(Protocol):
    def __call__(self, records: Iterable[PyTree], role: str | None) -> None:
        ...


class ProduceFn(Protocol):
    def __call__(self) -> Iterable[PyTree]:
        ...


class MapFn(Protocol):
    def __call__(self, record: PyTree, role: str | None) -> OutputIterable:
        ...


class MapAggregatorFn(Protocol):
    def __call__(self, records: Iterable[PyTree], role: str | None) -> OutputIterable:
        ...


class FilterFn(Protocol):
    def __call__(self, record: PyTree, role: str | None, **kwargs) -> bool:
        ...


class MapReduceMapFn(Protocol):
    def __call__(
        self, record: PyTree, role: str | None
    ) -> tuple[Any, Iterable[PyTree]]:
        ...


class MapReduceMapAggregatorFn(Protocol):
    def __call__(
        self,
        records: Iterable[PyTree],
        role: str | None,
    ) -> tuple[Any, Iterable[PyTree]]:
        ...


class MapReduceReduceFn(Protocol):
    def __call__(self, key: Any, records: Iterable[PyTree]) -> OutputIterable:
        ...


class MapReduceReduceAggregatorFn(Protocol):
    def __call__(
        self,
        record_groups: Iterable[tuple[Any, Iterable[PyTree]]],
    ) -> OutputIterable:
        ...
