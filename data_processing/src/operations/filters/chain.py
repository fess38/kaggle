import functools
from typing import Any, Literal, Protocol

from ...io.record import OutputIterable, OutputRecord
from .. import operation_library
from ..map import MapTransformBase, MapTransformConfigBase
from ..protocol import MapFn
from . import filter_library
from .config import FilterConfig


class FilterFn(Protocol):
    def __call__(self, record: Any, **kwargs) -> bool:
        ...


@operation_library("fess38.data_processing.operations.filters.chain.FilterChainTransform")
class FilterChainTransformConfig(MapTransformConfigBase):
    type: Literal["filter_chain"] = "filter_chain"
    filters: list[FilterConfig]


class FilterChainTransform(MapTransformBase):
    def __init__(self, config: FilterChainTransformConfig):
        super().__init__(
            config=config,
            map_fn=self._create_map_fn(config),
        )

    def _create_map_fn(self, config: FilterChainTransformConfig) -> MapFn:
        filter_fns: list[FilterFn] = []
        for filter_config in config.filters:
            filter_fn = filter_library[filter_config.type]
            filter_kwargs = filter_config.model_dump(exclude={"vars", "type", "record_dropped_to_role"})
            filter_fns.append(functools.partial(filter_fn, **filter_kwargs))

        def _map_fn(
            record: Any,
            role: str | None,
        ) -> OutputIterable:
            should_keep = True
            drop_to_role = None
            for filter_config, filter_fn in zip(config.filters, filter_fns):
                should_keep &= filter_fn(record)
                if not should_keep:
                    drop_to_role = filter_config.record_dropped_to_role
                    break

            if should_keep:
                yield record
            elif drop_to_role is not None:
                yield OutputRecord(record, role=drop_to_role)

        return _map_fn
