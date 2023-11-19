import functools
from typing import Any

from ...io.record import OutputIterable, OutputRecord
from ..filter import filter_library
from ..protocol import FilterFn, MapFn
from .base import MapOpBase
from .config import FilterChainMapOpConfig


class FilterChainMapOp(MapOpBase):
    def __init__(self, config: FilterChainMapOpConfig):
        super().__init__(
            config=config,
            map_fn=self._create_map_fn(config),
        )

    def _create_map_fn(self, config: FilterChainMapOpConfig) -> MapFn:
        filter_fns: list[FilterFn] = []
        for filter_config in config.filters:
            filter_fn = filter_library[filter_config.type]
            filter_kwargs = filter_config.model_dump(
                exclude={"vars", "type", "record_dropped_to_role"}
            )
            filter_fns.append(functools.partial(filter_fn, **filter_kwargs))

        def _map_fn(record: Any, role: str | None) -> OutputIterable:
            should_keep = True
            drop_to_role = None
            for filter_config, filter_fn in zip(config.filters, filter_fns):
                should_keep &= filter_fn(record, role)
                if not should_keep:
                    drop_to_role = filter_config.record_dropped_to_role
                    break

            if should_keep:
                yield record
            elif drop_to_role is not None:
                yield OutputRecord(record, role=drop_to_role)

        return _map_fn
