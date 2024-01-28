import functools

from ...io.record import OutputIterable
from .base import MapOpBase
from .config import FilterChainMapOpConfig
from .mapper_library import mapper_library


class MapperChainMapOp(MapOpBase):
    def __init__(self, config: FilterChainMapOpConfig):
        super().__init__(config=config, map_fn=self._map_fn)
        self._mapper_fns = []

        for mapper_config in config.mappers:
            mapper_fn = mapper_library[mapper_config.type]
            mapper_kwargs = mapper_config.model_dump(exclude={"vars", "type"})
            self._mapper_fns.append(functools.partial(mapper_fn, **mapper_kwargs))

    def _map_fn(self, record: dict, role: str | None) -> OutputIterable:
        for mapper_fn in self._mapper_fns:
            record = mapper_fn(record, role)

        yield record
