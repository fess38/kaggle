import random
from typing import Any, Iterable

from .base import MapOpBase
from .config import ShuffleMapOpConfig


class ShuffleMapOp(MapOpBase):
    def __init__(self, config: ShuffleMapOpConfig):
        super().__init__(config, self._map_fn)

    def _map_fn(self, records: Iterable[Any], role: str | None) -> Iterable[Any]:
        random.seed(self._config.random_state)
        records = list(records)
        random.shuffle(records)
        yield from records
