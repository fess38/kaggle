import random
from typing import Iterable

from fess38.util.typing import PyTree

from .base import MapOpBase
from .config import ShuffleMapOpConfig


class ShuffleMapOp(MapOpBase):
    def __init__(self, config: ShuffleMapOpConfig):
        super().__init__(config, self._map_fn)

    def _map_fn(self, records: Iterable[PyTree], role: str | None) -> Iterable[PyTree]:
        random.seed(self.config.seed)
        records = list(records)
        random.shuffle(records)
        yield from records
