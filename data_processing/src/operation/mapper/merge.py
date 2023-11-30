from typing import Any, Iterable

from .base import MapOpBase
from .config import MergeOpMapConfig


class MergeMapOp(MapOpBase):
    def __init__(self, config: MergeOpMapConfig):
        super().__init__(config, self._map_fn)

    def _map_fn(self, record: Any, role: str | None) -> Iterable[Any]:
        yield record
