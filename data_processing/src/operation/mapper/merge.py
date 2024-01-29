from typing import Iterable

from fess38.util.typing import PyTree

from .base import MapOpBase
from .config import MergeOpMapConfig


class MergeMapOp(MapOpBase):
    def __init__(self, config: MergeOpMapConfig):
        super().__init__(config, self._map_fn)

    def _map_fn(self, record: PyTree, role: str | None) -> Iterable[PyTree]:
        yield record
