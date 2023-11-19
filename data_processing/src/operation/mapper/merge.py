from typing import Any

from ...io.record import OutputIterable
from .base import MapOpBase
from .config import MergeOpMapConfig


class MergeMapOp(MapOpBase):
    def __init__(self, config: MergeOpMapConfig):
        def _map_fn(record: Any, role: str | None) -> OutputIterable:
            yield record

        super().__init__(config, _map_fn)
