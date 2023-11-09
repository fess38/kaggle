from typing import Any, Literal

from ..io.record import OutputIterable
from . import operation_library
from .map import MapOpBase, MapOpConfigBase


@operation_library("fess38.data_processing.operations.merge.MergeOp")
class MergeOpConfig(MapOpConfigBase):
    type: Literal["merge"] = "merge"


class MergeOp(MapOpBase):
    def __init__(self, config: MergeOpConfig):
        def _map_fn(record: Any, role: str | None) -> OutputIterable:
            yield record

        super().__init__(config, _map_fn)
