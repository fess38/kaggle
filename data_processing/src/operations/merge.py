from typing import Any, Literal

from ..io.record import OutputIterable
from . import operation_library
from .map import MapTransformBase, MapTransformConfigBase


@operation_library("fess38.data_processing.operations.merge.MergeTransform")
class MergeTransformConfig(MapTransformConfigBase):
    type: Literal["merge"] = "merge"


class MergeTransform(MapTransformBase):
    def __init__(self, config: MergeTransformConfig):
        def _map_fn(record: Any, role: str | None) -> OutputIterable:
            yield record

        super().__init__(config, _map_fn)
