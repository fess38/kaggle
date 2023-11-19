from typing import Literal

from .. import operation_library
from ..config import MapOpConfigBase
from ..filter.config import FilterConfig


@operation_library(
    "fess38.data_processing.operation.mapper.filter_chain.FilterChainMapOp"
)
class FilterChainMapOpConfig(MapOpConfigBase):
    type: Literal["filter_chain"] = "filter_chain"
    filters: list[FilterConfig]


@operation_library("fess38.data_processing.operation.mapper.merge.MergeMapOp")
class MergeOpMapConfig(MapOpConfigBase):
    type: Literal["merge"] = "merge"
