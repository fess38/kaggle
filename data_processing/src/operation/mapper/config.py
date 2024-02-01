from typing import Literal

from .. import operation_library
from ..config import MapOpConfigBase
from .filter_chain_config import FilterConfig
from .mapper_chain_config import MapperConfig


@operation_library(
    "fess38.data_processing.operation.mapper.filter_chain.FilterChainMapOp"
)
class FilterChainMapOpConfig(MapOpConfigBase):
    type: Literal["filter_chain"] = "filter_chain"
    filters: list[FilterConfig]


@operation_library(
    "fess38.data_processing.operation.mapper.mapper_chain.MapperChainMapOp"
)
class MapperChainMapOpConfig(MapOpConfigBase):
    type: Literal["mapper_chain"] = "mapper_chain"
    mappers: list[MapperConfig]


@operation_library("fess38.data_processing.operation.mapper.merge.MergeMapOp")
class MergeOpMapConfig(MapOpConfigBase):
    type: Literal["merge"] = "merge"


@operation_library("fess38.data_processing.operation.mapper.shuffle.ShuffleMapOp")
class ShuffleMapOpConfig(MapOpConfigBase):
    type: Literal["shuffle"] = "shuffle"
    seed: int = 0
