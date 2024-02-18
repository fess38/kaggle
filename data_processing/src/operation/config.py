from fess38.util.config import ConfigBase

from ..backend.config import (
    BackendConfig,
    ConsumeBackendOpConfig,
    LocalBackendConfig,
    MapBackendOpConfig,
    MapReduceBackendOpConfig,
    ProduceBackendOpConfig,
    RunBackendOpConfig,
)


class OpConfigBase(ConfigBase):
    name: str  # set automatically
    backend: BackendConfig = LocalBackendConfig()


class RunOpConfigBase(OpConfigBase, RunBackendOpConfig):
    ...


class ConsumeOpConfigBase(OpConfigBase, ConsumeBackendOpConfig):
    ...


class ProduceOpConfigBase(OpConfigBase, ProduceBackendOpConfig):
    ...


class MapOpConfigBase(OpConfigBase, MapBackendOpConfig):
    ...


class MapReduceOpConfigBase(OpConfigBase, MapReduceBackendOpConfig):
    ...
