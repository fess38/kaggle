from fess38.util.config import ConfigBase

from ..backend.config import BackendConfig, LocalBackendConfig
from ..io.dataset_reference import InputDatasetReference, OutputDatasetReference


class OpConfigBase(ConfigBase):
    name: str | None = None
    backend: BackendConfig = LocalBackendConfig()
    inputs: tuple[InputDatasetReference, ...] = ()
    outputs: tuple[OutputDatasetReference, ...] = ()


class ConsumeOpConfigBase(OpConfigBase):
    ...


class ProduceOpConfigBase(OpConfigBase):
    ...


class MapOpConfigBase(OpConfigBase):
    ...


class MapReduceOpConfigBase(OpConfigBase):
    ...
