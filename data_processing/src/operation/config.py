from fess38.util.config import ConfigBase

from ..backend.config import BackendConfig, LocalBackendConfig
from ..io.dataset_reference import InputDatasetReference, OutputDatasetReference


class OpConfigBase(ConfigBase):
    name: str  # set automatically
    backend: BackendConfig = LocalBackendConfig()
    input_files: dict[str, str] = {}
    output_files: dict[str, str] = {}
    inputs: tuple[InputDatasetReference, ...] = ()
    outputs: tuple[OutputDatasetReference, ...] = ()


class RunOpConfigBase(OpConfigBase):
    ...


class ConsumeOpConfigBase(OpConfigBase):
    ...


class ProduceOpConfigBase(OpConfigBase):
    ...


class MapOpConfigBase(OpConfigBase):
    ...


class MapReduceOpConfigBase(OpConfigBase):
    ...