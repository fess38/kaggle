from fess38.utils.config import ConfigBase

from ..backend.config import BackendConfig, LocalBackendConfig
from ..io.dataset_reference import InputDatasetReference, OutputDatasetReference


class TransformConfigBase(ConfigBase):
    name: str | None = None
    backend: BackendConfig = LocalBackendConfig()
    inputs: tuple[InputDatasetReference, ...] = ()
    outputs: tuple[OutputDatasetReference, ...] = ()


class CreateTransformConfigBase(TransformConfigBase):
    ...


class MapTransformConfigBase(TransformConfigBase):
    ...


class MapReduceTransformConfigBase(TransformConfigBase):
    ...
