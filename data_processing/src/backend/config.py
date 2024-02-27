from typing import Annotated, Literal

import pydantic
from fess38.util.config import ConfigBase

from ..io.dataset_reference import InputDatasetReference, OutputDatasetReference


class BackendConfigBase(ConfigBase):
    ...


class LocalBackendConfig(BackendConfigBase):
    type: Literal["local"] = "local"


BackendConfig = Annotated[
    LocalBackendConfig,
    pydantic.Field(discriminator="type"),
]


class BackendOpConfigBase(ConfigBase):
    input_files: dict[str, str] = {}
    output_files: dict[str, str] = {}
    inputs: tuple[InputDatasetReference, ...] = ()
    outputs: tuple[OutputDatasetReference, ...] = ()


class RunBackendOpConfig(BackendOpConfigBase):
    ...


class ConsumeBackendOpConfig(BackendOpConfigBase):
    ...


class ProduceBackendOpConfig(BackendOpConfigBase):
    ...


class MapBackendOpConfig(BackendOpConfigBase):
    ...


class MapReduceBackendOpConfig(BackendOpConfigBase):
    ...
