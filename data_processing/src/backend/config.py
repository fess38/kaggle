from typing import Annotated, Literal

import pydantic
from fess38.utils.config import ConfigBase


class BackendConfigBase(ConfigBase):
    ...


class LocalBackendConfig(BackendConfigBase):
    type: Literal["local"] = "local"


BackendConfig = Annotated[
    LocalBackendConfig,
    pydantic.Field(discriminator="type"),
]
