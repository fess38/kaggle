from typing import Annotated, Any, Literal

import pydantic
from fess38.util.config import ConfigBase
from fess38.util.typing import PyTreePath


class MapperConfigBase(ConfigBase):
    ...


class AddConstantFieldMapperConfig(MapperConfigBase):
    type: Literal["add_constant_field"] = "add_constant_field"
    path: PyTreePath
    value: Any


MapperConfig = Annotated[
    (AddConstantFieldMapperConfig),
    pydantic.Field(discriminator="type"),
]
