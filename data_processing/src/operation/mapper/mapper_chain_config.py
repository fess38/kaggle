from typing import Annotated, Any, Literal

import pydantic
from fess38.util.config import ConfigBase
from fess38.util.typing import PyTreePath


class MapperConfigBase(ConfigBase):
    ...


class ExecuteExpressionsMapperConfig(MapperConfigBase):
    type: Literal["execute_expressions"] = "execute_expressions"
    expressions: list[str]


class AddConstantFieldMapperConfig(MapperConfigBase):
    type: Literal["add_constant_field"] = "add_constant_field"
    path: PyTreePath
    value: Any


MapperConfig = Annotated[
    (ExecuteExpressionsMapperConfig | AddConstantFieldMapperConfig),
    pydantic.Field(discriminator="type"),
]
