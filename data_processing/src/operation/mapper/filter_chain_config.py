from typing import Annotated, Literal

import pydantic
from fess38.util.config import ConfigBase
from fess38.util.typing import PyTreePath


class FilterConfigBase(ConfigBase):
    record_dropped_to_role: str | None = None


class EvaluateExpressionsMapperConfig(FilterConfigBase):
    type: Literal["evaluate_expression"] = "evaluate_expression"
    expession: str


class DropEmptyFilterConfig(FilterConfigBase):
    type: Literal["drop_empty"] = "drop_empty"
    path: PyTreePath


FilterConfig = Annotated[
    (DropEmptyFilterConfig),
    pydantic.Field(discriminator="type"),
]
