from typing import Annotated, Literal

import pydantic
from fess38.util.config import ConfigBase


class FilterConfigBase(ConfigBase):
    record_dropped_to_role: str | None = None


class EvaluateExpressionsMapperConfig(FilterConfigBase):
    type: Literal["evaluate_expression"] = "evaluate_expression"
    expession: str


FilterConfig = Annotated[
    (EvaluateExpressionsMapperConfig),
    pydantic.Field(discriminator="type"),
]
