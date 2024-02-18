from typing import Annotated, Literal

import pydantic
from fess38.util.config import ConfigBase


class BackendInstructionConfigBase(ConfigBase):
    ...


class SetInputRecordClassInstructionConfig(BackendInstructionConfigBase):
    type: Literal["set_input_record_class"] = "set_input_record_class"
    record_class: str
    index: int | None = None
    role: str | None = None


class SetOutputRecordClassInstructionConfig(SetInputRecordClassInstructionConfig):
    type: Literal["set_output_record_class"] = "set_output_record_class"


BackendInstructionConfig = Annotated[
    SetInputRecordClassInstructionConfig,
    SetOutputRecordClassInstructionConfig,
    pydantic.Field(discriminator="type"),
]
