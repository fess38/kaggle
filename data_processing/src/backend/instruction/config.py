from typing import Literal

from fess38.util.config import ConfigBase


class BackendInstructionBase(ConfigBase):
    ...


class SetRecordClassInstruction(BackendInstructionBase):
    type: Literal["set_record_class"] = "set_record_class"
    io: Literal["inputs", "outputs"]
    record_class: str
    index: int | None = None
    role: str | None = None
