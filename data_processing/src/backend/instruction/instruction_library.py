from typing import Sequence

from fess38.util.registry import Registry

from ..config import BackendOpConfig
from .config import BackendInstructionBase, SetRecordClassInstruction

instruction_library = Registry("instruction_library")


def execute_instructions(
    config: BackendOpConfig, instructions: Sequence[BackendInstructionBase]
) -> BackendOpConfig:
    config_dict = config.model_dump()
    for instruction in instructions:
        instruction_library[instruction.type](config_dict, instruction)

    return type(config).model_validate(config_dict)


@instruction_library("set_record_class")
def set_record_class(
    config: BackendOpConfig,
    instruction: SetRecordClassInstruction,
) -> BackendOpConfig:
    for i, io_item in enumerate(config.get(instruction.io, [])):
        if (
            (instruction.index is None and instruction.role is None)
            or i == instruction.index
            or (
                io_item.get("role") is not None
                and io_item.get("role") == instruction.role
            )
        ):
            if io_item.get("record_class") is None:
                io_item["record_class"] = instruction.record_class
