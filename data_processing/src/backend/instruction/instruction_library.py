from typing import Sequence

from fess38.util.registry import Registry

from ..config import BackendOpConfigBase
from .config import BackendInstructionBase, SetRecordClassInstruction

instruction_library = Registry("instruction_library")


def execute_instructions(
    config: BackendOpConfigBase, instructions: Sequence[BackendInstructionBase]
) -> BackendOpConfigBase:
    config_dict = config.model_dump()
    for instruction in instructions:
        instruction_library[instruction.type](config_dict, instruction)

    return type(config).model_validate(config_dict)


@instruction_library("set_record_class")
def set_record_class(config: dict, instruction: SetRecordClassInstruction):
    for i, dataset_reference in enumerate(config.get(instruction.io, [])):
        if (
            (instruction.index is None and instruction.role is None)
            or i == instruction.index
            or (
                dataset_reference.get("role") is not None
                and dataset_reference.get("role") == instruction.role
            )
        ):
            if dataset_reference.get("record_class") is None:
                dataset_reference["record_class"] = instruction.record_class
