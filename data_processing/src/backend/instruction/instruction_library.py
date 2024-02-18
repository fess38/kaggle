from typing import Literal

from fess38.util.registry import Registry

from ..config import BackendOpConfig
from .config import BackendInstructionConfig

instruction_library = Registry("instruction_library")


def execute_instructions(
    config: BackendOpConfig, instructions: list[BackendInstructionConfig]
) -> BackendOpConfig:
    for instruction in instructions:
        config = instruction_library[instruction.type](config, instruction)

    return config


def _set_record_class(
    io_type: Literal["inputs", "outputs"],
    config: BackendOpConfig,
    instruction_config: BackendInstructionConfig,
) -> BackendOpConfig:
    patched_config = config.model_dump()
    for i, io_item in enumerate(patched_config[io_type]):
        if (
            (instruction_config.index is None and instruction_config.role is None)
            or i == instruction_config.index
            or (
                io_item["role"] is not None
                and io_item["role"] == instruction_config.role
            )
        ):
            io_item["record_class"] = instruction_config.record_class

    return type(config).model_validate(patched_config)


@instruction_library("set_input_record_class")
def set_input_record_class(
    config: BackendOpConfig,
    instruction_config: BackendInstructionConfig,
) -> BackendOpConfig:
    return _set_record_class("inputs", config, instruction_config)


@instruction_library("set_output_record_class")
def set_output_record_class(
    config: BackendOpConfig,
    instruction_config: BackendInstructionConfig,
) -> BackendOpConfig:
    return _set_record_class("outputs", config, instruction_config)
