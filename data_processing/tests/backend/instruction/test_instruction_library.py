import pytest
from fess38.data_processing.backend.instruction.config import BackendInstructionConfig
from fess38.data_processing.backend.instruction.instruction_library import (
    set_input_record_class,
)
from fess38.data_processing.operation.config import MapOpConfigBase


@pytest.mark.parametrize(
    ("config", "instruction", "expected"),
    [
        (
            {
                "name": "",
                "inputs": [
                    {
                        "type": "file",
                        "path": "0",
                        "record_formatter": {"type": "jsonl"},
                    },
                    {
                        "type": "file",
                        "path": "1",
                        "record_formatter": {"type": "jsonl"},
                    },
                ],
            },
            {
                "type": "set_input_record_class",
                "record_class": "foo.bar",
            },
            [0, 1],
        ),
        (
            {
                "name": "",
                "inputs": [
                    {
                        "type": "file",
                        "path": "0",
                        "record_formatter": {"type": "jsonl"},
                    },
                    {
                        "type": "file",
                        "path": "1",
                        "record_formatter": {"type": "jsonl"},
                    },
                ],
            },
            {
                "type": "set_input_record_class",
                "record_class": "foo.bar",
                "index": 1,
            },
            [1],
        ),
        (
            {
                "name": "",
                "inputs": [
                    {
                        "type": "file",
                        "path": "0",
                        "record_formatter": {"type": "jsonl"},
                    },
                    {
                        "type": "file",
                        "path": "1",
                        "role": "foo",
                        "record_formatter": {"type": "jsonl"},
                    },
                    {
                        "type": "file",
                        "path": "2",
                        "role": "foo",
                        "record_formatter": {"type": "jsonl"},
                    },
                ],
            },
            {
                "type": "set_input_record_class",
                "record_class": "foo.bar",
                "role": "foo",
            },
            [1, 2],
        ),
    ],
)
def test_set_input_record_class(config: dict, instruction: dict, expected: list[int]):
    actual = set_input_record_class(
        MapOpConfigBase.model_validate(config),
        BackendInstructionConfig.model_validate(instruction),
    )
    for i, input in enumerate(actual.inputs):
        if i in expected:
            assert input.record_class is not None
        else:
            assert input.record_class is None
