import pytest
from fess38.data_processing.backend.instruction.config import SetRecordClassInstruction
from fess38.data_processing.backend.instruction.instruction_library import (
    execute_instructions,
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
                        "record_class": "spam",
                    },
                    {
                        "type": "file",
                        "path": "1",
                        "record_formatter": {"type": "jsonl"},
                        "record_class": None,
                    },
                    {
                        "type": "file",
                        "path": "1",
                        "record_formatter": {"type": "jsonl"},
                    },
                ],
            },
            {
                "type": "set_record_class",
                "io": "inputs",
                "record_class": "foo.bar",
            },
            ["spam", "foo.bar", "foo.bar"],
        ),
        (
            {
                "name": "",
                "outputs": [
                    {
                        "type": "file",
                        "path": "0",
                        "record_formatter": {"type": "jsonl"},
                        "record_class": "spam",
                    },
                    {
                        "type": "file",
                        "path": "1",
                        "record_formatter": {"type": "jsonl"},
                        "record_class": None,
                    },
                    {
                        "type": "file",
                        "path": "1",
                        "record_formatter": {"type": "jsonl"},
                    },
                ],
            },
            {
                "type": "set_record_class",
                "io": "outputs",
                "record_class": "foo.bar",
                "index": 1,
            },
            ["spam", "foo.bar", None],
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
                        "record_class": "spam",
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
                "type": "set_record_class",
                "io": "inputs",
                "record_class": "foo.bar",
                "role": "foo",
            },
            [None, "spam", "foo.bar"],
        ),
    ],
)
def test_set_record_class(config: dict, instruction: dict, expected: list[str]):
    actual = execute_instructions(
        MapOpConfigBase.model_validate(config),
        [SetRecordClassInstruction.model_validate(instruction)],
    )
    for i, input in enumerate(actual.inputs):
        assert input.record_class == expected[i]
