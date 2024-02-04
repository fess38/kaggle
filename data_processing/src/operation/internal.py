import inspect

from fess38.data_processing.operation.config import (
    ConsumeOpConfigBase,
    MapOpConfigBase,
    MapReduceOpConfigBase,
    ProduceOpConfigBase,
)
from fess38.data_processing.operation.mapper.base import OpBase


def create_dummy_op(
    config: dict,
    op_class: type,
    num_inputs: int | None = None,
    num_outputs: int | None = None,
) -> OpBase:
    config["name"] = ""
    dataset_reference_config = {
        "type": "file",
        "path": "",
        "record_formatter": {"type": "jsonl"},
    }

    config_class = inspect.signature(op_class).parameters["config"].annotation
    if (
        issubclass(config_class, ConsumeOpConfigBase)
        or issubclass(config_class, MapOpConfigBase)
        or issubclass(config_class, MapReduceOpConfigBase)
    ):
        config.setdefault("inputs", [])
        for _ in range(num_inputs or 1):
            config["inputs"].append(dataset_reference_config)

    if (
        issubclass(config_class, ProduceOpConfigBase)
        or issubclass(config_class, MapOpConfigBase)
        or issubclass(config_class, MapReduceOpConfigBase)
    ):
        config.setdefault("outputs", [])
        for _ in range(num_outputs or 1):
            config["outputs"].append(dataset_reference_config)

    return op_class(config_class(**config))
