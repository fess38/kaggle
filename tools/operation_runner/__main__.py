import importlib
import logging
from typing import Annotated, Union

import hydra
import pydantic
from fess38.data_processing.operations import operation_library
from fess38.data_processing.operations.filters.chain import *  # noqa: F401 F403
from fess38.data_processing.operations.merge import *  # noqa: F401 F403
from fess38.util.config import ConfigBase
from omegaconf import DictConfig

logger = logging.getLogger(__name__)

TransformConfig = Annotated[
    Union[tuple(operation_library.values())],
    pydantic.Field(discriminator="type"),
]


class TransoformChainConfig(ConfigBase):
    transforms: list[TransformConfig]


@hydra.main(version_base=None)
def main(cfg: DictConfig):
    if "config_path" not in cfg:
        raise ValueError("Provide ++config_path=...")

    config_path = cfg["config_path"]
    logger.info(f"Start executing operations from {config_path}")
    for config in TransoformChainConfig.from_file(cfg["config_path"]).transforms:
        operation_class_name = operation_library[type(config), True]
        parts = operation_class_name.split(".")
        module_name = ".".join(parts[:-1])
        class_name = parts[-1]
        class_ = getattr(importlib.import_module(module_name), class_name)
        logger.info(f"Start executing {class_name} operation")
        class_(config).run()
        logger.info("Finished executing operation")


if __name__ == "__main__":
    main()
