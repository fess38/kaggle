import logging
from typing import Annotated, Union

import hydra
import pydantic
from fess38.competitions.titanic import *  # noqa: F401 F403
from fess38.data_processing.operations import operation_library
from fess38.data_processing.operations.filters.chain import *  # noqa: F401 F403
from fess38.data_processing.operations.merge import *  # noqa: F401 F403
from fess38.data_processing.operations.producer.from_kaggle import *  # noqa: F401 F403
from fess38.data_processing.operations.runner.bash import *  # noqa: F401 F403
from fess38.util.config import ConfigBase
from fess38.util.reflection import find_class
from omegaconf import DictConfig

logger = logging.getLogger(__name__)

OpConfig = Annotated[
    Union[tuple(operation_library.values())],
    pydantic.Field(discriminator="type"),
]


class OpChainConfig(ConfigBase):
    ops: list[OpConfig]


@hydra.main(version_base=None)
def main(cfg: DictConfig):
    if "config_path" not in cfg:
        raise ValueError("Provide ++config_path=...")

    config_path = cfg["config_path"]
    logger.info(f"Start executing operations from {config_path}")
    for config in OpChainConfig.from_file(cfg["config_path"]).ops:
        operation_class_name = operation_library[type(config), True]
        class_ = find_class(operation_class_name)
        logger.info(f"Start executing {class_.__name__} operation")
        class_(config).run()
        logger.info("Finished executing operation")


if __name__ == "__main__":
    main()
