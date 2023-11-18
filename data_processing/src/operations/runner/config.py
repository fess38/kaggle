from typing import Literal

from .. import operation_library
from ..config import ProduceOpConfigBase


@operation_library("fess38.data_processing.operations.runner.bash.BashRunOp")
class BashRunOpConfig(ProduceOpConfigBase):
    type: Literal["bash"] = "bash"
    vars: dict[str, str] = {}
    commands: list[str]
