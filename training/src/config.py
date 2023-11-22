from typing import Any, Literal

from fess38.data_processing.operation import operation_library
from fess38.data_processing.operation.config import ConsumeOpConfigBase, MapOpConfigBase


class TrainOpConfigBase(ConsumeOpConfigBase):
    random_state: int = 0
    kwargs: dict[str, Any] = {}


@operation_library("fess38.training.train_test_split.TrainTestSplitMapOp")
class TrainTestSplitMapOpConfig(MapOpConfigBase):
    type: Literal["train_test_split"] = "train_test_split"
    random_state: int = 0
    test_size: int | None = None
    test_proportion: float | None = None
