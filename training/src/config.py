from typing import Any, Literal

from fess38.data_processing.operation import operation_library
from fess38.data_processing.operation.config import ConsumeOpConfigBase, MapOpConfigBase


class TrainOpConfigBase(ConsumeOpConfigBase):
    random_state: int = 0
    kwargs: dict[str, Any] = {}


@operation_library("fess38.training.train_test_split.TrainTestSplitMapOp")
class TrainTestSplitMapOpConfig(MapOpConfigBase):
    type: Literal["train_test_split"] = "train_test_split"
    based_on: list[str]
    sampling_rate: float


@operation_library("fess38.training.metric_calculator.MetricCalculationConsumeOp")
class MetricCalculationConsumeOpConfig(MapOpConfigBase):
    type: Literal["metric_calculator"] = "metric_calculator"
    metric_configs: dict[str, Any]
