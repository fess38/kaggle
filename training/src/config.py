from typing import Literal

from fess38.data_processing.operation import operation_library
from fess38.data_processing.operation.config import ConsumeOpConfigBase, MapOpConfigBase
from fess38.util.typing import PyTreePath

from .metrics.metric_config import MetricConfig


class TrainOpConfigBase(ConsumeOpConfigBase):
    ...


@operation_library("fess38.training.train_test_split.TrainTestSplitMapOp")
class TrainTestSplitMapOpConfig(MapOpConfigBase):
    type: Literal["train_test_split"] = "train_test_split"
    based_on: list[PyTreePath]
    sampling_rate: float


@operation_library(
    "fess38.training.metrics.metric_calculator.MetricCalculationConsumeOp"
)
class MetricCalculationConsumeOpConfig(ConsumeOpConfigBase):
    type: Literal["metric_calculator"] = "metric_calculator"
    metric_configs: list[MetricConfig]
