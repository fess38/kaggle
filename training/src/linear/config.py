from typing import Literal

from fess38.data_processing.operation import operation_library
from fess38.data_processing.operation.config import MapOpConfigBase

from ..config import TrainOpConfigBase


@operation_library(
    "fess38.training.linear.logistic_regression.LogisticRegressionTrainOp"
)
class LogisticRegressionTrainOpConfig(TrainOpConfigBase):
    type: Literal["logistic_regression_trainer"] = "logistic_regression_trainer"


@operation_library(
    "fess38.training.linear.logistic_regression.LogisticRegressionInferenceOp"
)
class LogisticRegressionInferenceOpConfig(MapOpConfigBase):
    type: Literal["logistic_regression_inference"] = "logistic_regression_inference"
