from typing import Literal

from fess38.data_processing.operation import operation_library
from fess38.data_processing.operation.config import MapOpConfigBase

from ..config import TrainOpConfigBase


@operation_library("fess38.training.linear.LinearRegressionTrainOp")
class LinearRegressionTrainOpConfig(TrainOpConfigBase):
    type: Literal["linear_regression_trainer"] = "linear_regression_trainer"
    model_name: Literal["LinearRegression", "SGDRegressor"] = "SGDRegressor"


@operation_library("fess38.training.linear.LinearRegressionInferenceOp")
class LinearRegressionInferenceOpConfig(MapOpConfigBase):
    type: Literal["linear_regression_inference"] = "linear_regression_inference"
    batch_size: int = 128


@operation_library("fess38.training.linear.LogisticRegressionTrainOp")
class LogisticRegressionTrainOpConfig(TrainOpConfigBase):
    type: Literal["logistic_regression_trainer"] = "logistic_regression_trainer"
    model_name: Literal["LogisticRegression", "SGDClassifier"] = "SGDClassifier"


@operation_library("fess38.training.linear.LogisticRegressionInferenceOp")
class LogisticRegressionInferenceOpConfig(MapOpConfigBase):
    type: Literal["logistic_regression_inference"] = "logistic_regression_inference"
    batch_size: int = 128
