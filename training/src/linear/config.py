from typing import Literal

from fess38.data_processing.operation import operation_library

from ..config import InferenceOpConfigBase, TrainOpConfigBase


@operation_library("fess38.training.linear.LinearRegressionTrainOp")
class LinearRegressionTrainOpConfig(TrainOpConfigBase):
    """
    https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html
    """

    type: Literal["linear_regression_trainer"] = "linear_regression_trainer"


@operation_library("fess38.training.linear.SGDRegressorTrainOp")
class SGDRegressorTrainOpConfig(TrainOpConfigBase):
    """
    https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDRegressor.html
    """

    type: Literal["sgd_regressor_trainer"] = "sgd_regressor_trainer"


@operation_library("fess38.training.linear.LinearRegressionInferenceOp")
class LinearRegressionInferenceOpConfig(InferenceOpConfigBase):
    type: Literal["linear_regression_inference"] = "linear_regression_inference"
    batch_size: int = 128


@operation_library("fess38.training.linear.LogisticRegressionTrainOp")
class LogisticRegressionTrainOpConfig(TrainOpConfigBase):
    """
    https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html
    """

    type: Literal["logistic_regression_trainer"] = "logistic_regression_trainer"


@operation_library("fess38.training.linear.SGDClassifierTrainOp")
class SGDClassifierTrainOpConfig(TrainOpConfigBase):
    """
    https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDClassifier.html
    """

    type: Literal["sgd_classfier_trainer"] = "sgd_classfier_trainer"


@operation_library("fess38.training.linear.LogisticRegressionInferenceOp")
class LogisticRegressionInferenceOpConfig(InferenceOpConfigBase):
    type: Literal["logistic_regression_inference"] = "logistic_regression_inference"
    batch_size: int = 128
