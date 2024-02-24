from typing import Literal

from fess38.data_processing.operation import operation_library
from fess38.data_processing.operation.config import MapOpConfigBase

from ..config import TrainOpConfigBase


@operation_library("fess38.training.linear.LinearRegressionTrainOp")
class LinearRegressionTrainOpConfig(TrainOpConfigBase):
    type: Literal["linear_regression_trainer"] = "linear_regression_trainer"
    fit_intercept: bool = True
    copy_X: bool = True
    n_jobs: int | None = None
    positive: bool = False


@operation_library("fess38.training.linear.SGDRegressorTrainOp")
class SGDRegressorTrainOpConfig(TrainOpConfigBase):
    type: Literal["sgd_regressor_trainer"] = "sgd_regressor_trainer"
    loss: Literal[
        "squared_error", "huber", "epsilon_insensitive", "squared_epsilon_insensitive"
    ] = "squared_error"
    penalty: Literal["l2", "l1", "elasticnet"] | None = "l2"
    alpha: float = 0.0001
    l1_ratio: float = 0.15
    fit_intercept: bool = True
    max_iter: int = 1000
    tol: float | None = 1e-3
    shuffle: bool = True
    verbose: int = 0
    epsilon: float = 0.1
    random_state: int | None = None
    learning_rate: Literal[
        "constant", "optimal", "invscaling", "adaptive"
    ] = "invscaling"
    eta0: float = 0.01
    power_t: float = 0.25
    early_stopping: bool = False
    validation_fraction: float = 0.1
    n_iter_no_change: int = 5
    warm_start: bool = False
    average: bool | int = False


@operation_library("fess38.training.linear.LinearRegressionInferenceOp")
class LinearRegressionInferenceOpConfig(MapOpConfigBase):
    type: Literal["linear_regression_inference"] = "linear_regression_inference"
    batch_size: int = 128


@operation_library("fess38.training.linear.LogisticRegressionTrainOp")
class LogisticRegressionTrainOpConfig(TrainOpConfigBase):
    type: Literal["logistic_regression_trainer"] = "logistic_regression_trainer"
    penalty: Literal["l1", "l2", "elasticnet"] | None = "l2"
    dual: bool = False
    tol: float = 1e-4
    C: float = 1.0
    fit_intercept: bool = True
    intercept_scaling: float = 1
    class_weight: dict | Literal["balanced"] | None = None
    random_state: int | None = None
    solver: Literal[
        "lbfgs", "liblinear", "newton-cg", "newton-cholesky", "sag", "saga"
    ] = "lbfgs"
    max_iter: int = 100
    multi_class: Literal["auto", "ovr", "multinomial"] = "auto"
    verbose: int = 0
    warm_start: bool = False
    n_jobs: int | None = None
    l1_ratio: float | None = None


@operation_library("fess38.training.linear.SGDClassifierTrainOp")
class SGDClassifierTrainOpConfig(TrainOpConfigBase):
    type: Literal["sgd_classfier_trainer"] = "sgd_classfier_trainer"
    loss: Literal[
        "hinge",
        "log_loss",
        "modified_huber",
        "squared_hinge",
        "perceptron",
        "squared_error",
        "huber",
        "epsilon_insensitive",
        "squared_epsilon_insensitive",
    ] = "hinge"
    penalty: Literal["l2", "l1", "elasticnet", None] = "l2"
    alpha: float = 0.0001
    l1_ratio: float = 0.15
    fit_intercept: bool = True
    max_iter: int = 1000
    tol: float | None = 1e-3
    shuffle: bool = True
    verbose: int = 0
    epsilon: float = 0.1
    n_jobs: int | None = None
    random_state: int | None = None
    learning_rate: Literal["constant", "optimal", "invscaling", "adaptive"] = "optimal"
    eta0: float = 0.0
    power_t: float = 0.5
    early_stopping: bool = False
    validation_fraction: float = 0.1
    n_iter_no_change: int = 5
    class_weight: dict | Literal["balanced"] | None = None
    warm_start: bool = False
    average: bool | int | None = False


@operation_library("fess38.training.linear.LogisticRegressionInferenceOp")
class LogisticRegressionInferenceOpConfig(MapOpConfigBase):
    type: Literal["logistic_regression_inference"] = "logistic_regression_inference"
    batch_size: int = 128
