from typing import Annotated, Literal

import pydantic
from fess38.util.config import ConfigBase


class MetricConfigBase(ConfigBase):
    ...


class AccuracyMetricConfig(MetricConfigBase):
    type: Literal["accuracy"] = "accuracy"
    threshold: float
    normalize: bool = True


class PrecisionMetricConfig(MetricConfigBase):
    type: Literal["precision"] = "precision"
    threshold: float
    labels: list[str] | None = None
    pos_label: int | float | bool | str = 1
    average: str = "binary"
    zero_division: int = 0


class RecallMetricConfig(PrecisionMetricConfig):
    type: Literal["recall"] = "recall"


class F1MetricConfig(PrecisionMetricConfig):
    type: Literal["f1"] = "f1"


class RocAucMetricConfig(MetricConfigBase):
    type: Literal["roc_auc"] = "roc_auc"
    average: str | None = "macro"
    max_fpr: float | None = None
    multi_class: str = "raise"
    labels: list[str] | None = None


class R2MetricConfig(MetricConfigBase):
    type: Literal["r2"] = "r2"
    multioutput: str | None = "uniform_average"
    force_finite: bool = True


class MaxAccuracyThresholdMetricConfig(MetricConfigBase):
    type: Literal["max_accuracy_threshold"] = "max_accuracy_threshold"


class MaxF1ThresholdMetricConfig(MetricConfigBase):
    type: Literal["max_f1_threshold"] = "max_f1_threshold"
    pos_label: int | float | bool | str | None = None
    drop_intermediate: bool = False


class PrecisionRecallCurveMetricConfig(MetricConfigBase):
    type: Literal["precision_recall_curve"] = "precision_recall_curve"
    labels: list[str] | None = None
    classes_to_plot: list[str | int] | None = None


class RocCurveMetricConfig(MetricConfigBase):
    type: Literal["roc_curve"] = "roc_curve"
    labels: list[str] | None = None
    classes_to_plot: list[str | int] | None = None


class ConfusionMatrixMetricConfig(MetricConfigBase):
    type: Literal["confusion_matrix"] = "confusion_matrix"
    class_names: list[str] | None = None


MetricConfig = Annotated[
    (
        AccuracyMetricConfig
        | PrecisionMetricConfig
        | RecallMetricConfig
        | F1MetricConfig
        | RocAucMetricConfig
        | R2MetricConfig
        | MaxAccuracyThresholdMetricConfig
        | MaxF1ThresholdMetricConfig
        | PrecisionRecallCurveMetricConfig
        | RocCurveMetricConfig
        | ConfusionMatrixMetricConfig
    ),
    pydantic.Field(discriminator="type"),
]
