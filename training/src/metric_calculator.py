from typing import Iterable

import numpy as np
import wandb
import wandb.plot
import wandb.sklearn
from fess38.data_processing.operation.consumer.base import ConsumeOpBase
from fess38.util.wandb import wandb_init
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, r2_score

from .config import MetricCalculationConsumeOpConfig
from .types import PredictionRecord


class MetricCalculationConsumeOp(ConsumeOpBase):
    def __init__(self, config: MetricCalculationConsumeOpConfig):
        super().__init__(config, self._consume_fn)
        wandb_init(config)
        self._mapping = {
            "accuracy": self._accuracy,
            "precision": self._precision,
            "recall": self._recall,
            "f1": self._f1,
            "r2": self._r2,
            "precision_recall_curve": self._precision_recall_curve,
            "roc_curve": self._roc_curve,
            "confusion_matrix": self._confusion_matrix,
        }

    def _consume_fn(self, records: Iterable[PredictionRecord], role: str | None):
        records = list(records)
        for metric_name, metric_config in self.config.metric_configs.items():
            self._mapping[metric_name](metric_config, records)

    def _accuracy(self, metric_config: dict, records: list[PredictionRecord]):
        threshold = metric_config.get("threshold", 0.5)
        accuracy = accuracy_score(
            y_true=[record.labels[0] for record in records],
            y_pred=[int(record.predictions[1] >= threshold) for record in records],
            sample_weight=[record.sample_weight or 1 for record in records],
        )
        wandb.summary["accuracy"] = accuracy

    def _precision_recall_f1(
        self,
        metric_config: dict,
        records: list[PredictionRecord],
    ):
        if wandb.summary.get("precision") is not None:
            return

        threshold = metric_config.get("threshold", 0.5)
        precision, recall, fbeta_score, _ = precision_recall_fscore_support(
            y_true=[record.labels[0] for record in records],
            y_pred=[int(record.predictions[1] >= threshold) for record in records],
            labels=metric_config.get("labels"),
            pos_label=metric_config.get("pos_label", 1),
            average=metric_config.get("average", "binary"),
            sample_weight=[record.sample_weight or 1 for record in records],
            zero_division=metric_config.get("zero_division", 0),
        )

        wandb.summary["precision"] = precision
        wandb.summary["recall"] = recall
        wandb.summary["f1"] = fbeta_score

    def _precision(self, metric_config: dict, records: list[PredictionRecord]):
        self._precision_recall_f1(metric_config, records)

    def _recall(self, metric_config: dict, records: list[PredictionRecord]):
        self._precision_recall_f1(metric_config, records)

    def _f1(self, metric_config: dict, records: list[PredictionRecord]):
        self._precision_recall_f1(metric_config, records)

    def _r2(self, metric_config: dict, records: list[PredictionRecord]):
        wandb.summary["r2"] = r2_score(
            y_true=[record.labels[0] for record in records],
            y_pred=[record.predictions[0] for record in records],
            sample_weight=[record.sample_weight or 1 for record in records],
            multioutput=metric_config.get("multioutput", "uniform_average"),
        )

    def _precision_recall_curve(
        self,
        metric_config: dict,
        records: list[PredictionRecord],
    ):
        precision_recall_curve = wandb.plot.pr_curve(
            y_true=[record.labels[0] for record in records],
            y_probas=[record.predictions for record in records],
            labels=metric_config.get("labels"),
            classes_to_plot=metric_config.get("classes_to_plot"),
        )
        wandb.log({"precision_recall_curve": precision_recall_curve})

    def _roc_curve(
        self,
        metric_config: dict,
        records: list[PredictionRecord],
    ):
        roc_curve = wandb.plot.roc_curve(
            y_true=[record.labels[0] for record in records],
            y_probas=[record.predictions for record in records],
            labels=metric_config.get("labels"),
            classes_to_plot=metric_config.get("classes_to_plot"),
        )
        wandb.log({"roc_curve": roc_curve})

    def _confusion_matrix(
        self,
        metric_config: dict,
        records: list[PredictionRecord],
    ):
        confusion_matrix = wandb.plot.confusion_matrix(
            y_true=[record.labels[0] for record in records],
            probs=np.array([record.predictions for record in records]),
            class_names=metric_config.get("labels"),
        )
        wandb.log({"confusion_matrix": confusion_matrix})