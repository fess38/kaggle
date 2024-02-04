from typing import Iterable

import numpy as np
import wandb
import wandb.plot
import wandb.sklearn
from fess38.data_processing.operation.consumer.base import ConsumeOpBase
from fess38.util.wandb import wandb_init
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    r2_score,
    roc_auc_score,
    roc_curve,
)

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
            "roc_auc": self._roc_auc_score,
            "r2": self._r2,
            "max_accuracy_threshold": self._max_accuracy_threshold,
            "precision_recall_curve": self._precision_recall_curve,
            "roc_curve": self._roc_curve,
            "confusion_matrix": self._confusion_matrix,
        }

    def _consume_fn(self, records: Iterable[PredictionRecord], role: str | None):
        records = list(records)
        for metric_name, metric_config in self.config.metric_configs.items():
            self._mapping[metric_name](metric_config, records)

    def _accuracy(self, metric_config: dict, records: list[PredictionRecord]):
        threshold = metric_config["threshold"]
        accuracy = accuracy_score(
            y_true=[record.labels[0] for record in records],
            y_pred=[int(record.predictions[1] >= threshold) for record in records],
            sample_weight=[record.sample_weight or 1 for record in records],
            normalize=metric_config.get("normalize", True),
        )
        wandb.summary["accuracy"] = accuracy

    def _precision(self, metric_config: dict, records: list[PredictionRecord]):
        self._precision_recall_f1(metric_config, records)

    def _recall(self, metric_config: dict, records: list[PredictionRecord]):
        self._precision_recall_f1(metric_config, records)

    def _f1(self, metric_config: dict, records: list[PredictionRecord]):
        self._precision_recall_f1(metric_config, records)

    def _precision_recall_f1(
        self, metric_config: dict, records: list[PredictionRecord]
    ):
        if wandb.summary.get("precision") is not None:
            return

        threshold = metric_config["threshold"]
        precision, recall, fbeta_score, _ = precision_recall_fscore_support(
            y_true=[record.labels[0] for record in records],
            y_pred=[int(record.predictions[1] >= threshold) for record in records],
            sample_weight=[record.sample_weight or 1 for record in records],
            labels=metric_config.get("labels"),
            pos_label=metric_config.get("pos_label", 1),
            average=metric_config.get("average", "binary"),
            zero_division=metric_config.get("zero_division", 0),
        )

        wandb.summary["precision"] = precision
        wandb.summary["recall"] = recall
        wandb.summary["f1"] = fbeta_score

    def _roc_auc_score(self, metric_config, records: list[PredictionRecord]):
        roc_auc = roc_auc_score(
            y_true=[record.labels[0] for record in records],
            y_score=[record.predictions[1] for record in records],
            sample_weight=[record.sample_weight or 1 for record in records],
            **metric_config,
        )
        wandb.summary["roc_auc"] = roc_auc

    def _r2(self, metric_config: dict, records: list[PredictionRecord]):
        wandb.summary["r2"] = r2_score(
            y_true=[record.labels[0] for record in records],
            y_pred=[record.predictions[0] for record in records],
            sample_weight=[record.sample_weight or 1 for record in records],
            multioutput=metric_config.get("multioutput", "uniform_average"),
        )

    def _max_accuracy_threshold(
        self, metric_config: dict, records: list[PredictionRecord]
    ):
        y_true = [record.labels[0] for record in records]
        y_score = [record.predictions[1] for record in records]
        sample_weight = [record.sample_weight or 1 for record in records]
        _, _, thresholds = roc_curve(
            y_true=y_true,
            y_score=y_score,
            sample_weight=sample_weight,
        )
        thresholds = np.nan_to_num(thresholds, posinf=0, neginf=0)

        max_accuracy_index = np.argmax(
            [
                accuracy_score(
                    y_true=y_true,
                    y_pred=(y_score >= threshold).astype(int),
                    sample_weight=sample_weight,
                )
                for threshold in thresholds
            ]
        )
        wandb.summary["max_accuracy_threshold"] = thresholds[max_accuracy_index]

    def _precision_recall_curve(
        self, metric_config: dict, records: list[PredictionRecord]
    ):
        precision_recall_curve = wandb.plot.pr_curve(
            y_true=[record.labels[0] for record in records],
            y_probas=[record.predictions for record in records],
            labels=metric_config.get("labels"),
            classes_to_plot=metric_config.get("classes_to_plot"),
        )
        wandb.log({"precision_recall_curve": precision_recall_curve})

    def _roc_curve(self, metric_config: dict, records: list[PredictionRecord]):
        roc_curve = wandb.plot.roc_curve(
            y_true=[record.labels[0] for record in records],
            y_probas=[record.predictions for record in records],
            labels=metric_config.get("labels"),
            classes_to_plot=metric_config.get("classes_to_plot"),
        )
        wandb.log({"roc_curve": roc_curve})

    def _confusion_matrix(self, metric_config: dict, records: list[PredictionRecord]):
        confusion_matrix = wandb.plot.confusion_matrix(
            y_true=[record.labels[0] for record in records],
            probs=np.array([record.predictions for record in records]),
            class_names=metric_config.get("labels"),
        )
        wandb.log({"confusion_matrix": confusion_matrix})
