from typing import Callable, Iterable

import wandb
from fess38.data_processing.operation.consumer.base import ConsumeOpBase
from fess38.util.wandb import wandb_init
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

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
        }

    def _consume_fn(self, records: Iterable[PredictionRecord], role: str | None):
        records = list(records)
        for metric_name, metric_config in self.config.metric_configs.items():
            self._mapping[metric_name](metric_name, metric_config, records)

    def _accuracy(
        self,
        metric_name: str,
        metric_config: dict,
        records: list[PredictionRecord],
    ):
        threshold = metric_config.get("threshold", 0.5)
        accuracy = accuracy_score(
            y_true=[record.labels[0] for record in records],
            y_pred=[int(record.predictions[1] >= threshold) for record in records],
            sample_weight=[record.sample_weight or 1 for record in records],
        )
        wandb.summary[metric_name] = accuracy

    def _precision_recall_f1(
        self,
        metric_name: str,
        metric_config: dict,
        records: list[PredictionRecord],
        fn: Callable,
    ):
        threshold = metric_config.get("threshold", 0.5)
        wandb.summary[metric_name] = fn(
            y_true=[record.labels[0] for record in records],
            y_pred=[int(record.predictions[1] >= threshold) for record in records],
            average=metric_config.get("average", "binary"),
            sample_weight=[record.sample_weight or 1 for record in records],
            zero_division=metric_config.get("zero_division", 0),
        )

    def _precision(
        self,
        metric_name: str,
        metric_config: dict,
        records: list[PredictionRecord],
    ):
        self._precision_recall_f1(metric_name, metric_config, records, precision_score)

    def _recall(
        self,
        metric_name: str,
        metric_config: dict,
        records: list[PredictionRecord],
    ):
        self._precision_recall_f1(metric_name, metric_config, records, recall_score)

    def _f1(
        self,
        metric_name: str,
        metric_config: dict,
        records: list[PredictionRecord],
    ):
        self._precision_recall_f1(metric_name, metric_config, records, f1_score)
