import functools
from typing import Iterable

from fess38.data_processing.operation.consumer.base import ConsumeOpBase
from fess38.util.wandb import wandb_init

from ..config import MetricCalculationConsumeOpConfig
from ..types import PredictionRecord
from .metric_library import metric_library


class MetricCalculationConsumeOp(ConsumeOpBase):
    def __init__(self, config: MetricCalculationConsumeOpConfig):
        super().__init__(config, self._consume_fn)
        wandb_init(config)

        self._metric_fns = []
        for metric_config in config.metric_configs:
            metric_fn = metric_library[metric_config.type]
            metric_kwargs = metric_config.model_dump(exclude={"vars", "type"})
            self._metric_fns.append(functools.partial(metric_fn, **metric_kwargs))

    def _consume_fn(self, records: Iterable[PredictionRecord], role: str | None):
        records = list(records)
        for metric_fn in self._metric_fns:
            metric_fn(records)
