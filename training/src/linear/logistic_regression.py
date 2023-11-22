from typing import Iterable

import joblib
from fess38.data_processing.operation.mapper.base import MapOpBase
from sklearn.linear_model import LogisticRegression

from ..base import TrainOpBase
from ..types import PredictionRecord, SampleRecord
from .config import LogisticRegressionInferenceOpConfig, LogisticRegressionTrainOpConfig


class LogisticRegressionTrainOp(TrainOpBase):
    def __init__(self, config: LogisticRegressionTrainOpConfig):
        super().__init__(config, self._train_fn)

    def _train_fn(self, records: Iterable[SampleRecord], role: str | None):
        model = LogisticRegression(
            random_state=self._config.random_state,
            **self._config.kwargs,
        )

        records = list(records)
        model.fit(
            X=[record.num_features for record in records],
            y=[record.labels[0] for record in records],
            sample_weight=(
                [record.sample_weight for record in records]
                if records[0].sample_weight is not None
                else None
            ),
        )

        joblib.dump(model, self.config.output_files["model.bin"])


class LogisticRegressionInferenceOp(MapOpBase):
    def __init__(self, config: LogisticRegressionInferenceOpConfig):
        super().__init__(config, self._map_fn)

    def _map_fn(
        self,
        records: Iterable[SampleRecord],
        role: str | None,
    ) -> Iterable[PredictionRecord]:
        model: LogisticRegression = joblib.load(self.config.input_files["model.bin"])
        records = list(records)
        for record in records:
            predictions = model.predict_proba([record.num_features]).tolist()[0]
            yield PredictionRecord(
                id=record.id,
                labels=record.labels,
                predictions=predictions,
            )
