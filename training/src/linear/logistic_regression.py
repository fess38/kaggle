import abc
from typing import Iterable

import joblib
import more_itertools
from fess38.util.reflection import constructor_keys
from sklearn.linear_model import LogisticRegression, SGDClassifier

from ..base import InferenceOpBase, TrainOpBase
from ..types import PredictionRecord, SampleRecord
from .config import (
    LogisticRegressionInferenceOpConfig,
    LogisticRegressionTrainOpConfig,
    SGDClassifierTrainOpConfig,
)

_MODEL_TYPE = LogisticRegression | SGDClassifier


class LogisticRegressionTrainOpBase(TrainOpBase):
    def __init__(
        self, config: LogisticRegressionTrainOpConfig | SGDClassifierTrainOpConfig
    ):
        super().__init__(config, self._train_fn)
        self._model: _MODEL_TYPE = self.create_model()

    def _train_fn(self, records: Iterable[SampleRecord], role: str | None):
        records = list(records)
        self._model.fit(
            X=[record.num_features for record in records],
            y=[record.labels[0] for record in records],
            sample_weight=(
                [record.sample_weight for record in records]
                if records[0].sample_weight is not None
                else None
            ),
        )

        joblib.dump(self._model, self.config.output_files["model"])

    @abc.abstractmethod
    def create_model(self) -> _MODEL_TYPE:
        ...


class LogisticRegressionTrainOp(LogisticRegressionTrainOpBase):
    def __init__(self, config: LogisticRegressionTrainOpConfig):
        super().__init__(config)

    def create_model(self) -> LogisticRegression:
        params = self._config.model_dump(
            include=constructor_keys(LogisticRegression), exclude_unset=True
        )
        return LogisticRegression(**params)


class SGDClassifierTrainOp(LogisticRegressionTrainOpBase):
    def __init__(self, config: SGDClassifierTrainOpConfig):
        super().__init__(config)

    def create_model(self) -> SGDClassifier:
        params = self._config.model_dump(
            include=constructor_keys(SGDClassifier), exclude_unset=True
        )
        return SGDClassifier(**params)


class LogisticRegressionInferenceOp(InferenceOpBase):
    def __init__(self, config: LogisticRegressionInferenceOpConfig):
        super().__init__(config, self._map_fn)

    def _map_fn(
        self, records: Iterable[SampleRecord], role: str | None
    ) -> Iterable[PredictionRecord]:
        model: _MODEL_TYPE = joblib.load(self.config.input_files["model"])

        for batch in more_itertools.batched(records, self.config.batch_size):
            features = [record.num_features for record in batch]
            predictions = model.predict_proba(features).tolist()

            for record, predictions in zip(batch, predictions):
                yield PredictionRecord(
                    id=record.id,
                    labels=record.labels,
                    predictions=predictions,
                )
