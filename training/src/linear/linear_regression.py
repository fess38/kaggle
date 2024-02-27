import abc
from typing import Iterable

import joblib
import more_itertools
from sklearn.linear_model import LinearRegression, SGDRegressor

from ..base import InferenceOpBase, TrainOpBase
from ..types import PredictionRecord, SampleRecord
from .config import (
    LinearRegressionInferenceOpConfig,
    LinearRegressionTrainOpConfig,
    SGDRegressorTrainOpConfig,
)

_MODEL_TYPE = LinearRegression | SGDRegressor


class LinearRegressionTrainOpBase(TrainOpBase):
    def __init__(
        self, config: LinearRegressionTrainOpConfig | SGDRegressorTrainOpConfig
    ):
        super().__init__(config, self._consume_fn)
        self._model: _MODEL_TYPE = self.create_model()

    def _consume_fn(self, records: Iterable[SampleRecord], role: str | None):
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


class LinearRegressionTrainOp(LinearRegressionTrainOpBase):
    def __init__(self, config: LinearRegressionTrainOpConfig):
        super().__init__(config)

    def create_model(self) -> LinearRegression:
        return LinearRegression(**self.config.kwargs)


class SGDRegressorTrainOp(LinearRegressionTrainOpBase):
    def __init__(self, config: SGDRegressorTrainOpConfig):
        super().__init__(config)

    def create_model(self) -> SGDRegressor:
        return SGDRegressor(**self.config.kwargs)


class LinearRegressionInferenceOp(InferenceOpBase):
    def __init__(self, config: LinearRegressionInferenceOpConfig):
        super().__init__(config, self._map_fn)

    def _map_fn(
        self,
        records: Iterable[SampleRecord],
        role: str | None,
    ) -> Iterable[PredictionRecord]:
        model: _MODEL_TYPE = joblib.load(self.config.input_files["model"])

        for batch in more_itertools.batched(records, self.config.batch_size):
            features = [record.num_features for record in batch]
            predictions = model.predict(features).tolist()

            for record, prediction in zip(batch, predictions):
                yield PredictionRecord(
                    id=record.id,
                    labels=record.labels,
                    predictions=[prediction],
                )
