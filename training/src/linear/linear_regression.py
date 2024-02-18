from typing import Iterable

import joblib
import more_itertools
from fess38.data_processing.backend.instruction.config import (
    BackendInstructionConfig,
    SetInputRecordClassInstructionConfig,
    SetOutputRecordClassInstructionConfig,
)
from fess38.data_processing.operation.mapper.base import MapOpBase
from sklearn.linear_model import LinearRegression, SGDRegressor

from ..base import TrainOpBase
from ..types import PredictionRecord, SampleRecord
from .config import LinearRegressionInferenceOpConfig, LinearRegressionTrainOpConfig


class LinearRegressionTrainOp(TrainOpBase):
    def __init__(self, config: LinearRegressionTrainOpConfig):
        super().__init__(config, self._consume_fn)
        self._model_name_to_cls = {
            "LinearRegression": LinearRegression,
            "SGDRegressor": SGDRegressor,
        }

    def backend_instruction_configs(self) -> list[BackendInstructionConfig]:
        return [
            SetInputRecordClassInstructionConfig(
                record_class=f"{SampleRecord.__module__}.{SampleRecord.__name__}",
            )
        ]

    def _consume_fn(self, records: Iterable[SampleRecord], role: str | None):
        model = self._model_name_to_cls[self.config.model_name](**self.config.kwargs)

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


class LinearRegressionInferenceOp(MapOpBase):
    def __init__(self, config: LinearRegressionInferenceOpConfig):
        super().__init__(config, self._map_fn)

    def backend_instruction_configs(self) -> list[BackendInstructionConfig]:
        return [
            SetInputRecordClassInstructionConfig(
                record_class=f"{SampleRecord.__module__}.{SampleRecord.__name__}",
            ),
            SetOutputRecordClassInstructionConfig(
                record_class=(
                    f"{PredictionRecord.__module__}.{PredictionRecord.__name__}"
                ),
            ),
        ]

    def _map_fn(
        self,
        records: Iterable[SampleRecord],
        role: str | None,
    ) -> Iterable[PredictionRecord]:
        model = joblib.load(self.config.input_files["model.bin"])

        for batch in more_itertools.batched(records, self.config.batch_size):
            features = [record.num_features for record in batch]
            predictions = model.predict(features).tolist()

            for record, prediction in zip(batch, predictions):
                yield PredictionRecord(
                    id=record.id,
                    labels=record.labels,
                    predictions=[prediction],
                )
