from typing import Any

from fess38.data_processing.backend.instruction.config import SetRecordClassInstruction
from fess38.data_processing.operation.consumer.base import ConsumeOpBase
from fess38.data_processing.operation.mapper.base import MapOpBase
from fess38.data_processing.operation.protocol import ConsumeAggregatorFn
from fess38.util.reflection import full_path

from .config import TrainOpConfigBase
from .types import FeatureNameBase, PredictionRecord, SampleRecord


class FeatureCalculatorOpBase(MapOpBase):
    def _filter_features(self, record: SampleRecord):
        if self._num_feature_names() is not None:
            record.num_features = [
                record.num_features[feature_name.index()]
                for feature_name in self._num_feature_names()
            ]

        if self._cat_feature_names() is not None:
            record.cat_features = [
                record.cat_features[feature_name.index()]
                for feature_name in self._cat_feature_names()
            ]

    def _num_feature_names(self) -> list[FeatureNameBase] | None:
        return None

    def _cat_feature_names(self) -> list[FeatureNameBase] | None:
        return None


class TrainOpBase(ConsumeOpBase):
    def __init__(
        self,
        config: TrainOpConfigBase,
        consume_fn: ConsumeAggregatorFn,
    ):
        super().__init__(config, consume_fn)

    def _consume_kwargs(self) -> dict[str, Any]:
        return {
            "instructions": [
                SetRecordClassInstruction(
                    io="inputs", record_class=full_path(SampleRecord)
                )
            ]
        }


class InferenceOpBase(MapOpBase):
    def _map_kwargs(self) -> dict[str, Any]:
        return {
            "instructions": [
                SetRecordClassInstruction(
                    io="inputs", record_class=full_path(SampleRecord)
                ),
                SetRecordClassInstruction(
                    io="outputs", record_class=full_path(PredictionRecord)
                ),
            ]
        }
