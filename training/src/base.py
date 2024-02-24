from typing import Any

from fess38.data_processing.backend.instruction.config import SetRecordClassInstruction
from fess38.data_processing.operation.consumer.base import ConsumeOpBase
from fess38.data_processing.operation.mapper.base import MapOpBase
from fess38.data_processing.operation.protocol import ConsumeAggregatorFn
from fess38.util.reflection import full_path

from .config import TrainOpConfigBase
from .types import PredictionRecord, SampleRecord


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
