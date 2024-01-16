import random
from typing import Any, Iterable

from fess38.data_processing.io.record import OutputIterable, OutputRecord
from fess38.data_processing.operation.mapper.base import MapOpBase
from fess38.util.hashing import combine_hashes

from .config import TrainTestSplitMapOpConfig

_TRAIN_INDEX = 0
_TEST_INDEX = 1


class TrainTestSplitMapOp(MapOpBase):
    def __init__(self, config: TrainTestSplitMapOpConfig):
        self._validate_config(config)
        super().__init__(config, self._map_fn)

    def _map_fn(self, records: Iterable[Any], role: str | None) -> OutputIterable:
        for record in records:
            index = _TRAIN_INDEX

            seed = combine_hashes(*(map(record.get, self.config.based_on)))
            if random.Random(seed).random() < self.config.sampling_rate:
                index = _TEST_INDEX

            yield OutputRecord(record, index)

    def _validate_config(self, config: TrainTestSplitMapOpConfig):
        if not (0.0 < config.sampling_rate < 1.0):
            raise ValueError("Sampling rate should be between 0 and 1")

        if len(config.outputs) != 2:
            raise ValueError("TrainTestSplitMapOp should have 2 outputs")
