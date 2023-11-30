import random
from typing import Any, Iterable

from fess38.data_processing.io.record import OutputIterable, OutputRecord
from fess38.data_processing.operation.mapper.base import MapOpBase

from .config import TrainTestSplitMapOpConfig

_TRAIN_INDEX = 0
_TEST_INDEX = 1


class TrainTestSplitMapOp(MapOpBase):
    def __init__(self, config: TrainTestSplitMapOpConfig):
        self._validate_config(config)
        super().__init__(config, self._map_fn)

    def _map_fn(self, records: Iterable[Any], role: str | None) -> OutputIterable:
        random.seed(self.config.random_state)
        counter = 0

        for record in records:
            counter += 1
            index = _TRAIN_INDEX
            if self.config.test_size and counter <= self.config.test_size:
                index = _TEST_INDEX
            elif (
                self.config.test_proportion
                and random.random() < self.config.test_proportion
            ):
                index = _TEST_INDEX

            yield OutputRecord(record, index)

    def _validate_config(self, config: TrainTestSplitMapOpConfig):
        if config.test_size is None and config.test_proportion is None:
            raise ValueError("Provide test_size or test_proportion")

        if config.test_size is not None and config.test_proportion is not None:
            raise ValueError(
                "Only one of test_size and test_proportion should be provided"
            )

        if config.test_size is not None and config.test_size < 0:
            raise ValueError(f"config.test_size == {config.test_size}")

        if config.test_proportion is not None:
            if config.test_proportion < 0:
                raise ValueError(f"config.test_proportion == {config.test_proportion}")
            if config.test_proportion > 1:
                raise ValueError(f"config.test_proportion == {config.test_proportion}")
