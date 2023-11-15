import abc
import logging
from typing import Sequence

from ..operations.config import (
    ConsumeOpConfigBase,
    MapOpConfigBase,
    MapReduceOpConfigBase,
    ProduceOpConfigBase,
)
from ..operations.protocol import (
    ConsumeFn,
    MapFn,
    MapReduceMapFn,
    MapReduceReduceFn,
    ProduceFn,
)

logger = logging.getLogger(__name__)


class BackendBase(abc.ABC):
    @abc.abstractmethod
    def run_consume(
        self,
        config: ConsumeOpConfigBase,
        consume_fn: ConsumeFn,
    ):
        ...

    @abc.abstractmethod
    def run_produce(
        self,
        config: ProduceOpConfigBase,
        produce_fns: Sequence[ProduceFn],
    ):
        ...

    @abc.abstractmethod
    def run_map(self, config: MapOpConfigBase, map_fn: MapFn):
        ...

    @abc.abstractmethod
    def run_map_reduce(
        self,
        config: MapReduceOpConfigBase,
        map_fn: MapReduceMapFn,
        reduce_fn: MapReduceReduceFn,
    ):
        ...
