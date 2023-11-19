import abc
import logging
from typing import Sequence

from ..operation.config import (
    ConsumeOpConfigBase,
    MapOpConfigBase,
    MapReduceOpConfigBase,
    ProduceOpConfigBase,
    RunOpConfigBase,
)
from ..operation.protocol import (
    ConsumeFn,
    MapFn,
    MapReduceMapFn,
    MapReduceReduceFn,
    ProduceFn,
    RunFn,
)

logger = logging.getLogger(__name__)


class BackendBase(abc.ABC):
    @abc.abstractmethod
    def run(
        self,
        config: RunOpConfigBase,
        run_fn: RunFn,
    ):
        ...

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
