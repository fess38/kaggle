import abc
import logging
from typing import Sequence

from ..operation.protocol import (
    ConsumeFn,
    MapFn,
    MapReduceMapFn,
    MapReduceReduceFn,
    ProduceFn,
    RunFn,
)
from .config import (
    ConsumeBackendOpConfig,
    MapBackendOpConfig,
    MapReduceBackendOpConfig,
    ProduceBackendOpConfig,
    RunBackendOpConfig,
)
from .instruction.config import BackendInstructionConfig

logger = logging.getLogger(__name__)


class BackendBase(abc.ABC):
    @abc.abstractmethod
    def run(
        self,
        config: RunBackendOpConfig,
        run_fn: RunFn,
        instruction_configs: list[BackendInstructionConfig],
    ):
        ...

    @abc.abstractmethod
    def run_consume(
        self,
        config: ConsumeBackendOpConfig,
        consume_fn: ConsumeFn,
        instruction_configs: list[BackendInstructionConfig],
    ):
        ...

    @abc.abstractmethod
    def run_produce(
        self,
        config: ProduceBackendOpConfig,
        produce_fns: Sequence[ProduceFn],
        instruction_configs: list[BackendInstructionConfig],
    ):
        ...

    @abc.abstractmethod
    def run_map(
        self,
        config: MapBackendOpConfig,
        map_fn: MapFn,
        instruction_configs: list[BackendInstructionConfig],
    ):
        ...

    @abc.abstractmethod
    def run_map_reduce(
        self,
        config: MapReduceBackendOpConfig,
        map_fn: MapReduceMapFn,
        reduce_fn: MapReduceReduceFn,
        instruction_configs: list[BackendInstructionConfig],
    ):
        ...
