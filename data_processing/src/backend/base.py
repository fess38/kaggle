import abc
import logging
from typing import Sequence

from ..operations.config import (
    CreateOpConfigBase,
    MapOpConfigBase,
    MapReduceOpConfigBase,
)
from ..operations.protocol import CreateFn, MapFn, MapReduceMapFn, MapReduceReduceFn

logger = logging.getLogger(__name__)


class BackendBase(abc.ABC):
    @abc.abstractmethod
    def run_create(
        self,
        config: CreateOpConfigBase,
        create_fns: Sequence[CreateFn],
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
