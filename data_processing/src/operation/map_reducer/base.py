from ..base import OpBase
from ..config import MapReduceOpConfigBase
from ..protocol import (
    MapReduceMapAggregatorFn,
    MapReduceMapFn,
    MapReduceReduceAggregatorFn,
    MapReduceReduceFn,
)


class MapReduceOpBase(OpBase):
    def __init__(
        self,
        config: MapReduceOpConfigBase,
        map_fn: MapReduceMapFn | MapReduceMapAggregatorFn,
        reduce_fn: MapReduceReduceFn | MapReduceReduceAggregatorFn,
    ):
        if len(config.inputs) == 0:
            raise ValueError("MapReduce op should have inputs")

        if len(config.outputs) == 0:
            raise ValueError("MapReduce op should have outputs")

        super().__init__(config)
        self._map_fn = map_fn
        self._reduce_fn = reduce_fn

    def run(self):
        self._backend.run_map_reduce(
            config=self.config,
            map_fn=self._map_fn,
            reduce_fn=self._reduce_fn,
            instruction_configs=self.backend_instruction_configs(),
        )
