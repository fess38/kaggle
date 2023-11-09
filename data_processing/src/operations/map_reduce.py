from .base import OpBase
from .config import MapReduceOpConfigBase
from .protocol import MapReduceMapFn, MapReduceReduceFn


class MapReduceOpBase(OpBase):
    def __init__(
        self,
        config: MapReduceOpConfigBase,
        map_fn: MapReduceMapFn,
        reduce_fn: MapReduceReduceFn,
    ):
        super().__init__(config)
        self._map_fn = map_fn
        self._reduce_fn = reduce_fn

    def run(self):
        self._backend.run_map_reduce(self.config, self._map_fn, self._reduce_fn)
