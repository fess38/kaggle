from .base import TransformBase
from .config import MapTransformConfigBase
from .protocol import MapFn


class MapTransformBase(TransformBase):
    def __init__(self, config: MapTransformConfigBase, map_fn: MapFn):
        super().__init__(config)
        self._map_fn = map_fn

    def run(self):
        self._backend.run_map(self.config, self._map_fn)
