from typing import Any

from ..base import OpBase
from ..config import MapOpConfigBase
from ..protocol import MapAggregatorFn, MapFn


class MapOpBase(OpBase):
    def __init__(self, config: MapOpConfigBase, map_fn: MapFn | MapAggregatorFn):
        if len(config.inputs) == 0:
            raise ValueError("Map op should have inputs")

        if len(config.outputs) == 0:
            raise ValueError("Map op should have outputs")

        super().__init__(config)
        self._map_fn = map_fn

    def run(self):
        self._backend.run_map(
            config=self.config, map_fn=self._map_fn, **self._map_kwargs()
        )

    def _map_kwargs(self) -> dict[str, Any]:
        return {}
