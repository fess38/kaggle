from collections.abc import Sequence
from typing import Any

from ..base import OpBase
from ..config import ProduceOpConfigBase
from ..protocol import ProduceFn


class ProduceOpBase(OpBase):
    def __init__(self, config: ProduceOpConfigBase, produce_fns: Sequence[ProduceFn]):
        if len(config.inputs) != 0:
            raise ValueError("Produce op should have no inputs")

        if len(config.outputs) == 0:
            raise ValueError("Produce op should have outputs")

        super().__init__(config)
        self._produce_fns = produce_fns

    def run(self):
        self._backend.run_produce(
            config=self.config, produce_fns=self._produce_fns, **self._produce_kwargs()
        )

    def _produce_kwargs(self) -> dict[str, Any]:
        return {}
