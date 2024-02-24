from typing import Any

from ..base import OpBase
from ..config import ConsumeOpConfigBase
from ..protocol import ConsumeAggregatorFn, ConsumeFn


class ConsumeOpBase(OpBase):
    def __init__(
        self, config: ConsumeOpConfigBase, consume_fn: ConsumeFn | ConsumeAggregatorFn
    ):
        if len(config.inputs) == 0:
            raise ValueError("Consume op should have inputs")

        if len(config.outputs) != 0:
            raise ValueError("Consume op should have no outputs")

        super().__init__(config)
        self._consume_fn = consume_fn

    def run(self):
        self._backend.run_consume(
            config=self.config, consume_fn=self._consume_fn, **self._consume_kwargs()
        )

    def _consume_kwargs(self) -> dict[str, Any]:
        return {}
