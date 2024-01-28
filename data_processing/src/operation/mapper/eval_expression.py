from typing import Iterable

from fess38.util.typing import PyTree

from .base import MapOpBase
from .config import EvalExpressionMapOpConfig


class EvalExpressionMapOp(MapOpBase):
    def __init__(self, config: EvalExpressionMapOpConfig):
        super().__init__(config, self._map_fn)

    def _map_fn(self, records: Iterable[PyTree], role: str | None) -> Iterable[PyTree]:
        eval_fn = eval(self.config.expression)
        for record in records:
            yield eval_fn(record)
