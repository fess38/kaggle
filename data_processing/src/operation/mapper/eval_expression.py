from typing import Any, Iterable

from .base import MapOpBase
from .config import EvalExpressionMapOpConfig


class EvalExpressionMapOp(MapOpBase):
    def __init__(self, config: EvalExpressionMapOpConfig):
        super().__init__(config, self._map_fn)

    def _map_fn(self, records: Iterable[Any], role: str | None) -> Iterable[Any]:
        eval_fn = eval(self.config.expression)
        for record in records:
            yield eval_fn(record)
