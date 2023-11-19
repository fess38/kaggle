import random
import uuid
from typing import Iterable

from fess38.data_processing.operations.producer.base import ProduceOpBase

from .config import RandomProduceOpConfig


class RandomProduceOp(ProduceOpBase):
    def __init__(self, config: RandomProduceOpConfig):
        super().__init__(config, [self._produce_fn])
        self._type_to_generate_fn = {
            int: lambda: random.randint(0, 2**63 - 1),
            float: lambda: random.random(),
            str: lambda: str(uuid.uuid4()),
        }
        random.seed(config.seed)

    def _produce_fn(self) -> Iterable[dict]:
        col_id_to_type = dict(
            (col_id, random.choice([int, float, str]))
            for col_id in range(self._config.col_count)
        )

        for row_id in range(self._config.row_count):
            record = {"id": row_id}
            for col_id in range(self._config.col_count):
                col_type = col_id_to_type[col_id]
                generate_fn = self._type_to_generate_fn[col_type]
                record[f"column_{col_id}"] = generate_fn()

            yield record
