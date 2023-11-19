from typing import Literal

from fess38.data_processing.operation import operation_library
from fess38.data_processing.operation.config import ProduceOpConfigBase


@operation_library("fess38.competitions.examples.produce_random.RandomProduceOp")
class RandomProduceOpConfig(ProduceOpConfigBase):
    type: Literal["random_produce"] = "random_produce"
    seed: int = 42
    row_count: int
    col_count: int
