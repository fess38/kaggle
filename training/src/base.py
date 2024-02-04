from fess38.data_processing.operation.consumer.base import ConsumeOpBase
from fess38.data_processing.operation.protocol import ConsumeAggregatorFn

from .config import TrainOpConfigBase


class TrainOpBase(ConsumeOpBase):
    def __init__(
        self,
        config: TrainOpConfigBase,
        consume_fn: ConsumeAggregatorFn,
    ):
        super().__init__(config, consume_fn)
