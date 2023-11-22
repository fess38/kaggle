from typing import Any

from fess38.data_processing.operation.config import ConsumeOpConfigBase


class TrainOpConfigBase(ConsumeOpConfigBase):
    random_state: int = 42
    kwargs: dict[str, Any] = {}
