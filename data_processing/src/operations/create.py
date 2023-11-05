from collections.abc import Sequence

from .base import TransformBase
from .config import CreateTransformConfigBase
from .protocol import CreateFn


class CreateTransformBase(TransformBase):
    def __init__(
        self,
        config: CreateTransformConfigBase,
        create_fns: Sequence[CreateFn],
    ):
        if len(config.inputs) != 0:
            raise ValueError("Create transform should have no inputs.")

        super().__init__(config)
        self._create_fns = create_fns

    def run(self):
        self._backend.run_create(self.config, self._create_fns)
