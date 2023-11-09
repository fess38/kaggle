from collections.abc import Sequence

from ..base import OpBase
from ..config import CreateOpConfigBase
from ..protocol import CreateFn


class CreateOpBase(OpBase):
    def __init__(
        self,
        config: CreateOpConfigBase,
        create_fns: Sequence[CreateFn],
    ):
        if len(config.inputs) != 0:
            raise ValueError("Create op should have no inputs.")

        super().__init__(config)
        self._create_fns = create_fns

    def run(self):
        self._backend.run_create(self.config, self._create_fns)
