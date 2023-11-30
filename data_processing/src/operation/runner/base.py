from ..base import OpBase
from ..config import RunOpConfigBase
from ..protocol import RunFn


class RunOpBase(OpBase):
    def __init__(
        self,
        config: RunOpConfigBase,
        run_fn: RunFn,
    ):
        if len(config.inputs) != 0:
            raise ValueError("Run op should have no inputs.")

        if len(config.outputs) != 0:
            raise ValueError("Run op should have no outputs.")

        super().__init__(config)
        self._run_fn = run_fn

    def run(self):
        self._backend.run(self.config, self._run_fn)
