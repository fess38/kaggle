import subprocess

from ..protocol import RunFn
from .base import RunOpBase
from .config import BashRunOpConfig


class BashRunOp(RunOpBase):
    def __init__(self, config: BashRunOpConfig):
        super().__init__(config, self._create_run_fn(config))

    def _create_run_fn(self, config: BashRunOpConfig) -> RunFn:
        def run_fn():
            command = "\n".join(
                (
                    ["set -ex"]
                    + self._format_vars(config.input_files)
                    + self._format_vars(config.output_files)
                    + self._format_vars(config.vars)
                    + self.config.commands
                )
            )
            subprocess.check_call(command, shell=True)

        return run_fn

    def _format_vars(self, data: dict[str, str]) -> list[str]:
        return [f'{name}="{value}"' for name, value in data.items()]
