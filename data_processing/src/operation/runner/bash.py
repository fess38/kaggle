import subprocess

from .base import RunOpBase
from .config import BashRunOpConfig


class BashRunOp(RunOpBase):
    def __init__(self, config: BashRunOpConfig):
        super().__init__(config, self._run_fn)

    def _run_fn(self):
        command = "\n".join(
            (
                ["set -ex"]
                + self._format_vars(self.config.input_files)
                + self._format_vars(self.config.output_files)
                + self._format_vars(self.config.vars)
                + self.config.commands
            )
        )
        subprocess.check_call(command, shell=True)

    def _format_vars(self, data: dict[str, str]) -> list[str]:
        return [f'{name}="{value}"' for name, value in data.items()]
