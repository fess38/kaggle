import abc

from ..backend.backend import create_backend
from ..backend.config import BackendConfig
from ..backend.instruction.config import BackendInstructionConfig
from .config import OpConfigBase


class OpBase(abc.ABC):
    def __init__(self, config: OpConfigBase):
        self._config = config
        self._backend = create_backend(self.config.backend)

    @property
    def config(self) -> OpConfigBase:
        return self._config

    @property
    def backend(self) -> BackendConfig:
        return self._backend

    @abc.abstractmethod
    def run(self):
        ...

    def backend_instruction_configs(self) -> list[BackendInstructionConfig]:
        return []
