import abc

from ..backend.backend import create_backend
from ..backend.config import BackendConfig
from .config import TransformConfigBase


class TransformBase(abc.ABC):
    def __init__(self, config: TransformConfigBase):
        self._config = config
        self._backend = create_backend(self.config.backend)

    @property
    def config(self) -> TransformConfigBase:
        return self._config

    @property
    def backend(self) -> BackendConfig:
        return self._backend

    @abc.abstractmethod
    def run(self):
        ...
