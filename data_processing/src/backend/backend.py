from .base import BackendBase
from .config import BackendConfig, LocalBackendConfig
from .local_backend import LocalBackend


def create_backend(config: BackendConfig) -> BackendBase:
    if isinstance(config, LocalBackendConfig):
        return LocalBackend(config)
    else:
        raise ValueError(f"Unknown backend config type: {type(config).__name__}")
