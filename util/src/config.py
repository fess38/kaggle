import itertools
import os
from typing import Generic, TypeVar

import pydantic
from omegaconf import OmegaConf

from .typing import PathLike

T = TypeVar("T", bound="ConfigBase")


def set_record_formatter_by_extension(config: dict):
    for op in config.get("ops", []):
        for io in itertools.chain(op.get("inputs", []), op.get("outputs", [])):
            extension = os.path.splitext(io["path"])[1].replace(".", "")
            io.setdefault("record_formatter", {"type": extension})


class ConfigBase(pydantic.BaseModel, Generic[T]):
    class Config:
        frozen: bool = True
        extra = "forbid"
        protected_namespaces = ()

    vars: dict | None = pydantic.Field(default=None, repr=False)

    @classmethod
    def from_file(cls: type[T], path: PathLike) -> T:
        config = OmegaConf.to_object(OmegaConf.load(path))
        set_record_formatter_by_extension(config)
        return cls(**config)
