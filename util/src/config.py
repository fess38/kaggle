from typing import Generic, TypeVar

import pydantic
from omegaconf import OmegaConf

from .typing import PathLike

T = TypeVar("T", bound="ConfigBase")


class ConfigBase(pydantic.BaseModel, Generic[T]):
    class Config:
        frozen: bool = True
        extra = "forbid"

    vars: dict | None = pydantic.Field(default=None, repr=False)

    @classmethod
    def from_file(cls: type[T], path: PathLike) -> T:
        return cls(**OmegaConf.to_object(OmegaConf.load(path)))
