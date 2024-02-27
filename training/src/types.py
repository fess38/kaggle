from enum import Enum
from functools import cache

from pydantic import BaseModel


class FeatureNameBase(str, Enum):
    @cache
    def index(self) -> int:
        return list(type(self)).index(self)


class SampleRecord(BaseModel, extra="forbid"):
    id: int | str | bytes
    labels: list[int | float] | None = None
    sample_weight: float | None = None
    num_features: list[float] | None = None
    cat_features: list[str] | None = None


class PredictionRecord(BaseModel, extra="forbid"):
    id: int | str | bytes
    labels: list[int | float] | None = None
    sample_weight: float | None = None
    predictions: list[float]
