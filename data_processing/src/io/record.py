from dataclasses import dataclass
from typing import Iterable

from fess38.util.typing import PyTree


@dataclass
class OutputRecord:
    value: PyTree
    index: int | None = None
    role: str | None = None


OutputIterable = Iterable[PyTree | OutputRecord]
