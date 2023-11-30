import logging
from dataclasses import dataclass
from typing import Any, Iterable

logger = logging.getLogger(__name__)


@dataclass
class OutputRecord:
    value: Any
    index: int | None = None
    role: str | None = None


OutputIterable = Iterable[Any | OutputRecord]
