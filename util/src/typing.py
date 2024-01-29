from pathlib import Path

from pydantic import BaseModel

PyTree = dict | BaseModel
PathLike = str | Path
PyTreePath = str | tuple[str, ...]
