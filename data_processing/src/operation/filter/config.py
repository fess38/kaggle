from typing import Annotated

import pydantic

from .drop_empty import DropEmptyFilterConfig

FilterConfig = Annotated[
    (DropEmptyFilterConfig),
    pydantic.Field(discriminator="type"),
]
