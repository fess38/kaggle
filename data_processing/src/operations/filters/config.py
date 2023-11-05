from typing import Annotated

import pydantic

from .drop_domains import DropDomainsFilterConfig
from .drop_empty import DropEmptyFilterConfig

FilterConfig = Annotated[
    (DropDomainsFilterConfig | DropEmptyFilterConfig),
    pydantic.Field(discriminator="type"),
]
