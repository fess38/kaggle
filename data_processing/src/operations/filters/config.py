from typing import Annotated

import pydantic

from .drop_domains import DropDomainsFilterConfig
from .drop_empty import DropEmptyFilterConfig
from .drop_if_regexp_match import DropIfRegexpMatchFilterConfig

FilterConfig = Annotated[
    (DropDomainsFilterConfig | DropEmptyFilterConfig | DropIfRegexpMatchFilterConfig),
    pydantic.Field(discriminator="type"),
]
