from fess38.utils.config import ConfigBase


class FilterConfigBase(ConfigBase):
    record_dropped_to_role: str | None = None
