from fess38.util.config import ConfigBase


class FilterConfigBase(ConfigBase):
    record_dropped_to_role: str | None = None
