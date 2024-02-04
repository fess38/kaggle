import inspect
from typing import Any, Callable, Union

import numpy as np
from faker import Faker
from faker.providers import BaseProvider
from fess38.util.typing import PyTree

_RecordNodeTemplate = tuple[Callable, dict[str, Any]]
_RecordTemplate = dict[str, Union[_RecordNodeTemplate, "_RecordTemplate"]]


class RecordProvider(BaseProvider):
    def record(self, template: _RecordTemplate, class_: type | None = None) -> PyTree:
        record = dict(
            (
                key,
                (
                    self.record(value)
                    if isinstance(value, dict)
                    else (value[0](**value[1]) if inspect.ismethod(value[0]) else value)
                ),
            )
            for key, value in template.items()
        )
        if class_ is not None:
            record = class_(**record)

        return record

    def records(
        self, amount: int, template: _RecordTemplate, class_: type | None = None
    ) -> list[PyTree]:
        return [self.record(template, class_) for _ in range(amount)]

    def rand(self, size: int | list[int]) -> np.ndarray:
        if isinstance(size, int):
            size = [size]

        return np.random.rand(*size)


def create_faker(seed: int = 0) -> Faker:
    np.random.seed(seed)
    Faker.seed(seed)

    fake = Faker()
    fake.add_provider(RecordProvider)

    return fake
