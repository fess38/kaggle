from typing import Any, Callable, Union

import numpy as np
from faker import Faker
from faker.providers import BaseProvider
from fess38.util.typing import PyTree

_RecordNodeTemplate = tuple[Callable, dict[str, Any]]
_RecordTemplate = dict[str, Union[_RecordNodeTemplate, "_RecordTemplate"]]


class RecordProvider(BaseProvider):
    def record(self, template: _RecordTemplate) -> PyTree:
        return dict(
            (
                key,
                self.record(value) if isinstance(value, dict) else value[0](**value[1]),
            )
            for key, value in template.items()
        )

    def records(self, amount: int, template: _RecordTemplate) -> list[PyTree]:
        return [self.record(template) for _ in range(amount)]

    def rand(
        self, size: list[int], low: int = 0, high: int | None = None
    ) -> np.ndarray:
        result = None

        if high is None:
            result = np.random.rand(*size)
        else:
            result = np.random.randint(low, high, size)

        return result


def create_faker(seed: int = 0) -> Faker:
    np.random.seed(seed)
    Faker.seed(seed)

    fake = Faker()
    fake.add_provider(RecordProvider)

    return fake
