import pytest
from fess38.data_processing.operation.internal import create_dummy_op
from fess38.training.train_test_split import TrainTestSplitMapOp
from fess38.util.tests import create_faker
from fess38.util.typing import PyTree

FAKE = create_faker()


@pytest.mark.parametrize(
    ("config", "records", "expected"),
    [
        (
            {"based_on": ["id"], "sampling_rate": 0.1},
            FAKE.records(100, {"id": (FAKE.pystr, {})}),
            10,
        ),
        (
            {"based_on": ["id"], "sampling_rate": 0.6},
            FAKE.records(100, {"id": (FAKE.pystr, {})}),
            63,
        ),
    ],
)
def test_train_test_split(config: dict, records: list[PyTree], expected: int):
    op = create_dummy_op(config, TrainTestSplitMapOp, num_outputs=2)
    actual = sum(map(lambda x: x.index, op._map_fn(records, None)))
    assert actual == expected
