from pathlib import Path

import pytest
from fess38.data_processing.operation.internal import create_dummy_op
from fess38.training.linear.linear_regression import (
    LinearRegressionInferenceOp,
    LinearRegressionTrainOp,
)
from fess38.training.types import SampleRecord
from fess38.util.tests import create_faker
from fess38.util.typing import PyTree

FAKE = create_faker()


@pytest.mark.parametrize(
    (
        "train_config",
        "inference_config",
        "train_records",
        "inference_records",
        "expected",
    ),
    [
        (
            {"model_name": "LinearRegression"},
            {},
            FAKE.records(
                1000,
                {
                    "id": (FAKE.pyint, {}),
                    "labels": (FAKE.rand, {"size": [1]}),
                    "num_features": (FAKE.rand, {"size": [100]}),
                },
            ),
            FAKE.records(
                100,
                {
                    "id": (FAKE.pyint, {}),
                    "num_features": (FAKE.rand, {"size": [100]}),
                },
            ),
            49.64688,
        ),
        (
            {
                "model_name": "SGDRegressor",
                "kwargs": {"random_state": 1, "penalty": "l1", "max_iter": 100},
            },
            {"batch_size": 32},
            FAKE.records(
                10000,
                {
                    "id": (FAKE.pyint, {}),
                    "labels": (FAKE.rand, {"size": [1]}),
                    "num_features": (FAKE.rand, {"size": [100]}),
                },
            ),
            FAKE.records(
                100,
                {
                    "id": (FAKE.pyint, {}),
                    "num_features": (FAKE.rand, {"size": [100]}),
                },
            ),
            52.64604,
        ),
    ],
)
def test_linear_regression(
    tmp_path: Path,
    train_config: dict,
    inference_config: dict,
    train_records: list[PyTree],
    inference_records: list[PyTree],
    expected: float,
):
    train_config.setdefault("output_files", {})
    train_config["output_files"]["model.bin"] = str(tmp_path / "model.bin")
    train_op = create_dummy_op(train_config, LinearRegressionTrainOp)
    train_records = map(lambda x: SampleRecord(**x), train_records)
    train_op._consume_fn(train_records, None)

    inference_config.setdefault("input_files", {})
    inference_config["input_files"]["model.bin"] = str(tmp_path / "model.bin")
    inference_op = create_dummy_op(inference_config, LinearRegressionInferenceOp)
    inference_records = map(lambda x: SampleRecord(**x), inference_records)

    actual = sum(
        map(lambda x: x.predictions[0], inference_op._map_fn(inference_records, None))
    )
    assert abs(actual - expected) < 1e-5
