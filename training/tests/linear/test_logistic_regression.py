from pathlib import Path

import pytest
from fess38.data_processing.operation.internal import create_dummy_op
from fess38.training.linear.logistic_regression import (
    LogisticRegressionInferenceOp,
    LogisticRegressionTrainOp,
    SGDClassifierTrainOp,
)
from fess38.training.types import SampleRecord
from fess38.util.tests import create_faker

FAKE = create_faker()


@pytest.mark.parametrize(
    (
        "train_op_class",
        "train_config",
        "inference_config",
        "train_records",
        "inference_records",
        "expected",
    ),
    [
        (
            LogisticRegressionTrainOp,
            {"kwargs": {"random_state": 1}},
            {},
            FAKE.records(
                1000,
                {
                    "id": (FAKE.pyint, {}),
                    "labels": (
                        FAKE.random_choices,
                        {"elements": [True, False], "length": 1},
                    ),
                    "num_features": (FAKE.rand, {"size": 100}),
                },
                SampleRecord,
            ),
            FAKE.records(
                100,
                {
                    "id": (FAKE.pyint, {}),
                    "num_features": (FAKE.rand, {"size": 100}),
                },
                SampleRecord,
            ),
            48.49041,
        ),
        (
            SGDClassifierTrainOp,
            {
                "kwargs": {
                    "loss": "log_loss",
                    "penalty": "l1",
                    "max_iter": 100,
                    "random_state": 1,
                }
            },
            {"batch_size": 32},
            FAKE.records(
                10000,
                {
                    "id": (FAKE.pyint, {}),
                    "labels": (
                        FAKE.random_choices,
                        {"elements": [True, False], "length": 1},
                    ),
                    "num_features": (FAKE.rand, {"size": 100}),
                },
                SampleRecord,
            ),
            FAKE.records(
                100,
                {
                    "id": (FAKE.pyint, {}),
                    "num_features": (FAKE.rand, {"size": 100}),
                },
                SampleRecord,
            ),
            41.34718,
        ),
    ],
)
def test_logistic_regression(
    tmp_path: Path,
    train_op_class: type,
    train_config: dict,
    inference_config: dict,
    train_records: list[SampleRecord],
    inference_records: list[SampleRecord],
    expected: float,
):
    train_config.setdefault("output_files", {})
    train_config["output_files"]["model"] = str(tmp_path / "model.bin")
    train_op = create_dummy_op(train_config, train_op_class)
    train_op._consume_fn(train_records, None)

    inference_config.setdefault("input_files", {})
    inference_config["input_files"]["model"] = str(tmp_path / "model.bin")
    inference_op = create_dummy_op(inference_config, LogisticRegressionInferenceOp)

    actual = sum(
        map(lambda x: x.predictions[0], inference_op._map_fn(inference_records, None))
    )
    assert abs(actual - expected) < 1e-5
