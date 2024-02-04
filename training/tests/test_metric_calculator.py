from pathlib import Path
from typing import OrderedDict

import pytest
import wandb
from fess38.data_processing.operation.internal import create_dummy_op
from fess38.training.metric_calculator import MetricCalculationConsumeOp
from fess38.training.types import PredictionRecord
from fess38.util.tests import create_faker

FAKE = create_faker()


@pytest.fixture
def set_env_vars(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("WANDB_MODE", "offline")
    monkeypatch.setenv("WANDB_DIR", str(tmp_path))
    monkeypatch.setenv("WANDB_SILENT", "true")


@pytest.mark.parametrize(
    ("config", "records", "expected"),
    [
        (
            {
                "metric_configs": {
                    "accuracy": {"threshold": 0.0},
                    "precision": {"threshold": 0.0},
                    "recall": {"threshold": 0.0},
                    "f1": {"threshold": 0.0},
                    "roc_auc": {},
                    "max_accuracy_threshold": {},
                    "precision_recall_curve": {},
                    "roc_curve": {},
                    "confusion_matrix": {},
                }
            },
            FAKE.records(
                1000,
                {
                    "id": (FAKE.pyint, {}),
                    "labels": (
                        FAKE.random_elements,
                        {
                            "elements": OrderedDict({0: 0.1, 1: 0.9}),
                            "length": 1,
                            "use_weighting": True,
                        },
                    ),
                    "predictions": (FAKE.rand, {"size": 1}),
                },
                PredictionRecord,
            ),
            {
                "accuracy": 0.899,
                "precision": 0.899,
                "recall": 1.0,
                "f1": 0.946,
                "roc_auc": 0.478,
                "max_accuracy_threshold": 0,
            },
        ),
        (
            {
                "metric_configs": {
                    "accuracy": {"threshold": 1.0},
                    "precision": {"threshold": 1.0},
                    "recall": {"threshold": 1.0},
                    "f1": {"threshold": 1.0},
                    "roc_auc": {},
                    "max_accuracy_threshold": {},
                    "precision_recall_curve": {},
                    "roc_curve": {},
                    "confusion_matrix": {},
                }
            },
            FAKE.records(
                1000,
                {
                    "id": (FAKE.pyint, {}),
                    "labels": (
                        FAKE.random_elements,
                        {
                            "elements": OrderedDict({0: 0.1, 1: 0.9}),
                            "length": 1,
                            "use_weighting": True,
                        },
                    ),
                    "predictions": (FAKE.rand, {"size": 1}),
                },
                PredictionRecord,
            ),
            {
                "accuracy": 0.093,
                "precision": 0.0,
                "recall": 0.0,
                "f1": 0.0,
                "roc_auc": 0.486,
                "max_accuracy_threshold": 0,
            },
        ),
        (
            {
                "metric_configs": {
                    "accuracy": {"threshold": 0.5},
                    "precision": {"threshold": 0.5},
                    "recall": {"threshold": 0.5},
                    "f1": {"threshold": 0.5},
                    "roc_auc": {},
                    "max_accuracy_threshold": {},
                    "precision_recall_curve": {},
                    "roc_curve": {},
                    "confusion_matrix": {},
                }
            },
            FAKE.records(
                1000,
                {
                    "id": (FAKE.pyint, {}),
                    "labels": (
                        FAKE.random_elements,
                        {
                            "elements": OrderedDict({0: 0.5, 1: 0.5}),
                            "length": 1,
                            "use_weighting": True,
                        },
                    ),
                    "predictions": (FAKE.rand, {"size": 1}),
                },
                PredictionRecord,
            ),
            {
                "accuracy": 0.492,
                "precision": 0.482,
                "recall": 0.491,
                "f1": 0.486,
                "roc_auc": 0.489,
                "max_accuracy_threshold": 0.992,
            },
        ),
        (
            {"metric_configs": {"r2": {}}},
            FAKE.records(
                1000,
                {
                    "id": (FAKE.pyint, {}),
                    "labels": (FAKE.rand, {"size": 1}),
                    "predictions": (FAKE.rand, {"size": 1}),
                },
                PredictionRecord,
            ),
            {"r2": -0.875},
        ),
    ],
)
def test_metric_calculator(
    set_env_vars,
    config: dict,
    records: list[PredictionRecord],
    expected: dict[str, float],
):
    for record in records:
        record.predictions.append(1 - record.predictions[0])

    op = create_dummy_op(config, MetricCalculationConsumeOp)
    op._consume_fn(records, None)

    for metric, expected_value in expected.items():
        assert abs(wandb.summary[metric] - expected_value) < 1e-3

    wandb.finish()
