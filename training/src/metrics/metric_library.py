import numpy as np
import wandb
from fess38.util.registry import Registry
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_recall_curve as precision_recall_curve_impl
from sklearn.metrics import precision_recall_fscore_support, r2_score, roc_auc_score
from sklearn.metrics import roc_curve as roc_curve_impl

from ..types import PredictionRecord

metric_library = Registry("metric_library")


@metric_library("accuracy")
def accuracy(records: list[PredictionRecord], threshold: float, normalize: bool):
    wandb.summary["accuracy"] = accuracy_score(
        y_true=[record.labels[0] for record in records],
        y_pred=[int(record.predictions[1] >= threshold) for record in records],
        sample_weight=[record.sample_weight or 1 for record in records],
        normalize=normalize,
    )


def _precision_recall_f1(
    records: list[PredictionRecord],
    threshold: float,
    labels: list[str] | None,
    pos_label: int,
    average: str,
    zero_division: int,
):
    if wandb.summary.get("precision") is not None:
        return

    precision, recall, fbeta_score, _ = precision_recall_fscore_support(
        y_true=[record.labels[0] for record in records],
        y_pred=[int(record.predictions[1] >= threshold) for record in records],
        sample_weight=[record.sample_weight or 1 for record in records],
        labels=labels,
        pos_label=pos_label,
        average=average,
        zero_division=zero_division,
    )

    wandb.summary["precision"] = precision
    wandb.summary["recall"] = recall
    wandb.summary["f1"] = fbeta_score


@metric_library("precision")
def precision(records: list[PredictionRecord], **kwargs):
    _precision_recall_f1(records, **kwargs)


@metric_library("recall")
def recall(records: list[PredictionRecord], **kwargs):
    _precision_recall_f1(records, **kwargs)


@metric_library("f1")
def f1(records: list[PredictionRecord], **kwargs):
    _precision_recall_f1(records, **kwargs)


@metric_library("roc_auc")
def roc_auc(
    records: list[PredictionRecord],
    average: str | None,
    max_fpr: float | None,
    multi_class: str,
    labels: list[str] | None,
):
    wandb.summary["roc_auc"] = roc_auc_score(
        y_true=[record.labels[0] for record in records],
        y_score=[record.predictions[1] for record in records],
        sample_weight=[record.sample_weight or 1 for record in records],
        average=average,
        max_fpr=max_fpr,
        multi_class=multi_class,
        labels=labels,
    )


@metric_library("r2")
def r2(records: list[PredictionRecord], multioutput: str | None, force_finite: bool):
    wandb.summary["r2"] = r2_score(
        y_true=[record.labels[0] for record in records],
        y_pred=[record.predictions[0] for record in records],
        sample_weight=[record.sample_weight or 1 for record in records],
        multioutput=multioutput,
        force_finite=force_finite,
    )


@metric_library("max_accuracy_threshold")
def max_accuracy_threshold(records: list[PredictionRecord]):
    y_true = [record.labels[0] for record in records]
    y_score = [record.predictions[1] for record in records]
    sample_weight = [record.sample_weight or 1 for record in records]

    _, _, thresholds = roc_curve_impl(
        y_true=y_true,
        y_score=y_score,
        sample_weight=sample_weight,
    )
    thresholds = np.nan_to_num(thresholds, posinf=0.0, neginf=0.0)

    accuracies = [
        accuracy_score(
            y_true=y_true,
            y_pred=(y_score >= threshold).astype(int),
            sample_weight=sample_weight,
        )
        for threshold in thresholds
    ]
    max_accuracy_index = np.argmax(accuracies)

    wandb.summary["max_accuracy_threshold"] = thresholds[max_accuracy_index]
    wandb.summary["max_accuracy"] = accuracies[max_accuracy_index]


@metric_library("max_f1_threshold")
def max_f1_threshold(
    records: list[PredictionRecord], pos_label, drop_intermediate: bool
):
    precision, recall, thresholds = precision_recall_curve_impl(
        y_true=[record.labels[0] for record in records],
        probas_pred=[record.predictions[1] for record in records],
        sample_weight=[record.sample_weight or 1 for record in records],
        pos_label=pos_label,
        drop_intermediate=drop_intermediate,
    )
    thresholds = np.nan_to_num(thresholds, posinf=0.0, neginf=0.0)

    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-9)
    f1_scores = np.nan_to_num(f1_scores, posinf=0.0, neginf=0.0)
    max_f1_index = np.argmax(f1_scores)

    wandb.summary["max_f1_threshold"] = thresholds[max_f1_index]
    wandb.summary["max_f1"] = f1_scores[max_f1_index]


@metric_library("precision_recall_curve")
def precision_recall_curve(
    records: list[PredictionRecord],
    labels: list[str] | None,
    classes_to_plot: list[str] | None,
):
    precision_recall_curve = wandb.plot.pr_curve(
        y_true=[record.labels[0] for record in records],
        y_probas=[record.predictions for record in records],
        labels=labels,
        classes_to_plot=classes_to_plot,
    )
    wandb.log({"precision_recall_curve": precision_recall_curve})


@metric_library("roc_curve")
def roc_curve(
    records: list[PredictionRecord],
    labels: list[str] | None,
    classes_to_plot: list[str] | None,
):
    roc_curve = wandb.plot.roc_curve(
        y_true=[record.labels[0] for record in records],
        y_probas=[record.predictions for record in records],
        labels=labels,
        classes_to_plot=classes_to_plot,
    )
    wandb.log({"roc_curve": roc_curve})


@metric_library("confusion_matrix")
def confusion_matrix(records: list[PredictionRecord], class_names: list[str] | None):
    confusion_matrix = wandb.plot.confusion_matrix(
        y_true=[record.labels[0] for record in records],
        probs=np.array([record.predictions for record in records]),
        class_names=class_names,
    )
    wandb.log({"confusion_matrix": confusion_matrix})
