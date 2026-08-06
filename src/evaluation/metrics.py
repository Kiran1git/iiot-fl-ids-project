"""Evaluation metrics and visualization functions for the IIoT project.

This module owns the metric-computation and plot-generation contracts of
SDS Section 14.8. Each plot listed in SDS Section 17 has exactly one owning
function here, and no call site may reimplement any of them.

Function ownership:
  - generate_class_distribution_plot: called by experiments/run_preprocessing.py
    immediately after run_preprocessing_pipeline returns (created early — the
    explicit cross-boundary exception in SDS Section 14.2 / Phase 4 allow-list).
  - compute_all_metrics, generate_confusion_matrix,
    generate_classification_report, generate_training_curves_plot: called by
    src/evaluation/compare_fl_vs_centralized.py::run_evaluation_pipeline,
    driven by experiments/run_evaluation.py.

Per SDS Section 11's import graph this module imports nothing from
src/preprocessing/, src/models/, src/centralized/, or src/federated/.
"""

import os

import matplotlib
matplotlib.use("Agg")  # non-interactive backend — safe for scripts and tests

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)



def generate_class_distribution_plot(
    df: pd.DataFrame,
    label_column: str,
    output_path: str,
) -> None:
    """Plot a bar chart of sample counts per attack category and save to disk.

    Purpose:
        Count samples per unique value of ``label_column``, then produce and
        save a horizontal bar chart at ``output_path``.  Called from
        ``experiments/run_preprocessing.py`` immediately after
        ``run_preprocessing_pipeline`` returns.

    Args:
        df: Labeled (and optionally processed) DataFrame containing at minimum
            the ``label_column``.  The in-memory result of the preprocessing
            pipeline, or ``data/processed/edge_iiotset_processed.csv``
            reloaded.
        label_column: Name of the label column to count
            (always ``"label"`` in this project).
        output_path: Absolute or relative path where the PNG is saved
            (``outputs/results/class_distribution.png``).

    Returns:
        None

    Raises:
        KeyError: If ``label_column`` is not present in ``df``.

    Dependencies:
        pandas, matplotlib.
    """
    if label_column not in df.columns:
        raise KeyError(
            f"label_column '{label_column}' not found in DataFrame. "
            f"Available columns: {list(df.columns)}"
        )

    counts = df[label_column].value_counts().sort_values(ascending=True)

    fig, ax = plt.subplots(figsize=(10, max(4, len(counts) * 0.4)))
    ax.barh(counts.index, counts.values, color="steelblue", edgecolor="white")
    ax.set_xlabel("Sample Count", fontsize=12)
    ax.set_title("Class Distribution — Edge-IIoTset", fontsize=14)
    ax.tick_params(axis="y", labelsize=9)

    for i, (val, label) in enumerate(zip(counts.values, counts.index)):
        ax.text(val + max(counts.values) * 0.005, i, str(val),
                va="center", fontsize=8)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)



def compute_all_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list,
) -> dict:
    """Compute the full metric suite for a set of predictions.

    Purpose:
        Compute accuracy plus macro- and weighted-averaged precision, recall
        and F1 for one model's predictions against ground truth.  This is the
        single metric-computation function in the project — no call site may
        recompute any of these values independently.

    Args:
        y_true: Integer class indices of the ground-truth labels, shape
            ``(n_samples,)``.
        y_pred: Integer class indices of the predicted labels, shape
            ``(n_samples,)``.
        class_names: Ordered list of human-readable class names, indexed by
            class integer.  Retained for API symmetry with the other functions
            in this module; the averaged metrics themselves are name-agnostic.

    Returns:
        dict: Keys ``accuracy``, ``precision_macro``, ``recall_macro``,
            ``f1_macro``, ``precision_weighted``, ``recall_weighted``,
            ``f1_weighted``, and ``loss``.  ``loss`` is initialised to ``None``
            here and is always populated by the caller from
            ``model.evaluate()`` (SDS Section 14.9) — it is never recomputed
            independently.

    Raises:
        ValueError: If ``y_true`` and ``y_pred`` differ in length.

    Dependencies:
        sklearn.metrics (accuracy_score, precision_recall_fscore_support).
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    if len(y_true) != len(y_pred):
        raise ValueError(
            f"y_true and y_pred must have the same length: "
            f"got {len(y_true)} and {len(y_pred)}."
        )

    precision_macro, recall_macro, f1_macro, _ = (
        precision_recall_fscore_support(
            y_true, y_pred, average="macro", zero_division=0
        )
    )
    precision_weighted, recall_weighted, f1_weighted, _ = (
        precision_recall_fscore_support(
            y_true, y_pred, average="weighted", zero_division=0
        )
    )

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(precision_macro),
        "recall_macro": float(recall_macro),
        "f1_macro": float(f1_macro),
        "precision_weighted": float(precision_weighted),
        "recall_weighted": float(recall_weighted),
        "f1_weighted": float(f1_weighted),
        # Always overwritten by the caller with model.evaluate()'s loss
        # (SDS Section 14.9). Never computed here.
        "loss": None,
    }


def generate_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list,
    output_path: str,
) -> np.ndarray:
    """Compute a confusion matrix, save it as a heatmap PNG, and return it.

    Purpose:
        Own the confusion-matrix deliverable for both models (SDS Section 17).
        Called twice by ``run_evaluation_pipeline`` — once for the centralized
        model and once for the federated global model — with different
        ``output_path`` values.

    Args:
        y_true: Integer class indices of the ground-truth labels.
        y_pred: Integer class indices of the predicted labels.
        class_names: Ordered list of class names used as axis tick labels.
        output_path: Path where the PNG is saved
            (``outputs/results/*_confusion_matrix.png``).

    Returns:
        numpy.ndarray: The raw confusion matrix, shape
            ``(len(class_names), len(class_names))``.

    Raises:
        Nothing under normal operation.

    Dependencies:
        sklearn.metrics.confusion_matrix, matplotlib.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    labels = list(range(len(class_names)))
    matrix = confusion_matrix(y_true, y_pred, labels=labels)

    # Row-normalise for colour only; the printed cells stay as raw counts so
    # the plot remains readable under this dataset's 1614:1 class imbalance.
    row_sums = matrix.sum(axis=1, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        normalised = np.divide(
            matrix,
            row_sums,
            out=np.zeros_like(matrix, dtype=float),
            where=row_sums != 0,
        )

    size = max(6.0, len(class_names) * 0.7)
    fig, ax = plt.subplots(figsize=(size, size))
    image = ax.imshow(normalised, cmap="Blues", vmin=0.0, vmax=1.0)

    ax.set_xticks(labels)
    ax.set_yticks(labels)
    ax.set_xticklabels(class_names, rotation=90, fontsize=8)
    ax.set_yticklabels(class_names, fontsize=8)
    ax.set_xlabel("Predicted label", fontsize=11)
    ax.set_ylabel("True label", fontsize=11)
    ax.set_title("Confusion Matrix (colour = row-normalised)", fontsize=13)

    for row in labels:
        for col in labels:
            ax.text(
                col,
                row,
                str(matrix[row, col]),
                ha="center",
                va="center",
                fontsize=6,
                color="white" if normalised[row, col] > 0.5 else "black",
            )

    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    return matrix


def generate_classification_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list,
    output_path: str,
) -> str:
    """Compute the per-class precision/recall/F1 report and save it as text.

    Purpose:
        Own the classification-report deliverable for both models.  Called
        twice by ``run_evaluation_pipeline`` with different ``output_path``
        values.

    Args:
        y_true: Integer class indices of the ground-truth labels.
        y_pred: Integer class indices of the predicted labels.
        class_names: Ordered list of class names, indexed by class integer.
        output_path: Path where the text report is written
            (``outputs/results/*_classification_report.txt``).

    Returns:
        str: The report text, exactly as written to disk.

    Raises:
        Nothing under normal operation.

    Dependencies:
        sklearn.metrics.classification_report.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    report_text = classification_report(
        y_true,
        y_pred,
        labels=list(range(len(class_names))),
        target_names=list(class_names),
        digits=4,
        zero_division=0,
    )

    with open(output_path, "w", encoding="utf-8") as handle:
        handle.write(report_text)

    return report_text


def generate_training_curves_plot(
    history_csv_path: str,
    output_path: str,
) -> None:
    """Plot the centralized model's per-epoch loss and accuracy curves.

    Purpose:
        Own the ``centralized_training_curves.png`` deliverable
        (SDS Section 17).  Reads the history CSV written by
        ``train_centralized_model`` and renders train/validation loss and
        train/validation accuracy side by side.

    Args:
        history_csv_path: Path to ``outputs/results/centralized_history.csv``
            (columns ``epoch, loss, accuracy, val_loss, val_accuracy``).
        output_path: Path where the PNG is saved
            (``outputs/results/centralized_training_curves.png``).

    Returns:
        None

    Raises:
        FileNotFoundError: If ``history_csv_path`` does not exist.

    Dependencies:
        pandas, matplotlib.
    """
    if not os.path.exists(history_csv_path):
        raise FileNotFoundError(
            f"Training history CSV not found: '{history_csv_path}'. "
            f"Run experiments/run_centralized.py first."
        )

    history = pd.read_csv(history_csv_path)

    x_values = (
        history["epoch"] if "epoch" in history.columns
        else pd.Series(range(1, len(history) + 1))
    )

    fig, (loss_ax, acc_ax) = plt.subplots(1, 2, figsize=(13, 5))

    # Columns are selected by name and only plotted when present, so a history
    # file written without a validation split still produces a valid figure.
    if "loss" in history.columns:
        loss_ax.plot(x_values, history["loss"], marker="o",
                     label="Training loss")
    if "val_loss" in history.columns:
        loss_ax.plot(x_values, history["val_loss"], marker="s",
                     label="Validation loss")
    loss_ax.set_xlabel("Epoch")
    loss_ax.set_ylabel("Loss")
    loss_ax.set_title("Centralized Training — Loss")
    loss_ax.grid(alpha=0.3)
    loss_ax.legend()

    if "accuracy" in history.columns:
        acc_ax.plot(x_values, history["accuracy"], marker="o",
                    label="Training accuracy")
    if "val_accuracy" in history.columns:
        acc_ax.plot(x_values, history["val_accuracy"], marker="s",
                    label="Validation accuracy")
    acc_ax.set_xlabel("Epoch")
    acc_ax.set_ylabel("Accuracy")
    acc_ax.set_title("Centralized Training — Accuracy")
    acc_ax.grid(alpha=0.3)
    acc_ax.legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

