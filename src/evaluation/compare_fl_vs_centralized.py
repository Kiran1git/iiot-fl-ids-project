"""Federated-vs-centralized comparison for the IIoT Federated IDS project.

This module owns SDS Section 14.9 in full: the evaluation of both authoritative
models on the shared held-out test split, the two comparison CSVs, the
convergence plot, and ``run_evaluation_pipeline`` — the single top-level
orchestrator called by ``experiments/run_evaluation.py``.

Fixed decisions inherited from the SDS, restated here because they are the
easiest ones to get subtly wrong:

  - The "Centralized" row is always ``centralized_best_model.h5``; the
    "Federated" row is always ``federated_global_model.h5``.
    ``centralized_last_model.h5`` is audit-only and is never read here
    (SDS Section 15).
  - ``loss`` is always sourced from ``model.evaluate()`` and never recomputed
    independently (SDS Section 14.9 / Section 16).
  - ``num_classes`` is derived as ``len(class_mapping)`` — never hardcoded
    (SDS Section 6 / Section 22).
  - Tensor preparation goes through ``prepare_model_ready_data`` exclusively;
    the reshape/one-hot logic is never reimplemented here (SDS Section 14.2).
  - ``federated_history.csv`` carries five columns, not three, so the
    aggregated accuracy series is selected **by name**
    (``aggregated_accuracy``), never by position.
"""

import json
import logging
import os
import pickle
import time

import matplotlib
matplotlib.use("Agg")  # non-interactive backend — safe for scripts and tests

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow

from src.evaluation.metrics import (
    compute_all_metrics,
    generate_classification_report,
    generate_confusion_matrix,
    generate_training_curves_plot,
)
from src.preprocessing.encode_normalize import prepare_model_ready_data
from src.utils.dataio import read_processed


# Module-level fallback only. The real logger is always injected by
# experiments/run_evaluation.py via src.utils.logger.get_logger (SDS §13).
_logger = logging.getLogger(__name__)


def evaluate_model_on_test_set(
    model_path: str,
    X_test: np.ndarray,
    y_test: np.ndarray,
    class_names: list,
    logger: logging.Logger = None,
) -> dict:
    """Load a trained model and evaluate it on the shared held-out test set.

    Purpose:
        Measure both accuracy metrics and timing metrics for one model, using
        the exact computation fixed by SDS Section 14.9 — no alternative
        method is permitted.  Called once per model by
        ``run_evaluation_pipeline``.

    Args:
        model_path: Path to the ``.h5`` model file
            (``centralized_best_model.h5`` or ``federated_global_model.h5``).
        X_test: Test feature tensor, shape ``(n_test, num_features, 1)``,
            produced by ``prepare_model_ready_data``.
        y_test: One-hot test labels, shape ``(n_test, num_classes)``,
            produced by ``prepare_model_ready_data``.
        class_names: Ordered list of class names, indexed by class integer.
        logger: Optional logger injected by the orchestrator.  Falls back to
            this module's logger when omitted.

    Returns:
        dict: ``accuracy``, ``precision_macro``, ``recall_macro``, ``f1_macro``,
            ``precision_weighted``, ``recall_weighted``, ``f1_weighted``,
            ``loss`` (always from ``model.evaluate()``),
            ``inference_time_seconds``, ``prediction_latency_ms_per_sample``,
            ``model_size_mb``, plus ``y_pred`` and ``y_true`` integer arrays for
            downstream confusion-matrix and classification-report generation.

    Raises:
        FileNotFoundError: If ``model_path`` does not exist.

    Dependencies:
        tensorflow.keras.models.load_model,
        src.evaluation.metrics.compute_all_metrics, time, os, numpy.
    """
    log = logger or _logger

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model file not found: '{model_path}'. Run the corresponding "
            f"training script (experiments/run_centralized.py or "
            f"experiments/run_federated.py) first."
        )

    log.info("Loading model for evaluation: '%s'", model_path)
    model = tensorflow.keras.models.load_model(model_path)

    # Loss is always sourced here, from model.evaluate(), and never
    # recomputed independently anywhere downstream (SDS §14.9).
    loss_value, accuracy_value = model.evaluate(X_test, y_test, verbose=0)

    start = time.time()
    y_pred_probs = model.predict(X_test, verbose=0)
    inference_time_seconds = time.time() - start

    y_pred = np.argmax(y_pred_probs, axis=1)
    y_true = np.argmax(y_test, axis=1)  # y_test is one-hot by construction

    metrics = compute_all_metrics(y_true, y_pred, class_names)
    metrics["loss"] = float(loss_value)

    # Both figures measure the same quantity by two different routes; a
    # divergence beyond float rounding means the two are not looking at the
    # same predictions, which is worth surfacing rather than silently averaging.
    if abs(metrics["accuracy"] - float(accuracy_value)) > 1e-6:
        log.warning(
            "Accuracy divergence for '%s': compute_all_metrics=%.10f vs "
            "model.evaluate=%.10f (delta=%.3e)",
            model_path,
            metrics["accuracy"],
            float(accuracy_value),
            abs(metrics["accuracy"] - float(accuracy_value)),
        )

    metrics["inference_time_seconds"] = float(inference_time_seconds)
    metrics["prediction_latency_ms_per_sample"] = float(
        inference_time_seconds / len(X_test) * 1000
    )
    metrics["model_size_mb"] = float(
        os.path.getsize(model_path) / (1024 * 1024)
    )
    metrics["y_pred"] = y_pred
    metrics["y_true"] = y_true

    log.info(
        "Evaluated '%s' — accuracy=%.4f | macro-F1=%.4f | weighted-F1=%.4f | "
        "loss=%.4f | inference=%.2fs (%.4f ms/sample) | size=%.2f MB",
        os.path.basename(model_path),
        metrics["accuracy"],
        metrics["f1_macro"],
        metrics["f1_weighted"],
        metrics["loss"],
        metrics["inference_time_seconds"],
        metrics["prediction_latency_ms_per_sample"],
        metrics["model_size_mb"],
    )

    return metrics


def generate_comparison_table(
    centralized_metrics: dict,
    federated_metrics: dict,
    centralized_training_time_seconds: float,
    federated_training_time_seconds: float,
    output_path: str,
) -> pd.DataFrame:
    """Build and persist the final side-by-side comparison table.

    Purpose:
        Own ``outputs/results/comparison_table.csv`` — the artifact that
        answers the project's primary research question (SDS Section 1).

    Args:
        centralized_metrics: Result of ``evaluate_model_on_test_set`` for
            ``centralized_best_model.h5``.
        federated_metrics: Result of ``evaluate_model_on_test_set`` for
            ``federated_global_model.h5``.
        centralized_training_time_seconds: Value read from
            ``outputs/results/centralized_training_time.txt`` by the caller.
        federated_training_time_seconds: Value read from
            ``outputs/results/federated_training_time.txt`` by the caller.
        output_path: Path where the CSV is written.

    Returns:
        pandas.DataFrame: The two-row comparison table, also saved to disk.

    Raises:
        Nothing under normal operation.

    Dependencies:
        pandas.
    """
    def _row(name: str, metrics: dict, training_time: float) -> dict:
        return {
            "Model": name,
            "Accuracy": metrics["accuracy"],
            "Precision_Macro": metrics["precision_macro"],
            "Recall_Macro": metrics["recall_macro"],
            "F1_Macro": metrics["f1_macro"],
            "Precision_Weighted": metrics["precision_weighted"],
            "Recall_Weighted": metrics["recall_weighted"],
            "F1_Weighted": metrics["f1_weighted"],
            "Loss": metrics["loss"],
            "Training_Time_Seconds": training_time,
            "Inference_Time_Seconds": metrics["inference_time_seconds"],
            "Prediction_Latency_ms_per_sample": metrics[
                "prediction_latency_ms_per_sample"
            ],
            "Model_Size_MB": metrics["model_size_mb"],
        }

    table = pd.DataFrame(
        [
            _row("Centralized", centralized_metrics,
                 centralized_training_time_seconds),
            _row("Federated", federated_metrics,
                 federated_training_time_seconds),
        ]
    )

    table.to_csv(output_path, index=False)
    return table


def generate_model_size_comparison(
    centralized_model_path: str,
    federated_model_path: str,
    output_path: str,
) -> pd.DataFrame:
    """Compare on-disk size and parameter count of the two models.

    Purpose:
        Own ``outputs/results/model_size_comparison.csv`` (SDS Section 9/24).
        Both models are loaded solely to call ``count_params()``.

    Args:
        centralized_model_path: Path to ``centralized_best_model.h5``.
        federated_model_path: Path to ``federated_global_model.h5``.
        output_path: Path where the CSV is written.

    Returns:
        pandas.DataFrame: Exactly two rows with columns ``Model``,
            ``Model_Size_MB`` and ``Parameter_Count``.

    Raises:
        FileNotFoundError: If either model path does not exist.

    Dependencies:
        tensorflow.keras.models.load_model, pandas, os.
    """
    for path in (centralized_model_path, federated_model_path):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model file not found: '{path}'.")

    rows = []
    for name, path in (
        ("Centralized", centralized_model_path),
        ("Federated", federated_model_path),
    ):
        model = tensorflow.keras.models.load_model(path)
        rows.append(
            {
                "Model": name,
                "Model_Size_MB": os.path.getsize(path) / (1024 * 1024),
                "Parameter_Count": int(model.count_params()),
            }
        )
        del model

    table = pd.DataFrame(rows)
    table.to_csv(output_path, index=False)
    return table


def generate_convergence_plot(
    federated_history_path: str,
    centralized_history_path: str,
    output_path: str,
) -> None:
    """Overlay federated per-round accuracy with centralized per-epoch accuracy.

    Purpose:
        Own ``outputs/results/convergence_plot.png`` (SDS Section 17).

    Args:
        federated_history_path: Path to
            ``outputs/results/federated_history.csv``.  Written by
            ``run_federated_simulation`` with five columns — the accuracy
            series is therefore selected by the name ``aggregated_accuracy``,
            never by column position.
        centralized_history_path: Path to
            ``outputs/results/centralized_history.csv``.
        output_path: Path where the PNG is written.

    Returns:
        None

    Raises:
        FileNotFoundError: If either history CSV is missing.

    Dependencies:
        pandas, matplotlib.
    """
    for path in (federated_history_path, centralized_history_path):
        if not os.path.exists(path):
            raise FileNotFoundError(f"History CSV not found: '{path}'.")

    federated_history = pd.read_csv(federated_history_path)
    centralized_history = pd.read_csv(centralized_history_path)

    fig, ax = plt.subplots(figsize=(10, 6))

    # Selected by name: federated_history.csv has five columns
    # (round, aggregated_loss, aggregated_accuracy, centralized_loss,
    # centralized_accuracy), so positional indexing would silently plot loss.
    if "aggregated_accuracy" in federated_history.columns:
        rounds = (
            federated_history["round"]
            if "round" in federated_history.columns
            else pd.Series(range(1, len(federated_history) + 1))
        )
        ax.plot(
            rounds,
            federated_history["aggregated_accuracy"],
            marker="o",
            label="Federated — aggregated accuracy (per round)",
        )

    if "accuracy" in centralized_history.columns:
        epochs = (
            centralized_history["epoch"]
            if "epoch" in centralized_history.columns
            else pd.Series(range(1, len(centralized_history) + 1))
        )
        ax.plot(
            epochs,
            centralized_history["accuracy"],
            marker="s",
            linestyle="--",
            label="Centralized — training accuracy (per epoch)",
        )

    ax.set_xlabel("Federated round / Centralized epoch", fontsize=11)
    ax.set_ylabel("Accuracy", fontsize=11)
    ax.set_title("Convergence — Federated vs Centralized", fontsize=13)
    ax.grid(alpha=0.3)
    ax.legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _read_training_time(path: str) -> float:
    """Read a ``*_training_time.txt`` file as a single float.

    Args:
        path: Path to the training-time file.

    Returns:
        float: Total wall-clock training seconds.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file's contents are not parseable as a float.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Training-time file not found: '{path}'. Both training runs must "
            f"complete before the evaluation pipeline can run."
        )

    with open(path, "r", encoding="utf-8") as handle:
        return float(handle.read().strip())


def run_evaluation_pipeline(
    config: dict,
    logger: logging.Logger = None,
) -> None:
    """Orchestrate the entire evaluation stage.

    Purpose:
        The single top-level entry point for evaluation, called by
        ``experiments/run_evaluation.py``.  Executes the fixed 13-step sequence
        of SDS Section 14.9, producing every evaluation-stage deliverable in
        SDS Sections 9/17/24.

    Args:
        config: The fully merged configuration dict from ``load_config``.
        logger: Optional logger injected by the orchestrator.  Falls back to
            this module's logger when omitted.

    Returns:
        None

    Raises:
        Propagates any exception raised by the functions it calls, after
        logging at ERROR level.

    Dependencies:
        All functions in this file, src.evaluation.metrics,
        src.preprocessing.encode_normalize.prepare_model_ready_data,
        src.utils.dataio.read_processed, src.utils.logger, src.utils.seed.
    """
    log = logger or _logger

    try:
        artifacts_dir = config["paths"]["artifacts_dir"]
        models_dir = config["paths"]["models_dir"]
        results_dir = config["paths"]["results_dir"]
        os.makedirs(results_dir, exist_ok=True)

        log.info("=== run_evaluation_pipeline starting ===")

        # ---- Step 1: load processed data and preprocessing artifacts -----
        processed_path = os.path.join(
            config["paths"]["processed_data_dir"],
            config["paths"]["processed_data_file"],
        )
        log.info("Loading processed dataset from '%s'", processed_path)
        df = read_processed(processed_path)

        def _load_pickle(filename: str):
            path = os.path.join(artifacts_dir, filename)
            with open(path, "rb") as handle:
                return pickle.load(handle)

        test_indices = _load_pickle("test_indices.pkl")
        feature_columns = _load_pickle("feature_names.pkl")
        label_encoder = _load_pickle("label_encoder.pkl")

        class_mapping_path = os.path.join(artifacts_dir, "class_mapping.json")
        with open(class_mapping_path, "r", encoding="utf-8") as handle:
            class_mapping = json.load(handle)

        # Derived, never hardcoded (SDS §6/§22).
        num_classes = len(class_mapping)
        class_names = [class_mapping[str(i)] for i in range(num_classes)]

        log.info(
            "Artifacts loaded — test=%d rows | features=%d | classes=%d",
            len(test_indices),
            len(feature_columns),
            num_classes,
        )

        # ---- Step 2: build the shared test tensors -----------------------
        X_test, y_test = prepare_model_ready_data(
            df, test_indices, feature_columns, label_encoder, num_classes
        )
        log.info(
            "Test tensors prepared — X_test=%s | y_test=%s",
            X_test.shape,
            y_test.shape,
        )

        # The full processed frame is not needed past this point; on the
        # 1.9M-row production dataset holding it alongside two loaded Keras
        # models is the difference between a comfortable run and a swap storm.
        del df

        centralized_model_path = os.path.join(
            models_dir, "centralized_best_model.h5"
        )
        federated_model_path = os.path.join(
            models_dir, "federated_global_model.h5"
        )

        # ---- Steps 3-4: evaluate both authoritative models ---------------
        centralized_metrics = evaluate_model_on_test_set(
            centralized_model_path, X_test, y_test, class_names, log
        )
        federated_metrics = evaluate_model_on_test_set(
            federated_model_path, X_test, y_test, class_names, log
        )

        # ---- Step 5: read both persisted training times ------------------
        centralized_training_time_seconds = _read_training_time(
            os.path.join(results_dir, "centralized_training_time.txt")
        )
        federated_training_time_seconds = _read_training_time(
            os.path.join(results_dir, "federated_training_time.txt")
        )
        log.info(
            "Training times — centralized=%.2fs | federated=%.2fs",
            centralized_training_time_seconds,
            federated_training_time_seconds,
        )

        # ---- Step 6: comparison table ------------------------------------
        comparison_path = os.path.join(results_dir, "comparison_table.csv")
        comparison_table = generate_comparison_table(
            centralized_metrics,
            federated_metrics,
            centralized_training_time_seconds,
            federated_training_time_seconds,
            comparison_path,
        )
        log.info("Saved %s", comparison_path)
        log.info(
            "Comparison table:\n%s", comparison_table.to_string(index=False)
        )

        # ---- Step 7: model size comparison -------------------------------
        size_path = os.path.join(results_dir, "model_size_comparison.csv")
        generate_model_size_comparison(
            centralized_model_path, federated_model_path, size_path
        )
        log.info("Saved %s", size_path)

        # ---- Step 8: convergence plot ------------------------------------
        convergence_path = os.path.join(results_dir, "convergence_plot.png")
        generate_convergence_plot(
            os.path.join(results_dir, "federated_history.csv"),
            os.path.join(results_dir, "centralized_history.csv"),
            convergence_path,
        )
        log.info("Saved %s", convergence_path)

        # ---- Step 9: centralized training curves -------------------------
        curves_path = os.path.join(
            results_dir, "centralized_training_curves.png"
        )
        generate_training_curves_plot(
            os.path.join(results_dir, "centralized_history.csv"),
            curves_path,
        )
        log.info("Saved %s", curves_path)

        # ---- Steps 10-11: confusion matrices -----------------------------
        centralized_cm_path = os.path.join(
            results_dir, "centralized_confusion_matrix.png"
        )
        generate_confusion_matrix(
            centralized_metrics["y_true"],
            centralized_metrics["y_pred"],
            class_names,
            centralized_cm_path,
        )
        log.info("Saved %s", centralized_cm_path)

        federated_cm_path = os.path.join(
            results_dir, "federated_confusion_matrix.png"
        )
        generate_confusion_matrix(
            federated_metrics["y_true"],
            federated_metrics["y_pred"],
            class_names,
            federated_cm_path,
        )
        log.info("Saved %s", federated_cm_path)

        # ---- Steps 12-13: classification reports -------------------------
        centralized_report_path = os.path.join(
            results_dir, "centralized_classification_report.txt"
        )
        generate_classification_report(
            centralized_metrics["y_true"],
            centralized_metrics["y_pred"],
            class_names,
            centralized_report_path,
        )
        log.info("Saved %s", centralized_report_path)

        federated_report_path = os.path.join(
            results_dir, "federated_classification_report.txt"
        )
        generate_classification_report(
            federated_metrics["y_true"],
            federated_metrics["y_pred"],
            class_names,
            federated_report_path,
        )
        log.info("Saved %s", federated_report_path)

        log.info("=== run_evaluation_pipeline complete ===")

    except Exception:
        log.error(
            "run_evaluation_pipeline failed — full traceback:", exc_info=True
        )
        raise
