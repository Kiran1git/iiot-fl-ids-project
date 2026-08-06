"""Streamlit dashboard for the IIoT Federated IDS project.

Implements SDS Section 18's four-tab specification.

**Read-only enforcement (SDS Section 18, absolute rule).** This module never
calls ``train_centralized_model``, ``run_federated_simulation``,
``partition_iid``, ``run_preprocessing_pipeline`` or
``generate_shap_explanations``. It reads exclusively from ``outputs/`` and
``data/processed/``, and calls only the two read-only utilities Section 11
permits: ``prepare_model_ready_data`` and ``explain_single_prediction``.

**Per-tab isolation.** Every tab loads its own artifacts inside its own
``try``/``except`` and reports a missing artifact with ``st.error``. Deleting
one artifact degrades exactly one tab; the other three still render
(SDS Section 21 Milestone 10 tests precisely this).

**Background sampling is never performed here.** Tab 4's
``shap.GradientExplainer`` is always constructed from the persisted
``outputs/artifacts/shap_background.npy`` written by
``experiments/run_explainability.py`` (SDS Sections 14.10, 18 and 22).

Usage:
    streamlit run dashboard/app.py
"""

import json
import os
import pickle
import sys

# Ensure project root is on the path regardless of invocation style
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import streamlit as st

from src.utils.config_loader import load_config
from src.utils.dataio import read_processed
from src.preprocessing.encode_normalize import prepare_model_ready_data
from src.explainability.shap_utils import explain_single_prediction


# ---------------------------------------------------------------------------
# Cached loaders — every path is built from config["paths"] via os.path.join
# ---------------------------------------------------------------------------

@st.cache_resource
def _load_config() -> dict:
    """Load the project configuration once per session.

    Returns:
        dict: The fully merged configuration.
    """
    return load_config("configs/config.yaml")


@st.cache_resource
def _load_model(model_path: str):
    """Load a trained Keras model, cached across reruns.

    Args:
        model_path: Path to the ``.h5`` model file.

    Returns:
        tensorflow.keras.Model: The loaded model.

    Raises:
        FileNotFoundError: If ``model_path`` does not exist.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(model_path)

    # Imported lazily so a missing-model tab fails with st.error rather than
    # paying TensorFlow's import cost on every dashboard launch.
    import tensorflow

    return tensorflow.keras.models.load_model(model_path)


@st.cache_resource
def _load_background(background_path: str) -> np.ndarray:
    """Load the persisted SHAP background array, cached across reruns.

    Args:
        background_path: Path to ``outputs/artifacts/shap_background.npy``.

    Returns:
        numpy.ndarray: The background array, shape
            ``(background_size, num_features, 1)``.

    Raises:
        FileNotFoundError: If the file does not exist.
    """
    if not os.path.exists(background_path):
        raise FileNotFoundError(background_path)

    # Loaded, never re-sampled (SDS §18 / §22).
    return np.load(background_path)


@st.cache_resource
def _build_explainer(model_path: str, background_path: str):
    """Build a GradientExplainer from the persisted background array.

    Args:
        model_path: Path to the federated global model.
        background_path: Path to ``outputs/artifacts/shap_background.npy``.

    Returns:
        shap.GradientExplainer: Explainer over the persisted background.

    Raises:
        FileNotFoundError: If either input file is missing.
    """
    import shap

    model = _load_model(model_path)
    background = _load_background(background_path)
    return shap.GradientExplainer(model, background)


@st.cache_data
def _load_artifacts(artifacts_dir: str) -> tuple:
    """Load the preprocessing artifacts needed by more than one tab.

    Args:
        artifacts_dir: ``config["paths"]["artifacts_dir"]``.

    Returns:
        tuple: ``(test_indices, feature_columns, label_encoder,
            class_mapping)``.

    Raises:
        FileNotFoundError: If any artifact is missing.
    """
    def _load_pickle(filename: str):
        path = os.path.join(artifacts_dir, filename)
        if not os.path.exists(path):
            raise FileNotFoundError(path)
        with open(path, "rb") as handle:
            return pickle.load(handle)

    test_indices = _load_pickle("test_indices.pkl")
    feature_columns = _load_pickle("feature_names.pkl")
    label_encoder = _load_pickle("label_encoder.pkl")

    class_mapping_path = os.path.join(artifacts_dir, "class_mapping.json")
    if not os.path.exists(class_mapping_path):
        raise FileNotFoundError(class_mapping_path)
    with open(class_mapping_path, "r", encoding="utf-8") as handle:
        class_mapping = json.load(handle)

    return test_indices, feature_columns, label_encoder, class_mapping


@st.cache_data
def _load_test_tensors(
    processed_path: str,
    artifacts_dir: str,
    sample_limit: int = 500,
) -> tuple:
    """Build the test tensors used by Tabs 2 and 4.

    Purpose:
        Prepare a bounded slice of the test split for interactive browsing.
        The full test split is ~443k rows; the dashboard only ever displays one
        row at a time, so materialising all of it would cost hundreds of
        megabytes for no visible benefit.

    Args:
        processed_path: Path to the processed dataset file.
        artifacts_dir: ``config["paths"]["artifacts_dir"]``.
        sample_limit: Maximum number of test rows exposed in the selector.

    Returns:
        tuple: ``(X_test, y_test, class_mapping, feature_columns)``.

    Raises:
        FileNotFoundError: If the processed data or any artifact is missing.
    """
    if not os.path.exists(processed_path):
        raise FileNotFoundError(processed_path)

    test_indices, feature_columns, label_encoder, class_mapping = (
        _load_artifacts(artifacts_dir)
    )

    limited_indices = np.asarray(test_indices)[:sample_limit]
    df = read_processed(processed_path)

    # The sole tensor-preparation path (SDS §14.2) — the reshape/one-hot logic
    # is never reimplemented in the dashboard.
    X_test, y_test = prepare_model_ready_data(
        df,
        limited_indices,
        feature_columns,
        label_encoder,
        len(class_mapping),
    )
    del df

    return X_test, y_test, class_mapping, feature_columns


# ---------------------------------------------------------------------------
# Tabs — each renders independently and reports its own missing artifacts
# ---------------------------------------------------------------------------

def render_overview_tab(config: dict) -> None:
    """Render Tab 1: dataset overview and class distribution.

    Args:
        config: The project configuration.

    Returns:
        None
    """
    st.header("Overview")

    try:
        artifacts_dir = config["paths"]["artifacts_dir"]
        results_dir = config["paths"]["results_dir"]

        _, feature_columns, _, class_mapping = _load_artifacts(artifacts_dir)

        col_a, col_b = st.columns(2)
        col_a.metric("Features", len(feature_columns))
        # num_classes is always len(class_mapping) — never hardcoded (§6/§22).
        col_b.metric("Classes", len(class_mapping))

        distribution_path = os.path.join(results_dir, "class_distribution.png")
        if os.path.exists(distribution_path):
            st.image(distribution_path, caption="Class distribution")
        else:
            st.error(
                "Preprocessing artifacts not found. "
                "Run experiments/run_preprocessing.py first."
            )

        with st.expander("Class mapping"):
            st.json(class_mapping)

    except FileNotFoundError:
        st.error(
            "Preprocessing artifacts not found. "
            "Run experiments/run_preprocessing.py first."
        )
    except Exception as exc:  # pragma: no cover - defensive UI guard
        st.error(f"Overview tab could not render: {exc}")


def render_predictions_tab(config: dict) -> None:
    """Render Tab 2: per-sample prediction against the federated model.

    Args:
        config: The project configuration.

    Returns:
        None
    """
    st.header("Predictions")

    try:
        models_dir = config["paths"]["models_dir"]
        artifacts_dir = config["paths"]["artifacts_dir"]
        processed_path = os.path.join(
            config["paths"]["processed_data_dir"],
            config["paths"]["processed_data_file"],
        )

        model_path = os.path.join(models_dir, "federated_global_model.h5")
        model = _load_model(model_path)
        X_test, y_test, class_mapping, _ = _load_test_tensors(
            processed_path, artifacts_dir
        )

        # Shared across tabs via session_state so Tab 4 explains the sample
        # selected here (SDS §18, Tab 4).
        sample_index = st.slider(
            "Test sample index",
            min_value=0,
            max_value=len(X_test) - 1,
            value=int(st.session_state.get("sample_index", 0)),
        )
        st.session_state["sample_index"] = sample_index

        sample = X_test[sample_index: sample_index + 1]
        probabilities = model.predict(sample, verbose=0)[0]
        predicted_index = int(np.argmax(probabilities))
        true_index = int(np.argmax(y_test[sample_index]))

        # Reverse lookup with keys cast to int (SDS §18, Tab 2).
        reverse_mapping = {int(k): v for k, v in class_mapping.items()}

        col_a, col_b = st.columns(2)
        col_a.metric("Predicted class", reverse_mapping[predicted_index])
        col_b.metric("True class", reverse_mapping[true_index])
        st.metric("Confidence", f"{float(probabilities[predicted_index]):.4f}")

        st.bar_chart(
            pd.DataFrame(
                {"probability": probabilities},
                index=[reverse_mapping[i] for i in range(len(probabilities))],
            )
        )

    except FileNotFoundError:
        st.error(
            "Trained model or test data not found. "
            "Run experiments/run_federated.py and "
            "experiments/run_preprocessing.py first."
        )
    except Exception as exc:  # pragma: no cover - defensive UI guard
        st.error(f"Predictions tab could not render: {exc}")


def render_comparison_tab(config: dict) -> None:
    """Render Tab 3: federated vs centralized comparison artifacts.

    Args:
        config: The project configuration.

    Returns:
        None
    """
    st.header("FL vs. Centralized")

    try:
        results_dir = config["paths"]["results_dir"]

        comparison_path = os.path.join(results_dir, "comparison_table.csv")
        size_path = os.path.join(results_dir, "model_size_comparison.csv")
        convergence_path = os.path.join(results_dir, "convergence_plot.png")
        centralized_cm = os.path.join(
            results_dir, "centralized_confusion_matrix.png"
        )
        federated_cm = os.path.join(
            results_dir, "federated_confusion_matrix.png"
        )

        required = [
            comparison_path,
            size_path,
            convergence_path,
            centralized_cm,
            federated_cm,
        ]
        missing = [path for path in required if not os.path.exists(path)]
        if missing:
            raise FileNotFoundError(missing[0])

        st.subheader("Comparison table")
        st.dataframe(pd.read_csv(comparison_path))

        st.subheader("Model size comparison")
        st.dataframe(pd.read_csv(size_path))

        st.subheader("Convergence")
        st.image(convergence_path)

        st.subheader("Confusion matrices")
        left, right = st.columns(2)
        left.image(centralized_cm, caption="Centralized")
        right.image(federated_cm, caption="Federated")

    except FileNotFoundError:
        st.error(
            "Comparison results not found. "
            "Run experiments/run_evaluation.py first."
        )
    except Exception as exc:  # pragma: no cover - defensive UI guard
        st.error(f"Comparison tab could not render: {exc}")


def render_explainability_tab(config: dict) -> None:
    """Render Tab 4: SHAP explanation for the selected sample.

    Args:
        config: The project configuration.

    Returns:
        None
    """
    st.header("Explainability")

    try:
        artifacts_dir = config["paths"]["artifacts_dir"]
        models_dir = config["paths"]["models_dir"]
        shap_plots_dir = config["paths"]["shap_plots_dir"]
        processed_path = os.path.join(
            config["paths"]["processed_data_dir"],
            config["paths"]["processed_data_file"],
        )

        model_path = os.path.join(models_dir, "federated_global_model.h5")
        background_path = os.path.join(artifacts_dir, "shap_background.npy")

        # Always built from the persisted background array — the dashboard
        # never re-samples a background set (SDS §14.10 / §18 / §22).
        explainer = _build_explainer(model_path, background_path)
        model = _load_model(model_path)

        X_test, _, _, feature_columns = _load_test_tensors(
            processed_path, artifacts_dir
        )

        sample_index = int(st.session_state.get("sample_index", 0))
        sample_index = min(sample_index, len(X_test) - 1)
        st.caption(
            f"Explaining test sample #{sample_index} "
            f"(selected on the Predictions tab)."
        )

        sample = X_test[sample_index: sample_index + 1]
        shap_values = explain_single_prediction(model, explainer, sample)

        # Mean absolute contribution across classes for this one sample.
        contributions = np.mean(
            [np.abs(np.asarray(sv)[0]) for sv in shap_values], axis=0
        )
        top_n = min(20, len(contributions))
        order = np.argsort(contributions)[::-1][:top_n]

        st.subheader(f"Top {top_n} contributing features for this prediction")
        st.bar_chart(
            pd.DataFrame(
                {"mean |SHAP value|": contributions[order]},
                index=[feature_columns[i] for i in order],
            )
        )

        summary_path = os.path.join(shap_plots_dir, "shap_summary_plot.png")
        if os.path.exists(summary_path):
            st.subheader("Global explanation")
            st.image(summary_path, caption="SHAP summary plot")
        else:
            raise FileNotFoundError(summary_path)

    except FileNotFoundError:
        st.error(
            "SHAP artifacts not found. "
            "Run experiments/run_explainability.py first."
        )
    except Exception as exc:  # pragma: no cover - defensive UI guard
        st.error(f"Explainability tab could not render: {exc}")


def main() -> None:
    """Render the four-tab dashboard.

    Returns:
        None
    """
    config = _load_config()

    st.set_page_config(
        page_title=config["dashboard"]["title"],
        layout="wide",
    )
    st.title(config["dashboard"]["title"])

    overview, predictions, comparison, explainability = st.tabs(
        ["Overview", "Predictions", "FL vs. Centralized", "Explainability"]
    )

    # Each tab is rendered independently; a failure inside one is caught by
    # that tab's own handler and never prevents the others from rendering.
    with overview:
        render_overview_tab(config)

    with predictions:
        render_predictions_tab(config)

    with comparison:
        render_comparison_tab(config)

    with explainability:
        render_explainability_tab(config)


if __name__ == "__main__":
    main()
