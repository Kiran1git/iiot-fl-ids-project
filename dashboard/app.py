"""Streamlit dashboard for the IIoT Federated IDS project.

Implements SDS Section 18's four-tab specification, plus a fifth,
user-facing tab: **Live Prediction**, which runs an uploaded network-traffic
CSV through the persisted preprocessing artifacts and the persisted federated
global model. That tab is read-only in exactly the same sense as the other
four — it loads fitted encoders, it never fits them, and it never trains.

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
from src.explainability.shap_utils import (
    explain_single_prediction,
    get_top_feature_contributions,
)
from src.inference.predict import (
    FEDERATED_GLOBAL_MODEL_FILENAME,
    InvalidInputError,
    get_expected_input_columns,
    load_inference_artifacts,
    predict_dataframe,
)


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
    """Load a trained Keras model from a persisted H5 file.

    The dashboard is read-only, so the model is loaded without compilation.
    This avoids optimizer/loss deserialization issues and is sufficient for
    prediction and SHAP explanation.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(model_path)

    import tensorflow as tf

    return tf.keras.models.load_model(
        model_path,
        compile=False,
    )


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


def _read_pickle_artifact(artifacts_dir: str, filename: str):
    """Read one pickled artifact from ``outputs/artifacts/``.

    Args:
        artifacts_dir: ``config["paths"]["artifacts_dir"]``.
        filename: Artifact filename, e.g. ``"feature_names.pkl"``.

    Returns:
        object: The unpickled artifact.

    Raises:
        FileNotFoundError: If the artifact does not exist.
    """
    path = os.path.join(artifacts_dir, filename)
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    with open(path, "rb") as handle:
        return pickle.load(handle)


def _read_json_artifact(artifacts_dir: str, filename: str) -> dict:
    """Read one JSON artifact from ``outputs/artifacts/``.

    Args:
        artifacts_dir: ``config["paths"]["artifacts_dir"]``.
        filename: Artifact filename, e.g. ``"class_mapping.json"``.

    Returns:
        dict: The parsed JSON object.

    Raises:
        FileNotFoundError: If the artifact does not exist.
    """
    path = os.path.join(artifacts_dir, filename)
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


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
    test_indices = _read_pickle_artifact(artifacts_dir, "test_indices.pkl")
    feature_columns = _read_pickle_artifact(artifacts_dir, "feature_names.pkl")
    label_encoder = _read_pickle_artifact(artifacts_dir, "label_encoder.pkl")
    class_mapping = _read_json_artifact(artifacts_dir, "class_mapping.json")

    return test_indices, feature_columns, label_encoder, class_mapping


@st.cache_data
def _load_split_sizes(artifacts_dir: str) -> tuple:
    """Load the train/test split sizes for the Overview tab's statistics.

    Purpose:
        Only the lengths are needed, so the index arrays are read, measured and
        released rather than kept in the cache (they total ~1.9M int64 values).

    Args:
        artifacts_dir: ``config["paths"]["artifacts_dir"]``.

    Returns:
        tuple: ``(num_training_samples, num_test_samples)``.

    Raises:
        FileNotFoundError: If either index artifact is missing.
    """
    train_indices = _read_pickle_artifact(artifacts_dir, "train_indices.pkl")
    test_indices = _read_pickle_artifact(artifacts_dir, "test_indices.pkl")
    return len(train_indices), len(test_indices)


@st.cache_data
def _scaler_is_available(artifacts_dir: str) -> bool:
    """Report whether the fitted scaler artifact exists on disk.

    Purpose:
        The processed dataset on disk is *already* scaled by
        ``run_preprocessing_pipeline`` using this exact scaler, so re-applying
        it in the dashboard would double-scale every feature and corrupt the
        prediction. The artifact is therefore only ever reported as present —
        never used to transform anything here.

    Args:
        artifacts_dir: ``config["paths"]["artifacts_dir"]``.

    Returns:
        bool: True when ``scaler.pkl`` exists.
    """
    return os.path.exists(os.path.join(artifacts_dir, "scaler.pkl"))


@st.cache_data
def _load_dataset_preview(processed_path: str, num_rows: int = 20) -> tuple:
    """Load a bounded preview of the processed dataset plus its shape.

    Purpose:
        The processed dataset is ~1.9M rows; the preview shows a handful. The
        row count is taken from the Parquet metadata where possible, so the
        full frame is never materialised just to display its shape.

    Args:
        processed_path: Path to the processed dataset file.
        num_rows: Number of leading rows to display.

    Returns:
        tuple: ``(preview_dataframe, num_total_rows, num_total_columns)``.

    Raises:
        FileNotFoundError: If the processed dataset does not exist.
    """
    if not os.path.exists(processed_path):
        raise FileNotFoundError(processed_path)

    # read_processed owns format dispatch (Parquet or CSV) for the whole
    # project — the dashboard never opens data/processed/ directly.
    df = read_processed(processed_path)
    preview = df.head(num_rows).copy()
    num_total_rows, num_total_columns = df.shape
    del df

    return preview, int(num_total_rows), int(num_total_columns)


@st.cache_data
def _load_single_sample(
    processed_path: str,
    artifacts_dir: str,
    row_position: int,
) -> tuple:
    """Prepare one processed test row exactly as the inference path does.

    Purpose:
        Tab 2's sample selector. The row is taken from the held-out test split
        (``test_indices.pkl``) and converted by ``prepare_model_ready_data`` —
        the project's single tensor-preparation function (SDS §14.2) — so the
        dashboard's input is byte-identical to what evaluation feeds the model.

    Args:
        processed_path: Path to the processed dataset file.
        artifacts_dir: ``config["paths"]["artifacts_dir"]``.
        row_position: Position within the test split, not a raw DataFrame index.

    Returns:
        tuple: ``(X_sample, true_class_name, raw_row)`` where ``X_sample`` has
            shape ``(1, num_features, 1)``.

    Raises:
        FileNotFoundError: If the processed dataset or any artifact is missing.
        IndexError: If ``row_position`` is outside the test split.
    """
    test_indices, feature_columns, label_encoder, class_mapping = (
        _load_artifacts(artifacts_dir)
    )

    test_indices = np.asarray(test_indices)
    if row_position < 0 or row_position >= len(test_indices):
        raise IndexError(row_position)

    if not os.path.exists(processed_path):
        raise FileNotFoundError(processed_path)

    selected_index = np.asarray([test_indices[row_position]])
    df = read_processed(processed_path)
    raw_row = df.loc[selected_index, feature_columns + ["label"]].copy()

    # Imported, never reimplemented (SDS §14.2 / §20).
    X_sample, _ = prepare_model_ready_data(
        df,
        selected_index,
        feature_columns,
        label_encoder,
        len(class_mapping),
    )
    true_class_name = str(df.loc[selected_index[0], "label"])
    del df

    return X_sample, true_class_name, raw_row


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

_PROJECT_DESCRIPTION = (
    "A simulation-only Industrial IoT intrusion detection system. A single "
    "CNN-GRU architecture is trained two ways — federated (Flower + FedAvg "
    "across virtual clients holding an IID partition) and centralized on the "
    "full training split — so the two training paradigms can be compared "
    "side by side on identical held-out data. Predictions are explained with "
    "SHAP. This dashboard is strictly read-only: it displays artifacts "
    "produced by the experiment scripts and never trains, partitions, or "
    "reprocesses anything."
)


def render_overview_tab(config: dict) -> None:
    """Render Tab 1: project overview and dataset statistics.

    Purpose:
        Display the project title and description, the dataset statistics
        (name, classes, features, train/test sample counts) read from the
        persisted preprocessing artifacts, and the two saved figures: the class
        distribution plot and the federated architecture diagram.

    Args:
        config: The project configuration.

    Returns:
        None
    """
    st.header("Project Overview")

    try:
        artifacts_dir = config["paths"]["artifacts_dir"]
        results_dir = config["paths"]["results_dir"]
        models_dir = config["paths"]["models_dir"]

        st.subheader(config["dashboard"]["title"])
        st.write(_PROJECT_DESCRIPTION)

        _, feature_columns, _, class_mapping = _load_artifacts(artifacts_dir)
        num_training_samples, num_test_samples = _load_split_sizes(
            artifacts_dir
        )

        st.subheader("Project statistics")

        top_row = st.columns(3)
        top_row[0].metric("Dataset", config["dataset"]["name"])
        # num_classes is always len(class_mapping) — never hardcoded (§6/§22).
        top_row[1].metric("Classes", len(class_mapping))
        top_row[2].metric("Features", len(feature_columns))

        bottom_row = st.columns(3)
        bottom_row[0].metric("Training samples", f"{num_training_samples:,}")
        bottom_row[1].metric("Test samples", f"{num_test_samples:,}")
        bottom_row[2].metric(
            "Total samples",
            f"{num_training_samples + num_test_samples:,}",
        )

        with st.expander("Class mapping"):
            st.json(class_mapping)

        st.subheader("Class distribution")
        distribution_path = os.path.join(results_dir, "class_distribution.png")
        if os.path.exists(distribution_path):
            st.image(
                distribution_path,
                caption="Sample count per attack category",
            )
        else:
            # A missing figure degrades this section only — the statistics
            # above stay visible rather than the whole tab going dark.
            st.warning(
                "class_distribution.png not found. "
                "Run experiments/run_preprocessing.py to generate it."
            )

        st.subheader("Federated architecture")
        federated_architecture_path = os.path.join(
            models_dir, "federated_architecture.png"
        )
        if os.path.exists(federated_architecture_path):
            st.image(
                federated_architecture_path,
                caption="CNN-GRU architecture (federated global model)",
            )
        else:
            st.warning(
                "federated_architecture.png not found. "
                "Run experiments/run_federated.py to generate it."
            )

    except FileNotFoundError:
        st.error(
            "Preprocessing artifacts not found. "
            "Run experiments/run_preprocessing.py first."
        )
    except Exception as exc:  # pragma: no cover - defensive UI guard
        st.error(f"Overview tab could not render: {exc}")


def render_explorer_tab(config: dict) -> None:
    """Render Tab 2: dataset preview and per-sample federated prediction.

    Purpose:
        Show the processed dataset's preview, shape, feature list and class
        count, then let the user select one held-out test row, prepare it
        through ``prepare_model_ready_data`` exactly as inference does, and run
        it through the saved federated global model. No retraining occurs.

    Args:
        config: The project configuration.

    Returns:
        None
    """
    st.header("Dataset & Model Explorer")

    try:
        models_dir = config["paths"]["models_dir"]
        artifacts_dir = config["paths"]["artifacts_dir"]
        processed_path = os.path.join(
            config["paths"]["processed_data_dir"],
            config["paths"]["processed_data_file"],
        )

        test_indices, feature_columns, _, class_mapping = _load_artifacts(
            artifacts_dir
        )
        preview, num_total_rows, num_total_columns = _load_dataset_preview(
            processed_path
        )

        st.subheader("Dataset preview")
        st.dataframe(preview, use_container_width=True)

        shape_row = st.columns(3)
        shape_row[0].metric("Rows", f"{num_total_rows:,}")
        shape_row[1].metric("Columns", num_total_columns)
        shape_row[2].metric("Classes", len(class_mapping))
        st.caption(
            f"Processed dataset shape: "
            f"({num_total_rows:,}, {num_total_columns})"
        )

        with st.expander(f"Feature list ({len(feature_columns)} features)"):
            st.dataframe(
                pd.DataFrame(
                    {"feature": feature_columns},
                    index=range(1, len(feature_columns) + 1),
                ),
                use_container_width=True,
            )

        if not _scaler_is_available(artifacts_dir):
            # Reported, not applied: the processed dataset on disk is already
            # scaled with this artifact, so transforming again would corrupt
            # the sample before it reaches the model.
            st.warning(
                "scaler.pkl not found in outputs/artifacts/. The processed "
                "dataset is already scaled, so predictions below are "
                "unaffected."
            )

        st.subheader("Sample selector")
        num_test_samples = int(len(np.asarray(test_indices)))
        st.caption(
            f"Rows are drawn from the held-out test split "
            f"({num_test_samples:,} rows)."
        )

        # Shared across tabs via session_state so Tab 4 explains the sample
        # selected here (SDS §18, Tab 4).
        selected_position = st.number_input(
            "Test split row position",
            min_value=0,
            max_value=num_test_samples - 1,
            value=int(st.session_state.get("sample_index", 0)),
            step=1,
        )
        selected_position = int(selected_position)
        st.session_state["sample_index"] = selected_position

        model_path = os.path.join(models_dir, "federated_global_model_keras215_v2.h5")
        model = _load_model(model_path)

        # Prepared through the project's single tensor-preparation function,
        # so this input matches the evaluation pipeline's exactly.
        X_sample, true_class_name, raw_row = _load_single_sample(
            processed_path, artifacts_dir, selected_position
        )

        probabilities = model.predict(X_sample, verbose=0)[0]
        predicted_index = int(np.argmax(probabilities))

        # Reverse lookup with keys cast to int (SDS §18, Tab 2).
        reverse_mapping = {int(k): v for k, v in class_mapping.items()}
        predicted_class_name = reverse_mapping[predicted_index]

        st.subheader("Prediction")
        prediction_row = st.columns(3)
        prediction_row[0].metric("Predicted class", predicted_class_name)
        prediction_row[1].metric("True class", true_class_name)
        prediction_row[2].metric(
            "Confidence",
            f"{float(probabilities[predicted_index]):.4f}",
        )

        st.subheader("Prediction probabilities")
        probability_frame = pd.DataFrame(
            {"probability": probabilities},
            index=[
                reverse_mapping[index] for index in range(len(probabilities))
            ],
        )
        st.bar_chart(probability_frame)
        st.dataframe(
            probability_frame.sort_values("probability", ascending=False),
            use_container_width=True,
        )

        with st.expander("Selected row (processed feature values)"):
            st.dataframe(raw_row.T, use_container_width=True)

    except FileNotFoundError:
        st.error(
            "Trained model or test data not found. "
            "Run experiments/run_federated.py and "
            "experiments/run_preprocessing.py first."
        )
    except Exception as exc:  # pragma: no cover - defensive UI guard
        st.error(f"Dataset & Model Explorer tab could not render: {exc}")


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

        model_path = os.path.join(models_dir, "federated_global_model_keras215_v2.h5")
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


@st.cache_resource
def _load_inference_bundle(artifacts_dir: str) -> dict:
    """Load the fitted encoders/scaler/class map used by the prediction tab.

    Purpose:
        Cached wrapper over ``load_inference_artifacts`` so the pickles are
        read once per session rather than once per upload. ``artifacts_dir`` is
        the cache key, which is why it is passed explicitly rather than read
        from the config inside the function.

    Args:
        artifacts_dir: ``config["paths"]["artifacts_dir"]``.

    Returns:
        dict: The bundle described in ``src.inference.predict``.

    Raises:
        FileNotFoundError: If any required artifact is missing.
    """
    return load_inference_artifacts(_load_config())


@st.cache_data(show_spinner=False)
def _predict_uploaded_file(
    file_bytes: bytes,
    artifacts_dir: str,
    model_path: str,
) -> tuple:
    """Predict an uploaded CSV once, cached on the file's own bytes.

    Purpose:
        Selecting a row to explain triggers a Streamlit rerun, which would
        otherwise re-predict the entire uploaded file on every click. Keying
        the cache on the raw bytes means re-uploading the same file is free and
        uploading a different one is correctly recomputed.

        The prepared tensor is returned alongside the results so the SHAP step
        explains *the exact array the model scored*, rather than recomputing a
        second one that could silently differ.

    Args:
        file_bytes: The uploaded file's raw content, used as the cache key.
        artifacts_dir: ``config["paths"]["artifacts_dir"]``.
        model_path: Path to the persisted federated global model.

    Returns:
        tuple: ``(results, report, X)`` from ``predict_dataframe``.

    Raises:
        InvalidInputError: If the upload cannot be mapped onto the model input.
        FileNotFoundError: If the model or any artifact is missing.
    """
    import io

    artifacts = _load_inference_bundle(artifacts_dir)
    model = _load_model(model_path)
    frame = pd.read_csv(io.BytesIO(file_bytes), low_memory=False)

    # The same entry point the non-cached path used — no second prediction
    # implementation.
    return predict_dataframe(frame, model, artifacts, return_features=True)


def _render_live_prediction_explanation(
    config: dict,
    results: pd.DataFrame,
    X_uploaded: np.ndarray,
    artifacts: dict,
) -> None:
    """Render "Why did the model make this prediction?" for one uploaded row.

    Purpose:
        Explain a single prediction from the uploaded file using SHAP. The
        explainer is built from the persisted
        ``outputs/artifacts/shap_background.npy`` via the same cached
        ``_build_explainer`` helper Tab 4 uses, so the background distribution
        is never re-sampled and the two tabs explain against an identical
        reference (SDS §14.10 / §18 / §22).

        Contributions are shown for the **predicted class**, signed: positive
        values are the evidence that pushed the model toward the class it
        chose, negative values are evidence against it. A mean-across-classes
        magnitude would answer a different question and would discard the sign
        that makes the chart readable.

        Missing SHAP artifacts degrade this section alone — the predictions and
        the download button above stay usable.

    Args:
        config: The project configuration.
        results: The per-row prediction table from ``predict_dataframe``.
        X_uploaded: The prepared tensor the model scored, row-aligned with
            ``results``.
        artifacts: The inference bundle, supplying feature order and classes.

    Returns:
        None
    """
    st.divider()
    st.subheader("Why did the model make this prediction?")

    background_path = os.path.join(
        config["paths"]["artifacts_dir"], "shap_background.npy"
    )
    model_path = os.path.join(
        config["paths"]["models_dir"], FEDERATED_GLOBAL_MODEL_FILENAME
    )

    if not os.path.exists(background_path):
        st.warning(
            "SHAP background not found at "
            f"'{background_path}'. Run experiments/run_explainability.py to "
            "generate it — predictions above are unaffected."
        )
        return

    row_options = results["row"].tolist()
    selected_row = st.selectbox(
        "Select a predicted row to explain",
        options=row_options,
        index=0,
        key="live_prediction_explain_row",
        help="Row numbers match the uploaded CSV's data lines.",
    )

    # Positional lookup: results is row-aligned with X_uploaded, so the
    # position in the table is the position in the tensor.
    position = int(results.index[results["row"] == selected_row][0])
    sample = X_uploaded[position: position + 1]

    predicted_class = str(results.loc[position, "predicted_class"])
    confidence = float(results.loc[position, "confidence"])
    status = str(results.loc[position, "status"])
    predicted_class_index = artifacts["class_names"].index(predicted_class)

    detail_row = st.columns(4)
    detail_row[0].metric("Row", int(selected_row))
    detail_row[1].metric("Predicted class", predicted_class)
    detail_row[2].metric("Confidence", f"{confidence:.4f}")
    detail_row[3].metric("Status", status)

    top_n = st.slider(
        "Number of features to show",
        min_value=10,
        max_value=20,
        value=15,
        key="live_prediction_top_n",
    )

    try:
        with st.spinner("Computing SHAP explanation..."):
            # Cached across reruns and built from the persisted background —
            # never re-sampled here.
            explainer = _build_explainer(model_path, background_path)
            model = _load_model(model_path)

            # The project's existing single-sample SHAP entry point.
            shap_values = explain_single_prediction(model, explainer, sample)

        contributions = get_top_feature_contributions(
            shap_values,
            sample,
            artifacts["feature_columns"],
            predicted_class_index,
            top_n=top_n,
        )
    except FileNotFoundError:
        st.warning(
            "SHAP artifacts not found. "
            "Run experiments/run_explainability.py first."
        )
        return
    except Exception as exc:  # pragma: no cover - defensive UI guard
        st.warning(f"SHAP explanation could not be computed: {exc}")
        return

    contribution_frame = pd.DataFrame(contributions)

    st.caption(
        f"Signed SHAP contributions toward **{predicted_class}**. Positive "
        f"values pushed the model toward this class; negative values pushed "
        f"against it."
    )

    # Ascending so the largest magnitude lands at the top of a horizontal bar.
    chart_frame = contribution_frame.set_index("feature")[
        ["shap_value"]
    ].iloc[::-1]
    st.bar_chart(chart_frame, horizontal=True)

    st.dataframe(
        contribution_frame[
            ["feature", "feature_value", "shap_value", "direction"]
        ].style.format(
            {"feature_value": "{:.4f}", "shap_value": "{:+.6f}"}
        ),
        use_container_width=True,
    )

    strongest = contribution_frame.iloc[0]
    st.info(
        f"The strongest single factor was **{strongest['feature']}** "
        f"(value {strongest['feature_value']:.4f}), which "
        f"{strongest['direction']} the model's confidence in "
        f"**{predicted_class}**."
    )

    st.download_button(
        "Download this explanation as CSV",
        data=contribution_frame.to_csv(index=False).encode("utf-8"),
        file_name=f"explanation_row_{int(selected_row)}.csv",
        mime="text/csv",
    )


def render_live_prediction_tab(config: dict) -> None:
    """Render Tab 5: batch prediction on a user-uploaded traffic CSV.

    Purpose:
        The project's user-facing inference workflow. An uploaded CSV is
        validated against the expected input structure, transformed by the
        *already fitted* encoders and scaler, and classified by the persisted
        federated global model. Per-row class, confidence and Normal/Attack
        status are displayed and offered as a CSV download.

        Nothing here trains, fits, or writes to ``outputs/``: the encoders come
        from ``outputs/artifacts/`` and the weights from the persisted
        federated global model, exactly as the other tabs' do.

        The tab also explains one selected prediction with SHAP, using the
        persisted background array — see
        :func:`_render_live_prediction_explanation`.

    Args:
        config: The project configuration.

    Returns:
        None
    """
    st.header("Live Prediction")

    try:
        artifacts_dir = config["paths"]["artifacts_dir"]
        models_dir = config["paths"]["models_dir"]

        artifacts = _load_inference_bundle(artifacts_dir)
        expected_columns = get_expected_input_columns(artifacts)

        st.write(
            "Upload a network-traffic CSV in the raw Edge-IIoTset column "
            "format. Each row is passed through the project's existing "
            "preprocessing artifacts and the federated global CNN-GRU model."
        )

        summary_row = st.columns(3)
        summary_row[0].metric("Required columns", len(expected_columns))
        summary_row[1].metric("Model features", len(artifacts["feature_columns"]))
        summary_row[2].metric("Classes", artifacts["num_classes"])

        with st.expander(f"Expected input columns ({len(expected_columns)})"):
            st.caption(
                "Any additional columns — including `Attack_type` and "
                "`Attack_label` — are ignored, not rejected."
            )
            st.dataframe(
                pd.DataFrame(
                    {"column": expected_columns},
                    index=range(1, len(expected_columns) + 1),
                ),
                use_container_width=True,
            )

        uploaded_file = st.file_uploader(
            "Network traffic CSV",
            type=["csv"],
            key="live_prediction_upload",
        )

        if uploaded_file is None:
            st.info("Upload a CSV file to generate predictions.")
            return

        model_path = os.path.join(models_dir, FEDERATED_GLOBAL_MODEL_FILENAME)
        model = _load_model(model_path)

        uploaded_frame = pd.read_csv(uploaded_file, low_memory=False)

        # Cached on the uploaded bytes, so choosing a row to explain re-runs
        # the SHAP step only — not the prediction over the whole file.
        with st.spinner(f"Predicting {len(uploaded_frame):,} row(s)..."):
            results, report, X_uploaded = _predict_uploaded_file(
                uploaded_file.getvalue(), artifacts_dir, model_path
            )

        st.subheader("Summary")
        attack_count = int((results["status"] == "Attack").sum())
        normal_count = int((results["status"] == "Normal").sum())

        metrics_row = st.columns(4)
        metrics_row[0].metric("Rows uploaded", f"{report['num_input_rows']:,}")
        metrics_row[1].metric("Rows predicted", f"{report['num_predicted_rows']:,}")
        metrics_row[2].metric("Normal", f"{normal_count:,}")
        metrics_row[3].metric("Attack", f"{attack_count:,}")

        if report["invalid_row_numbers"]:
            skipped = report["invalid_row_numbers"]
            preview = ", ".join(str(number) for number in skipped[:10])
            suffix = "" if len(skipped) <= 10 else ", ..."
            st.warning(
                f"{len(skipped)} row(s) were skipped because a required "
                f"numeric value was missing or unparseable. Row number(s): "
                f"{preview}{suffix}"
            )

        if report["ignored_columns"]:
            st.caption(
                f"{len(report['ignored_columns'])} uploaded column(s) were not "
                f"model inputs and were ignored."
            )

        st.subheader("Predictions")
        st.dataframe(
            results.style.format({"confidence": "{:.4f}"}),
            use_container_width=True,
        )

        st.subheader("Predicted class distribution")
        st.bar_chart(results["predicted_class"].value_counts())

        st.download_button(
            "Download predictions as CSV",
            data=results.to_csv(index=False).encode("utf-8"),
            file_name="predictions.csv",
            mime="text/csv",
        )

        _render_live_prediction_explanation(
            config, results, X_uploaded, artifacts
        )

    except InvalidInputError as exc:
        # The uploader's own mistake, phrased for the uploader.
        st.error(str(exc))
    except FileNotFoundError:
        st.error(
            "Preprocessing artifacts or the federated global model were not "
            "found. Run experiments/run_preprocessing.py and "
            "experiments/run_federated.py first."
        )
    except Exception as exc:  # pragma: no cover - defensive UI guard
        st.error(f"Live Prediction tab could not render: {exc}")


def main() -> None:
    """Render the dashboard's five tabs."""

    # MUST be the first Streamlit command
    st.set_page_config(
        page_title="IIoT Federated IDS Dashboard",
        layout="wide",
    )

    # Load configuration AFTER set_page_config()
    config = _load_config()

    st.title(config["dashboard"]["title"])

    overview, explorer, comparison, explainability, live_prediction = st.tabs(
        [
            "Project Overview",
            "Dataset & Model Explorer",
            "FL vs. Centralized",
            "Explainability",
            "Live Prediction",
        ]
    )

    # Tab 1
    with overview:
        render_overview_tab(config)

    # Tab 2
    with explorer:
        render_explorer_tab(config)

    # Tab 3
    with comparison:
        render_comparison_tab(config)

    # Tab 4
    with explainability:
        render_explainability_tab(config)

    # Tab 5
    with live_prediction:
        render_live_prediction_tab(config)


if __name__ == "__main__":
    main()
