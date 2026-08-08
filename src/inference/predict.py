"""User-facing batch inference for the IIoT Federated IDS project.

This module implements the read-only path

    user CSV -> existing preprocessing artifacts -> federated global CNN-GRU
             -> predicted class + confidence + Normal/Attack status

It is deliberately thin. Every fitted object it uses was produced by
``run_preprocessing_pipeline`` and persisted to ``outputs/artifacts/``, and the
model is the persisted federated global model. Nothing here fits, trains,
re-samples, or writes to ``outputs/`` or ``data/``.

Three rules this module exists to honour
----------------------------------------

1. **No second preprocessing pipeline (SDS §14.2).** The categorical encoding
   is applied by ``apply_categorical_transformer`` — the same function
   ``run_preprocessing_pipeline`` calls — using the same persisted bundle. The
   scaling is applied by the same persisted ``MinMaxScaler`` instance. The
   tensor reshape is done by ``prepare_model_features``, which is also what
   ``prepare_model_ready_data`` calls. There is no encoding, scaling or
   reshaping logic written out again here.

2. **Feature order comes from ``feature_names.pkl``, never from the upload.**
   The final frame is reindexed onto the persisted 88-column order before it
   reaches the model, so a user's column ordering is irrelevant and a silently
   transposed feature vector is impossible.

3. **The class mapping is loaded, never hardcoded.** ``num_classes`` is
   ``len(class_mapping)`` and the Normal/Attack split is decided by comparing
   the predicted class name against ``config["dataset"]["normal_class_value"]``.

Expected upload format
----------------------
A CSV in the *raw* Edge-IIoTset column format — the same columns the raw
capture has, minus nothing in particular. Only the columns the fitted encoders
actually consume are required (see :func:`get_expected_input_columns`); the
identifier columns dropped by ``config["dataset"]["drop_columns"]``, and the
``Attack_type`` / ``Attack_label`` columns if the user leaves them in, are
ignored rather than rejected.
"""

import json
import os
import pickle

import numpy as np
import pandas as pd

from src.preprocessing.encode_normalize import (
    apply_categorical_transformer,
    prepare_model_features,
)


#: The persisted federated global model consumed by every user-facing
#: prediction. Declared once here so the dashboard and this module cannot drift
#: onto two different checkpoints.
FEDERATED_GLOBAL_MODEL_FILENAME = "federated_global_model_keras215_v2.h5"


class InvalidInputError(ValueError):
    """Raised when an uploaded CSV cannot be mapped onto the model's inputs.

    Carries a message written for the person who uploaded the file, since it is
    surfaced directly in the dashboard rather than in a stack trace.
    """


def load_inference_artifacts(config: dict) -> dict:
    """Load every persisted artifact the prediction path needs.

    Purpose:
        Read the fitted encoders, the scaler, the feature order and the class
        mapping that ``run_preprocessing_pipeline`` already wrote. Nothing is
        fitted or regenerated here — a missing artifact is an error, never a
        trigger to recompute.

    Args:
        config: The fully merged configuration from ``load_config``.

    Returns:
        dict: The inference bundle, with keys ``feature_columns``,
            ``categorical_encoder``, ``scaler``, ``class_mapping``,
            ``class_names``, ``num_classes``, ``numeric_columns`` and
            ``normal_class_value``.

    Raises:
        FileNotFoundError: If any required artifact is absent from
            ``config["paths"]["artifacts_dir"]``.
    """
    artifacts_dir = config["paths"]["artifacts_dir"]

    def _read_pickle(filename: str):
        path = os.path.join(artifacts_dir, filename)
        if not os.path.exists(path):
            raise FileNotFoundError(path)
        with open(path, "rb") as handle:
            return pickle.load(handle)

    feature_columns = _read_pickle("feature_names.pkl")
    categorical_encoder = _read_pickle("categorical_encoder.pkl")
    scaler = _read_pickle("scaler.pkl")

    class_mapping_path = os.path.join(artifacts_dir, "class_mapping.json")
    if not os.path.exists(class_mapping_path):
        raise FileNotFoundError(class_mapping_path)
    with open(class_mapping_path, "r", encoding="utf-8") as handle:
        class_mapping = json.load(handle)

    num_classes = len(class_mapping)
    class_names = [class_mapping[str(i)] for i in range(num_classes)]

    # The columns the scaler was fit on are the columns it must be handed back,
    # in that same order. sklearn records them when it is fit on a DataFrame;
    # the fallback derives the same set from the persisted feature order for
    # scalers fit on a bare array.
    recorded = getattr(scaler, "feature_names_in_", None)
    if recorded is not None:
        numeric_columns = list(recorded)
    else:
        generated = set(categorical_encoder["onehot_feature_names"])
        frequency = set(categorical_encoder["frequency_columns"])
        numeric_columns = [
            column
            for column in feature_columns
            if column not in generated and column not in frequency
        ]

    return {
        "feature_columns": list(feature_columns),
        "categorical_encoder": categorical_encoder,
        "scaler": scaler,
        "class_mapping": class_mapping,
        "class_names": class_names,
        "num_classes": num_classes,
        "numeric_columns": numeric_columns,
        "normal_class_value": config["dataset"]["normal_class_value"],
    }


def get_expected_input_columns(artifacts: dict) -> list:
    """Return the raw column names an uploaded CSV must provide.

    Purpose:
        Define the upload contract from the fitted artifacts rather than from a
        hardcoded schema list, so it can never fall out of step with what the
        encoders actually consume. The set is the scaler's numeric columns plus
        the source columns of the categorical encoders — i.e. exactly the raw
        columns that survive ``drop_columns`` and feed the 88 model features.

    Args:
        artifacts: The bundle from :func:`load_inference_artifacts`.

    Returns:
        list[str]: Required raw column names, numeric columns first, then the
            one-hot sources, then the frequency-encoded sources.
    """
    categorical_encoder = artifacts["categorical_encoder"]
    return (
        list(artifacts["numeric_columns"])
        + list(categorical_encoder["onehot_columns"])
        + list(categorical_encoder["frequency_columns"])
    )


def validate_uploaded_dataframe(df: pd.DataFrame, artifacts: dict) -> tuple:
    """Check an uploaded frame against the expected input structure.

    Purpose:
        Reject an unusable file with a message the uploader can act on, and
        separate merely-unusable *rows* from an unusable *file*. A missing
        column is fatal for the whole upload; a row whose numeric cell cannot
        be parsed is dropped and reported, so one malformed line does not cost
        the user the other ten thousand predictions.

    Args:
        df: The DataFrame parsed from the uploaded CSV.
        artifacts: The bundle from :func:`load_inference_artifacts`.

    Returns:
        tuple: ``(valid_frame, report)``. ``valid_frame`` contains only the
            required columns, only the parseable rows, with numeric columns
            coerced to float and a ``RangeIndex``. ``report`` carries
            ``row_numbers`` (1-based line numbers in the uploaded CSV, aligned
            with ``valid_frame``), ``num_input_rows``, ``num_valid_rows``,
            ``invalid_row_numbers`` and ``ignored_columns``.

    Raises:
        InvalidInputError: If the frame has no rows, or if any required column
            is absent, or if every row fails numeric parsing.
    """
    if df is None or len(df) == 0:
        raise InvalidInputError("The uploaded CSV contains no data rows.")

    required_columns = get_expected_input_columns(artifacts)
    missing = [column for column in required_columns if column not in df.columns]
    if missing:
        preview = ", ".join(missing[:10])
        suffix = "" if len(missing) <= 10 else f", and {len(missing) - 10} more"
        raise InvalidInputError(
            f"The uploaded CSV is missing {len(missing)} required column(s) "
            f"out of {len(required_columns)}: {preview}{suffix}. The file must "
            f"use the raw Edge-IIoTset column names."
        )

    ignored_columns = [
        column for column in df.columns if column not in set(required_columns)
    ]

    working = df.loc[:, required_columns].copy()

    # 1-based line numbers as the user sees them in a spreadsheet, captured
    # before any row is dropped so the reported numbers still point at the
    # original file.
    row_numbers = np.arange(1, len(working) + 1)

    numeric_columns = list(artifacts["numeric_columns"])
    if numeric_columns:
        # errors="coerce" turns an unparseable cell into NaN instead of raising,
        # which is what lets a single bad row be isolated and reported rather
        # than aborting the whole upload.
        working[numeric_columns] = working[numeric_columns].apply(
            pd.to_numeric, errors="coerce"
        )
        valid_mask = working[numeric_columns].notna().all(axis=1).to_numpy()
    else:
        valid_mask = np.ones(len(working), dtype=bool)

    invalid_row_numbers = row_numbers[~valid_mask].tolist()

    if not valid_mask.any():
        raise InvalidInputError(
            f"None of the {len(working)} uploaded row(s) could be parsed: every "
            f"row has a missing or non-numeric value in at least one required "
            f"numeric column."
        )

    valid_frame = working.loc[valid_mask].reset_index(drop=True)

    report = {
        "row_numbers": row_numbers[valid_mask],
        "num_input_rows": int(len(working)),
        "num_valid_rows": int(len(valid_frame)),
        "invalid_row_numbers": invalid_row_numbers,
        "ignored_columns": ignored_columns,
    }

    return valid_frame, report


def prepare_uploaded_features(df: pd.DataFrame, artifacts: dict) -> np.ndarray:
    """Turn a validated upload into the model's input tensor.

    Purpose:
        Run the validated rows through the *already fitted* encoders and
        scaler, then through the project's single reshape function. This is the
        step that guarantees a user's row is transformed byte-identically to a
        training row: same encoder objects, same scaler object, same feature
        order, same reshape.

    Args:
        df: The validated frame from :func:`validate_uploaded_dataframe`.
        artifacts: The bundle from :func:`load_inference_artifacts`.

    Returns:
        numpy.ndarray: Shape ``(len(df), num_features, 1)``, ready for
            ``model.predict``.

    Raises:
        InvalidInputError: If the transformed frame does not produce every
            column named in ``feature_names.pkl``.
    """
    feature_columns = artifacts["feature_columns"]

    # Same function run_preprocessing_pipeline uses, same persisted bundle.
    # Unseen categories encode as an all-zero indicator block, exactly as they
    # do for a test-split row (handle_unknown="ignore").
    encoded = apply_categorical_transformer(df, artifacts["categorical_encoder"])

    numeric_columns = list(artifacts["numeric_columns"])
    if numeric_columns:
        # The persisted scaler, applied — never re-fit. Values outside the
        # training range land outside [0, 1] and are deliberately not clipped,
        # because that is what the evaluation path does too.
        encoded[numeric_columns] = artifacts["scaler"].transform(
            encoded[numeric_columns]
        ).astype(np.float32)

    missing = [
        column for column in feature_columns if column not in encoded.columns
    ]
    if missing:
        raise InvalidInputError(
            f"After preprocessing, {len(missing)} model feature(s) could not be "
            f"produced from the uploaded file (for example: "
            f"{', '.join(missing[:5])}). The upload does not match the "
            f"structure this model was trained on."
        )

    # Reindexed onto the persisted order before the reshape, so the upload's own
    # column order can never reach the model.
    ordered = encoded.loc[:, feature_columns]

    # The project's single reshape (SDS §14.2) — not reimplemented here.
    return prepare_model_features(
        ordered, np.arange(len(ordered)), feature_columns
    )


def predict_dataframe(
    df: pd.DataFrame,
    model,
    artifacts: dict,
    return_features: bool = False,
) -> tuple:
    """Predict an attack class for every valid row of an uploaded CSV.

    Purpose:
        The single entry point the dashboard calls. Validates, preprocesses,
        predicts, and formats the per-row result table.

    Args:
        df: The DataFrame parsed from the uploaded CSV.
        model: The loaded federated global Keras model.
        artifacts: The bundle from :func:`load_inference_artifacts`.
        return_features: When ``True``, also return the prepared input tensor.
            The SHAP explanation must run against *the very array the model
            scored*, not against a second, separately-recomputed one — if the
            two ever differed, the explanation would be describing a different
            input than the prediction it claims to explain. Returning the
            tensor is what makes that class of bug impossible rather than
            merely unlikely. Defaults to ``False`` so existing callers keep
            their two-value unpacking.

    Returns:
        tuple: ``(results, report)``, or ``(results, report, X)`` when
            ``return_features`` is ``True``. ``results`` is a DataFrame with
            columns ``row`` (1-based line number in the uploaded CSV),
            ``predicted_class``, ``confidence`` and ``status``
            (``"Normal"``/``"Attack"``). ``report`` is the validation report,
            extended with ``num_predicted_rows`` and the model input shape.
            ``X`` has shape ``(num_predicted_rows, num_features, 1)`` and is
            row-aligned with ``results``.

    Raises:
        InvalidInputError: If the upload cannot be mapped onto the model input.
    """
    valid_frame, report = validate_uploaded_dataframe(df, artifacts)
    X = prepare_uploaded_features(valid_frame, artifacts)

    probabilities = model.predict(X, verbose=0)

    predicted_indices = np.argmax(probabilities, axis=1)
    confidences = probabilities[np.arange(len(probabilities)), predicted_indices]

    class_names = artifacts["class_names"]
    predicted_classes = [class_names[int(index)] for index in predicted_indices]

    normal_class_value = artifacts["normal_class_value"]
    statuses = [
        "Normal" if name == normal_class_value else "Attack"
        for name in predicted_classes
    ]

    results = pd.DataFrame(
        {
            "row": report["row_numbers"],
            "predicted_class": predicted_classes,
            "confidence": confidences.astype(float),
            "status": statuses,
        }
    )

    report = dict(report)
    report["num_predicted_rows"] = int(len(results))
    report["model_input_shape"] = tuple(int(dim) for dim in X.shape)

    if return_features:
        return results, report, X

    return results, report
