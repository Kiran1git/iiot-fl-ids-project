"""Feature-type inference, encoding, scaling, splitting, and pipeline orchestration.

This module owns the complete second half of the preprocessing pipeline:
  - infer_feature_column_types  — sole classifier of categorical/numeric columns.
  - fit_categorical_encoder     — OrdinalEncoder, fit on training rows only.
  - fit_scaler                  — MinMaxScaler, fit on training rows only.
  - fit_label_encoder           — LabelEncoder + class_mapping, training rows only.
  - split_train_test            — single stratified 80/20 split; never repeated.
  - prepare_model_ready_data    — sole tensor-preparation function for the CNN-GRU.
  - run_preprocessing_pipeline  — top-level 11-step orchestrator.

Ownership rules (SDS Section 14.2):
  - infer_feature_column_types is the sole, authoritative column-classifier.
  - prepare_model_ready_data is the sole tensor-preparation function;
    every downstream module (centralized, federated, evaluation, explainability,
    dashboard) calls this function rather than reimplementing it.
  - All encoders/scalers are fit ONLY on df.loc[train_indices] — never on the
    full dataset (that would be a leakage defect).
  - The train/test split is computed once, persisted, and reused everywhere.
  - run_preprocessing_pipeline respects force_reprocess: false — a second run
    skips regeneration if the processed CSV already exists.
"""

import json
import logging
import os
import pickle

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split as sk_train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, OrdinalEncoder
from tensorflow.keras.utils import to_categorical

from src.preprocessing.load_dataset import (
    create_labels,
    drop_identifier_columns,
    load_raw_dataset,
)
from src.utils.config_loader import load_config
from src.utils.logger import get_logger
from src.utils.seed import set_global_seed


# ---------------------------------------------------------------------------
# 1. infer_feature_column_types
# ---------------------------------------------------------------------------

def infer_feature_column_types(
    df: pd.DataFrame,
    label_columns: list = None,
    rule: str = "dtype_object",
) -> tuple:
    """Classify DataFrame columns into categorical and numeric groups.

    Purpose:
        Deterministically partition the feature columns of ``df`` into
        ``categorical_columns`` (dtype object or category) and
        ``numeric_columns`` (any numeric dtype, including bool), excluding
        the label columns.  This is the single, authoritative column-
        classification function — no other module may independently
        reclassify columns.

    Args:
        df: DataFrame after ``drop_identifier_columns`` and ``create_labels``
            have run, so ``label`` and ``binary_label`` exist and
            ``Attack_type`` / ``Attack_label`` are absent.
        label_columns: Columns to exclude from both output lists.
            Defaults to ``["label", "binary_label"]``.
        rule: Classification strategy. Only ``"dtype_object"`` is implemented;
            any other value raises ``ValueError``.

    Returns:
        tuple[list[str], list[str]]: ``(categorical_columns, numeric_columns)``
            where each list contains column names (in their original DataFrame
            order) and the two lists are disjoint and together cover every
            non-label column.

    Raises:
        ValueError: If ``rule`` is not ``"dtype_object"``.

    Dependencies:
        pandas.
    """
    if label_columns is None:
        label_columns = ["label", "binary_label"]

    if rule != "dtype_object":
        raise ValueError(
            f"Unsupported rule '{rule}'. Only 'dtype_object' is implemented."
        )

    feature_cols = [c for c in df.columns if c not in label_columns]
    categorical_columns = [
        c for c in feature_cols
        if df[c].dtype == object or str(df[c].dtype) == "category"
    ]
    numeric_columns = [
        c for c in feature_cols if c not in categorical_columns
    ]
    return categorical_columns, numeric_columns


# ---------------------------------------------------------------------------
# 2. fit_categorical_encoder
# ---------------------------------------------------------------------------

def fit_categorical_encoder(
    df: pd.DataFrame,
    categorical_columns: list,
) -> tuple:
    """Fit an OrdinalEncoder on categorical columns using only training rows.

    Purpose:
        Fit ``sklearn.preprocessing.OrdinalEncoder`` on the training-partition
        slice ``df.loc[train_indices]`` (slicing is the caller's
        responsibility — this function receives an already-sliced DataFrame).
        Returns the encoded training-slice DataFrame and the fitted encoder.
        The fitted encoder is later applied to the full dataset by the
        orchestrator to avoid leakage.

    Args:
        df: Training-slice DataFrame (``df.loc[train_indices]``).
        categorical_columns: List of column names to encode.

    Returns:
        tuple[pd.DataFrame, OrdinalEncoder]: The training-slice DataFrame
            with categorical columns replaced by ordinal integers, and the
            fitted ``OrdinalEncoder``.

    Raises:
        ValueError: If ``categorical_columns`` is non-empty but the list
            passed does not match any columns in ``df``.

    Dependencies:
        sklearn.preprocessing.OrdinalEncoder, pandas.
    """
    encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    if categorical_columns:
        df = df.copy()
        df[categorical_columns] = encoder.fit_transform(df[categorical_columns])
    else:
        encoder.fit(df[[]])  # fit on empty to keep interface consistent
    return df, encoder


# ---------------------------------------------------------------------------
# 3. fit_scaler
# ---------------------------------------------------------------------------

def fit_scaler(
    df: pd.DataFrame,
    numeric_columns: list,
) -> tuple:
    """Fit a MinMaxScaler on numeric columns using only training rows.

    Purpose:
        Fit ``sklearn.preprocessing.MinMaxScaler`` on the training-partition
        slice (passed in by the orchestrator). Returns the scaled training-
        slice DataFrame and the fitted scaler.

    Args:
        df: Training-slice DataFrame (``df.loc[train_indices]``).
        numeric_columns: List of numeric column names to scale.

    Returns:
        tuple[pd.DataFrame, MinMaxScaler]: The training-slice DataFrame
            with numeric columns scaled to [0, 1], and the fitted
            ``MinMaxScaler``.

    Raises:
        Nothing under normal operation.

    Dependencies:
        sklearn.preprocessing.MinMaxScaler, pandas.
    """
    scaler = MinMaxScaler()
    df = df.copy()
    if numeric_columns:
        df[numeric_columns] = scaler.fit_transform(df[numeric_columns])
    return df, scaler


# ---------------------------------------------------------------------------
# 4. fit_label_encoder
# ---------------------------------------------------------------------------

def fit_label_encoder(
    labels: pd.Series,
) -> tuple:
    """Fit a LabelEncoder on the label column of the training partition.

    Purpose:
        Fit ``sklearn.preprocessing.LabelEncoder`` on the training rows'
        ``label`` column to produce integer class indices, and build a
        ``class_mapping`` dict whose keys are stringified integer class
        indices and whose values are class-name strings (required because
        JSON object keys are always strings).

    Args:
        labels: The ``label`` column of the training-partition slice
            (``df.loc[train_indices, "label"]``).

    Returns:
        tuple[numpy.ndarray, LabelEncoder, dict]: Integer-encoded labels for
            the training rows passed in, the fitted ``LabelEncoder``, and the
            ``class_mapping`` dict
            (``{str(class_index): class_name, ...}``).

    Raises:
        Nothing under normal operation.

    Dependencies:
        sklearn.preprocessing.LabelEncoder.
    """
    encoder = LabelEncoder()
    encoded_labels = encoder.fit_transform(labels)
    class_mapping = {
        str(idx): cls for idx, cls in enumerate(encoder.classes_)
    }
    return encoded_labels, encoder, class_mapping


# ---------------------------------------------------------------------------
# 5. split_train_test
# ---------------------------------------------------------------------------

def split_train_test(
    df: pd.DataFrame,
    label_column: str,
    test_size: float,
    seed: int,
) -> tuple:
    """Perform a single stratified train/test split on the labeled DataFrame.

    Purpose:
        Compute one stratified 80/20 (or ``test_size``-fraction) split on the
        labeled, **unencoded, unscaled** DataFrame — before any encoding or
        scaling has been fit or applied.  Returns row-index arrays into ``df``
        (not re-shuffled copies of the data) so the indices can be persisted
        and reused throughout the project without re-splitting.

    Args:
        df: Labeled DataFrame post ``drop_identifier_columns`` +
            ``create_labels``, before any encoding or scaling.
        label_column: Name of the stratification column (``"label"``).
        test_size: Fraction of data to reserve for testing
            (``config["dataset"]["test_size"]``).
        seed: Random seed for reproducibility
            (``config["seed"]``).

    Returns:
        tuple[numpy.ndarray, numpy.ndarray]: ``(train_indices, test_indices)``
            — arrays of integer row indices into ``df``.

    Raises:
        ValueError: If any class has fewer than 2 samples (stratification
            requires at least 2 samples per class).

    Dependencies:
        sklearn.model_selection.train_test_split.
    """
    indices = np.arange(len(df))
    stratify_labels = df[label_column].values

    # Guard: stratification requires at least 2 samples per class
    unique, counts = np.unique(stratify_labels, return_counts=True)
    if np.any(counts < 2):
        raise ValueError(
            "Some classes have fewer than 2 samples; stratified split is not "
            f"possible. Classes with < 2 samples: "
            f"{unique[counts < 2].tolist()}"
        )

    train_indices, test_indices = sk_train_test_split(
        indices,
        test_size=test_size,
        random_state=seed,
        stratify=stratify_labels,
    )
    return train_indices, test_indices


# ---------------------------------------------------------------------------
# 6. prepare_model_ready_data
# ---------------------------------------------------------------------------

def prepare_model_ready_data(
    df: pd.DataFrame,
    indices: np.ndarray,
    feature_columns: list,
    label_encoder: LabelEncoder,
    num_classes: int,
) -> tuple:
    """Convert processed tabular data into CNN-GRU-ready tensors.

    Purpose:
        This is the single, shared tensor-preparation function for the entire
        project. Every consumer — ``train_baseline.py``, ``client_app.py``,
        ``compare_fl_vs_centralized.py``, ``shap_utils.py``, and
        ``dashboard/app.py`` — calls this function rather than reimplementing
        the reshape and one-hot-encoding logic.

    Args:
        df: The fully processed (encoded, scaled) DataFrame, or a row-subset.
        indices: Array of row indices into ``df`` selecting the subset to use.
        feature_columns: Ordered list of feature column names, sourced from
            ``feature_names.pkl`` (the fixed, persisted order).
        label_encoder: Fitted ``sklearn.preprocessing.LabelEncoder`` loaded
            from ``label_encoder.pkl``.
        num_classes: Total number of classes, derived from
            ``len(class_mapping)`` (``class_mapping.json``). Never hardcoded.

    Returns:
        tuple[numpy.ndarray, numpy.ndarray]:
            - ``X``: shape ``(len(indices), len(feature_columns), 1)`` —
              the feature tensor reshaped for ``Conv1D`` input.
            - ``y``: shape ``(len(indices), num_classes)`` — one-hot-encoded
              integer class labels.

    Raises:
        ValueError: If any value in ``df.loc[indices, "label"]`` is unseen
            by ``label_encoder`` (not present in ``class_mapping.json``).

    Dependencies:
        numpy, tensorflow.keras.utils.to_categorical.
    """
    X = df.loc[indices, feature_columns].values.reshape(
        -1, len(feature_columns), 1
    )
    y_int = label_encoder.transform(df.loc[indices, "label"])
    y = to_categorical(y_int, num_classes=num_classes)
    return X, y


# ---------------------------------------------------------------------------
# 7. run_preprocessing_pipeline
# ---------------------------------------------------------------------------

def run_preprocessing_pipeline(config: dict) -> None:
    """Run the full 11-step preprocessing pipeline and persist all artifacts.

    Purpose:
        Top-level orchestrator that executes steps 1–11 in the exact fixed
        order specified in SDS Section 14.2.  Respects
        ``config["dataset"]["force_reprocess"]``: if ``False`` and the
        processed CSV already exists, logs an INFO message and returns
        immediately without regeneration.

    Exact 11-step execution order (must not be reordered — SDS §14.2):
        1.  load_raw_dataset
        2.  drop_identifier_columns
        3.  create_labels  (sole native-column removal point)
        4.  infer_feature_column_types
        5.  split_train_test  (on the unencoded, unscaled labeled DataFrame)
        6.  fit_categorical_encoder (on df.loc[train_indices] only)
        7.  fit_scaler            (on df.loc[train_indices] only)
        8.  fit_label_encoder     (on df.loc[train_indices, "label"] only)
        9.  Apply fitted encoder + scaler to the full dataset
        10. Persist processed CSV to data/processed/edge_iiotset_processed.csv
        11. Persist all artifacts to outputs/artifacts/

    Args:
        config: Fully loaded config dict from ``load_config()``.

    Returns:
        None

    Raises:
        Propagates any exception raised by the functions it calls, after
        logging the full stack trace at ERROR level.

    Dependencies:
        All functions in this file, src.preprocessing.load_dataset,
        src.utils.logger, src.utils.seed, pickle, json.
    """
    logs_dir = config["paths"]["logs_dir"]
    logger = get_logger("preprocessing", logs_dir)

    logger.info("=== run_preprocessing_pipeline starting ===")
    logger.info(
        "Config summary — seed=%s | raw=%s/%s | processed=%s/%s | "
        "force_reprocess=%s | test_size=%s | drop_columns=%s",
        config["seed"],
        config["paths"]["raw_data_dir"],
        config["paths"]["raw_data_file"],
        config["paths"]["processed_data_dir"],
        config["paths"]["processed_data_file"],
        config["dataset"]["force_reprocess"],
        config["dataset"]["test_size"],
        config["dataset"]["drop_columns"],
    )

    processed_csv_path = os.path.join(
        config["paths"]["processed_data_dir"],
        config["paths"]["processed_data_file"],
    )
    artifacts_dir = config["paths"]["artifacts_dir"]

    # ---- force_reprocess guard ----
    if not config["dataset"]["force_reprocess"] and os.path.exists(
        processed_csv_path
    ):
        logger.info(
            "Preprocessing skipped: '%s' already exists and "
            "force_reprocess=false. Reusing existing processed data.",
            processed_csv_path,
        )
        return

    # Ensure output directories exist
    os.makedirs(config["paths"]["processed_data_dir"], exist_ok=True)
    os.makedirs(artifacts_dir, exist_ok=True)

    try:
        # Step 1 — Load raw dataset
        raw_path = os.path.join(
            config["paths"]["raw_data_dir"],
            config["paths"]["raw_data_file"],
        )
        logger.info("Step 1: Loading raw dataset from '%s'", raw_path)
        df = load_raw_dataset(raw_path)
        logger.info(
            "Dataset loaded: %d rows x %d columns", df.shape[0], df.shape[1]
        )

        # Step 2 — Drop identifier columns
        logger.info(
            "Step 2: Dropping identifier columns: %s",
            config["dataset"]["drop_columns"],
        )
        df = drop_identifier_columns(df, config["dataset"]["drop_columns"])
        logger.info(
            "After drop: %d rows x %d columns", df.shape[0], df.shape[1]
        )

        # Step 3 — Create labels (sole native-column removal point)
        logger.info(
            "Step 3: Creating labels from '%s' / '%s' (normal='%s')",
            config["dataset"]["target_column"],
            config["dataset"]["raw_binary_column"],
            config["dataset"]["normal_class_value"],
        )
        df = create_labels(
            df,
            config["dataset"]["target_column"],
            config["dataset"]["raw_binary_column"],
            config["dataset"]["normal_class_value"],
            logger,
        )
        class_counts = df["label"].value_counts().to_dict()
        logger.info("Label distribution: %s", class_counts)

        # Step 4 — Infer feature column types
        logger.info(
            "Step 4: Inferring feature column types "
            "(rule='%s')",
            config["dataset"]["categorical_column_rule"],
        )
        categorical_columns, numeric_columns = infer_feature_column_types(
            df,
            label_columns=["label", "binary_label"],
            rule=config["dataset"]["categorical_column_rule"],
        )
        logger.info(
            "Feature types — categorical: %d, numeric: %d",
            len(categorical_columns),
            len(numeric_columns),
        )

        # Step 5 — Train/test split (on unencoded, unscaled data)
        logger.info(
            "Step 5: Splitting data (test_size=%s, seed=%s)",
            config["dataset"]["test_size"],
            config["seed"],
        )
        train_indices, test_indices = split_train_test(
            df,
            label_column="label",
            test_size=config["dataset"]["test_size"],
            seed=config["seed"],
        )
        logger.info(
            "Split: train=%d rows, test=%d rows",
            len(train_indices),
            len(test_indices),
        )

        # Step 6 — Fit categorical encoder (train rows only)
        logger.info(
            "Step 6: Fitting OrdinalEncoder on %d training rows",
            len(train_indices),
        )
        _, cat_encoder = fit_categorical_encoder(
            df.loc[train_indices], categorical_columns
        )

        # Step 7 — Fit scaler (train rows only)
        logger.info(
            "Step 7: Fitting MinMaxScaler on %d training rows",
            len(train_indices),
        )
        _, scaler = fit_scaler(df.loc[train_indices], numeric_columns)

        # Step 8 — Fit label encoder (train rows only)
        logger.info(
            "Step 8: Fitting LabelEncoder on %d training labels",
            len(train_indices),
        )
        _, label_encoder, class_mapping = fit_label_encoder(
            df.loc[train_indices, "label"]
        )
        logger.info(
            "Class mapping (%d classes): %s", len(class_mapping), class_mapping
        )

        # Step 9 — Apply fitted encoder + scaler to the full dataset
        logger.info("Step 9: Applying fitted transformers to full dataset")
        df_processed = df.copy()
        if categorical_columns:
            df_processed[categorical_columns] = cat_encoder.transform(
                df_processed[categorical_columns]
            )
        if numeric_columns:
            df_processed[numeric_columns] = scaler.transform(
                df_processed[numeric_columns]
            )

        # Determine feature columns (everything except label, binary_label)
        feature_columns = [
            c for c in df_processed.columns
            if c not in ("label", "binary_label")
        ]
        logger.info(
            "Feature count: %d columns", len(feature_columns)
        )

        # Zero-NaN check before persisting
        nan_count = df_processed[feature_columns].isna().sum().sum()
        if nan_count > 0:
            logger.warning(
                "Processed data contains %d NaN value(s) in feature columns.",
                nan_count,
            )

        # Step 10 — Persist processed CSV
        logger.info(
            "Step 10: Saving processed CSV to '%s'", processed_csv_path
        )
        df_processed.to_csv(processed_csv_path, index=False)
        logger.info("Processed CSV saved (%d rows).", len(df_processed))

        # Step 11 — Persist all artifacts
        logger.info("Step 11: Persisting all artifacts to '%s'", artifacts_dir)

        def _pkl(obj, fname):
            path = os.path.join(artifacts_dir, fname)
            with open(path, "wb") as fh:
                pickle.dump(obj, fh)
            logger.info("  Saved %s", path)

        _pkl(scaler, "scaler.pkl")
        _pkl(label_encoder, "label_encoder.pkl")
        _pkl(cat_encoder, "categorical_encoder.pkl")
        _pkl(train_indices, "train_indices.pkl")
        _pkl(test_indices, "test_indices.pkl")
        _pkl(feature_columns, "feature_names.pkl")

        class_mapping_path = os.path.join(artifacts_dir, "class_mapping.json")
        with open(class_mapping_path, "w", encoding="utf-8") as fh:
            json.dump(class_mapping, fh, indent=2)
        logger.info("  Saved %s", class_mapping_path)

        logger.info("=== run_preprocessing_pipeline complete ===")

    except Exception:
        logger.error(
            "run_preprocessing_pipeline failed — full traceback:", exc_info=True
        )
        raise
