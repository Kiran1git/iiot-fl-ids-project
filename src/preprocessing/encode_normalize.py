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
from sklearn.preprocessing import (
    LabelEncoder,
    MinMaxScaler,
    OneHotEncoder,
    OrdinalEncoder,
)

from tensorflow.keras.utils import to_categorical

from src.preprocessing.load_dataset import (
    create_labels,
    drop_identifier_columns,
    load_raw_dataset,
)
from src.utils.dataio import write_processed
from src.utils.logger import get_logger



# ---------------------------------------------------------------------------
# 1. infer_feature_column_types
# ---------------------------------------------------------------------------

def infer_feature_column_types(
    df: pd.DataFrame,
    label_columns: list = None,
    rule: str = "dtype_object",
    low_cardinality_max: int = 20,
) -> tuple:
    """Classify DataFrame columns into categorical and numeric groups.

    Purpose:
        Deterministically partition the feature columns of ``df``, excluding
        the label columns.  This is the single, authoritative column-
        classification function — no other module may independently
        reclassify columns.

        Two rules are implemented:

        ``"dtype_object"`` (legacy)
            Returns a 2-tuple ``(categorical_columns, numeric_columns)``.
            Every object/category column lands in one undifferentiated
            categorical group, which the caller then ordinal-encodes.

        ``"cardinality"`` (production)
            Returns a 3-tuple
            ``(low_cardinality_columns, high_cardinality_columns,
            numeric_columns)``.  Object/category columns are split on their
            distinct-value count in ``df``: at most ``low_cardinality_max``
            distinct values means the column is safe to one-hot encode, more
            than that means one-hot would explode the feature count and the
            column is routed to a frequency encoder instead.

            This split exists because ordinal-encoding a nominal column is a
            modelling error, not just a style choice: ``OrdinalEncoder`` maps
            categories to 0, 1, 2, ... and the Conv1D/GRU stack then reads
            those integers as a *magnitude*, inventing an ordering
            (``GET < POST < PUT``) that does not exist in the data.

        Because this function is called on ``df.loc[train_indices]`` by
        ``run_preprocessing_pipeline``, the cardinality counts are measured on
        training rows only — a test-set-only category can never influence how
        a column is classified.

    Args:
        df: DataFrame after ``drop_identifier_columns`` and ``create_labels``
            have run, so ``label`` and ``binary_label`` exist and
            ``Attack_type`` / ``Attack_label`` are absent.
        label_columns: Columns to exclude from every output list.
            Defaults to ``["label", "binary_label"]``.
        rule: ``"dtype_object"`` or ``"cardinality"``. Any other value raises
            ``ValueError``.
        low_cardinality_max: Distinct-value threshold separating the one-hot
            group from the frequency-encoded group. Read from
            ``config["dataset"]["low_cardinality_max"]``. Only used by the
            ``"cardinality"`` rule.

    Returns:
        tuple: For ``"dtype_object"``, a 2-tuple
            ``(categorical_columns, numeric_columns)``. For ``"cardinality"``,
            a 3-tuple ``(low_cardinality_columns, high_cardinality_columns,
            numeric_columns)``. In both cases the lists preserve the original
            DataFrame column order, are pairwise disjoint, and together cover
            every non-label column.

    Raises:
        ValueError: If ``rule`` is not a supported rule name.

    Dependencies:
        pandas.
    """
    if label_columns is None:
        label_columns = ["label", "binary_label"]

    if rule not in ("dtype_object", "cardinality"):
        raise ValueError(
            f"Unsupported rule '{rule}'. Supported rules: "
            f"'dtype_object', 'cardinality'."
        )

    feature_cols = [c for c in df.columns if c not in label_columns]
    categorical_columns = [
        c for c in feature_cols
        if df[c].dtype == object or str(df[c].dtype) == "category"
    ]
    numeric_columns = [
        c for c in feature_cols if c not in categorical_columns
    ]

    if rule == "dtype_object":
        return categorical_columns, numeric_columns

    low_cardinality_columns = []
    high_cardinality_columns = []
    for column in categorical_columns:
        if df[column].nunique(dropna=False) <= low_cardinality_max:
            low_cardinality_columns.append(column)
        else:
            high_cardinality_columns.append(column)

    return low_cardinality_columns, high_cardinality_columns, numeric_columns



# ---------------------------------------------------------------------------
# 2. fit_categorical_encoder
# ---------------------------------------------------------------------------

def fit_categorical_encoder(
    df: pd.DataFrame,
    categorical_columns: list,
) -> tuple:
    """Fit an OrdinalEncoder on categorical columns using only training rows.

    Purpose:
        Legacy single-group encoder, retained for the ``"dtype_object"`` rule
        and for the existing unit tests. Fits
        ``sklearn.preprocessing.OrdinalEncoder`` on the training-partition
        slice ``df.loc[train_indices]`` (slicing is the caller's
        responsibility — this function receives an already-sliced DataFrame).

        **Production preprocessing no longer uses this function.** Ordinal
        codes impose a false ordering on nominal features, so
        ``run_preprocessing_pipeline`` calls
        ``fit_categorical_transformer`` instead when
        ``categorical_column_rule`` is ``"cardinality"``.

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
# 2b. fit_categorical_transformer  (production encoder)
# ---------------------------------------------------------------------------

def fit_categorical_transformer(
    df: pd.DataFrame,
    low_cardinality_columns: list,
    high_cardinality_columns: list,
) -> dict:
    """Fit the production categorical encoders on training rows only.

    Purpose:
        Replace the single ``OrdinalEncoder`` with two encoders chosen to
        match what each column actually is:

        - **Low-cardinality columns → one-hot.** ``OrdinalEncoder`` turns
          ``http.request.method`` into 0-8, which the Conv1D kernel then reads
          as a continuous magnitude, implying ``GET < POST < PUT``. That
          ordering is fiction, and the convolution slides across adjacent
          feature positions, so the fiction propagates. One-hot removes the
          false metric entirely. Measured cost on this dataset is small: the
          six genuinely nominal columns have 9, 5, 13, 13, 3, and 3 distinct
          training values, so the expansion is roughly 46 columns.

        - **High-cardinality columns → frequency encoding.** One-hot on a
          column with thousands of levels would add thousands of near-empty
          columns. Frequency encoding maps each category to its *training*
          relative frequency, which is a genuinely ordered quantity (common
          vs. rare), so a numeric code is meaningful here rather than
          arbitrary. Categories absent from training map to 0.0, which is the
          honest encoding of "never seen during training".

        Both encoders are fit on the passed-in training slice only. The
        returned bundle is applied to the full dataset by
        ``apply_categorical_transformer``.

    Args:
        df: Training-slice DataFrame (``df.loc[train_indices]``).
        low_cardinality_columns: Columns to one-hot encode.
        high_cardinality_columns: Columns to frequency encode.

    Returns:
        dict: The fitted transformer bundle, persisted verbatim as
            ``categorical_encoder.pkl``. Keys:
              - ``"onehot_columns"``: the input ``low_cardinality_columns``.
              - ``"onehot_encoder"``: the fitted ``OneHotEncoder``, or ``None``
                when there are no low-cardinality columns.
              - ``"onehot_feature_names"``: the generated output column names,
                in the exact order the encoder emits them.
              - ``"frequency_columns"``: the input ``high_cardinality_columns``.
              - ``"frequency_maps"``: ``{column: {category: relative_freq}}``,
                computed on training rows only.

    Raises:
        Nothing under normal operation.

    Dependencies:
        sklearn.preprocessing.OneHotEncoder, pandas, numpy.
    """
    onehot_encoder = None
    onehot_feature_names: list = []

    if low_cardinality_columns:
        # sparse_output=False: the expansion here is ~46 columns, so a dense
        # block is small, and the downstream MinMaxScaler + reshape path
        # expects dense arrays anyway.
        # handle_unknown="ignore": a category seen only in the test split
        # encodes as an all-zero row rather than raising. That is the correct
        # semantics for "this category was never in training".
        onehot_encoder = OneHotEncoder(
            handle_unknown="ignore",
            sparse_output=False,
            dtype=np.float32,
        )
        onehot_encoder.fit(df[low_cardinality_columns].astype(str))
        onehot_feature_names = list(
            onehot_encoder.get_feature_names_out(low_cardinality_columns)
        )

    frequency_maps: dict = {}
    for column in high_cardinality_columns:
        # normalize=True gives relative frequency, so the encoding is
        # dataset-size independent and already lies in [0, 1].
        frequency_maps[column] = (
            df[column].astype(str).value_counts(normalize=True).to_dict()
        )

    return {
        "onehot_columns": list(low_cardinality_columns),
        "onehot_encoder": onehot_encoder,
        "onehot_feature_names": onehot_feature_names,
        "frequency_columns": list(high_cardinality_columns),
        "frequency_maps": frequency_maps,
    }


def apply_categorical_transformer(
    df: pd.DataFrame,
    transformer: dict,
) -> pd.DataFrame:
    """Apply a fitted categorical transformer bundle to a DataFrame.

    Purpose:
        The counterpart to ``fit_categorical_transformer``. Applied by
        ``run_preprocessing_pipeline`` to the **full** dataset using encoders
        fit on training rows only, so no test-set information ever reaches the
        fitted parameters.

        Original categorical columns are replaced: one-hot columns are dropped
        and their generated indicator columns appended; frequency columns are
        overwritten in place with their float frequency code.

    Args:
        df: The DataFrame to transform (typically the full dataset).
        transformer: The bundle returned by ``fit_categorical_transformer``.

    Returns:
        pd.DataFrame: The transformed DataFrame. Column order is
            ``[untouched columns..., one-hot indicator columns...]``, which is
            captured in ``feature_names.pkl`` and is therefore the single
            authoritative feature order for the rest of the project.

    Raises:
        KeyError: If a column named in the bundle is absent from ``df``.

    Dependencies:
        pandas, numpy.
    """
    df = df.copy()

    # Frequency columns are overwritten in place — no shape change.
    for column in transformer["frequency_columns"]:
        mapping = transformer["frequency_maps"][column]
        # Unseen categories -> 0.0: "never observed in training".
        df[column] = (
            df[column].astype(str).map(mapping).fillna(0.0).astype(np.float32)
        )

    onehot_columns = transformer["onehot_columns"]
    if onehot_columns and transformer["onehot_encoder"] is not None:
        encoded = transformer["onehot_encoder"].transform(
            df[onehot_columns].astype(str)
        )
        encoded_frame = pd.DataFrame(
            encoded,
            columns=transformer["onehot_feature_names"],
            index=df.index,
        )
        df = df.drop(columns=onehot_columns)
        df = pd.concat([df, encoded_frame], axis=1)

    return df



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
    # float32, not the pandas float64 default. Keras casts to float32 on the
    # way into the graph regardless, so building the tensor as float64 first
    # allocates exactly twice the RAM needed and then throws half of it away.
    # For the centralized training split (1.55M rows x ~95 features) this is
    # the difference between ~1.2 GB and ~590 MB, and the federated path pays
    # it once per Ray actor.
    X = (
        df.loc[indices, feature_columns]
        .to_numpy(dtype=np.float32, copy=False)
        .reshape(-1, len(feature_columns), 1)
    )
    y_int = label_encoder.transform(df.loc[indices, "label"])
    y = to_categorical(y_int, num_classes=num_classes).astype(np.float32)
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

    Execution order:
        1.  load_raw_dataset
        2.  drop_identifier_columns
        3.  create_labels  (sole native-column removal point)
        3b. drop_duplicates  (BEFORE the split — see below)
        4.  split_train_test  (on the unencoded, unscaled labeled DataFrame)
        5.  infer_feature_column_types  (on df.loc[train_indices] only)
        6.  fit categorical encoder(s)  (on df.loc[train_indices] only)
        7.  Apply categorical encoder(s) to the full dataset
        8.  Fit MinMaxScaler on train rows, apply to full dataset
        9.  fit_label_encoder  (on df.loc[train_indices, "label"] only)
        10. Persist the processed dataset (Parquet or CSV, per config)
        11. Persist all artifacts to outputs/artifacts/

    Two deliberate deviations from the original SDS §14.2 ordering, both of
    which close leakage paths rather than introduce them:

      - **De-duplication (3b) runs before the split.** ~13% of Edge-IIoTset
        rows are exact duplicates. Splitting first and de-duplicating later
        cannot undo the contamination, because the copies are already spread
        across both partitions.
      - **The split (4) now runs before column-type inference (5).** Inference
        measures per-column cardinality, which is a property fitted from data;
        measuring it on the full frame let test-set-only categories influence
        how a column was classified and encoded. Every fitted decision in the
        pipeline now sees training rows exclusively.


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

        # Step 3b — Drop exact duplicate rows BEFORE the split.
        # Order matters: de-duplicating after split_train_test would leave the
        # copies already distributed across both partitions, which is exactly
        # the contamination being removed. Measured at ~13% of raw rows, with
        # feature-only and feature+label duplicate rates identical, so no
        # duplicate group ever disagrees on its label and dropping is lossless
        # in information terms.
        if config["dataset"].get("drop_duplicates", False):
            rows_before = len(df)
            df = df.drop_duplicates().reset_index(drop=True)
            removed = rows_before - len(df)
            logger.info(
                "Step 3b: Removed %d duplicate row(s) (%.2f%% of %d) before "
                "splitting — prevents byte-identical training rows appearing "
                "in the test split. Remaining: %d rows.",
                removed,
                100.0 * removed / rows_before if rows_before else 0.0,
                rows_before,
                len(df),
            )
            logger.info(
                "Label distribution after de-duplication: %s",
                df["label"].value_counts().to_dict(),
            )
        else:
            logger.warning(
                "Step 3b: drop_duplicates is disabled. Duplicate rows will be "
                "split across train and test, inflating test accuracy."
            )

        # Step 4 — Train/test split (on unencoded, unscaled data).
        # The split now precedes column-type inference so that inference —
        # like every other fitted decision — sees training rows only.
        logger.info(
            "Step 4: Splitting data (test_size=%s, seed=%s)",
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

        # Step 5 — Infer feature column types on TRAINING ROWS ONLY.
        # Cardinality is measured on df.loc[train_indices], so a category that
        # appears only in the test split can never influence how a column is
        # classified or encoded.
        rule = config["dataset"]["categorical_column_rule"]
        low_cardinality_max = config["dataset"].get("low_cardinality_max", 20)
        logger.info(
            "Step 5: Inferring feature column types on %d training rows "
            "(rule='%s', low_cardinality_max=%s)",
            len(train_indices),
            rule,
            low_cardinality_max,
        )
        train_slice = df.loc[train_indices]

        if rule == "cardinality":
            (
                low_cardinality_columns,
                high_cardinality_columns,
                numeric_columns,
            ) = infer_feature_column_types(
                train_slice,
                label_columns=["label", "binary_label"],
                rule=rule,
                low_cardinality_max=low_cardinality_max,
            )
        else:
            categorical_columns, numeric_columns = infer_feature_column_types(
                train_slice,
                label_columns=["label", "binary_label"],
                rule=rule,
            )
            # The legacy rule ordinal-encodes everything as one group; express
            # that as "no one-hot group, all categoricals frequency-encoded"
            # would change semantics, so keep the legacy encoder path instead.
            low_cardinality_columns = []
            high_cardinality_columns = []

        if rule == "cardinality":
            logger.info(
                "Feature types — one-hot (low-cardinality): %d %s | "
                "frequency (high-cardinality): %d %s | numeric: %d",
                len(low_cardinality_columns),
                low_cardinality_columns,
                len(high_cardinality_columns),
                high_cardinality_columns,
                len(numeric_columns),
            )
        else:
            logger.info(
                "Feature types — categorical: %d, numeric: %d",
                len(categorical_columns),
                len(numeric_columns),
            )

        # Step 6 — Fit categorical encoders (train rows only)
        if rule == "cardinality":
            logger.info(
                "Step 6: Fitting OneHotEncoder (%d cols) + frequency encoder "
                "(%d cols) on %d training rows",
                len(low_cardinality_columns),
                len(high_cardinality_columns),
                len(train_indices),
            )
            cat_encoder = fit_categorical_transformer(
                train_slice,
                low_cardinality_columns,
                high_cardinality_columns,
            )
        else:
            logger.info(
                "Step 6: Fitting OrdinalEncoder on %d training rows (legacy "
                "'dtype_object' rule)",
                len(train_indices),
            )
            _, cat_encoder = fit_categorical_encoder(
                train_slice, categorical_columns
            )

        # Step 7 — Apply the categorical encoders to the FULL dataset.
        # Applied before the scaler is fit because one-hot changes the column
        # set, and the scaler must be fit on the final post-encoding columns.
        logger.info("Step 7: Applying categorical encoders to full dataset")
        if rule == "cardinality":
            df = apply_categorical_transformer(df, cat_encoder)
            # One-hot indicators are already 0/1 and frequency codes are
            # already in [0, 1]; scaling only the originally-numeric columns
            # keeps them that way and avoids pointless work.
            scale_columns = [c for c in numeric_columns if c in df.columns]
        else:
            if categorical_columns:
                df[categorical_columns] = cat_encoder.transform(
                    df[categorical_columns]
                )
            scale_columns = numeric_columns
        logger.info(
            "After categorical encoding: %d rows x %d columns",
            df.shape[0],
            df.shape[1],
        )

        # Step 8 — Fit scaler on training rows only, then apply to full data.
        logger.info(
            "Step 8: Fitting MinMaxScaler on %d training rows (%d columns)",
            len(train_indices),
            len(scale_columns),
        )
        scaler = MinMaxScaler()
        if scale_columns:
            scaler.fit(df.loc[train_indices, scale_columns])
            # Transform in place. The previous implementation did
            # `df_processed = df.copy()` first, which held two full copies of
            # a multi-GB frame simultaneously for no benefit — `df` is not
            # needed in its pre-transform state after this point.
            df[scale_columns] = scaler.transform(df[scale_columns]).astype(
                np.float32
            )

        # Step 9 — Fit label encoder (train rows only)
        logger.info(
            "Step 9: Fitting LabelEncoder on %d training labels",
            len(train_indices),
        )
        _, label_encoder, class_mapping = fit_label_encoder(
            df.loc[train_indices, "label"]
        )
        logger.info(
            "Class mapping (%d classes): %s", len(class_mapping), class_mapping
        )

        # binary_label was derived and validated by create_labels (its sole
        # owner, SDS §14.1) but nothing downstream trains on it: the project is
        # multi-class throughout. Carrying it into the processed file would
        # place a perfectly label-correlated column next to the features, one
        # `feature_columns` bug away from total leakage. Dropped here, after
        # the agreement check has already served its purpose.
        if "binary_label" in df.columns:
            df = df.drop(columns=["binary_label"])
            logger.info(
                "Dropped 'binary_label' after its agreement check — it is a "
                "direct function of the target and is never a model input."
            )

        feature_columns = [c for c in df.columns if c != "label"]
        logger.info("Feature count: %d columns", len(feature_columns))

        # Zero-NaN check before persisting
        nan_count = df[feature_columns].isna().sum().sum()
        if nan_count > 0:
            logger.warning(
                "Processed data contains %d NaN value(s) in feature columns.",
                nan_count,
            )

        # Step 10 — Persist the processed dataset
        logger.info(
            "Step 10: Saving processed dataset to '%s'", processed_csv_path
        )
        write_processed(df, processed_csv_path)
        logger.info(
            "Processed dataset saved (%d rows x %d columns).",
            len(df),
            df.shape[1],
        )
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
