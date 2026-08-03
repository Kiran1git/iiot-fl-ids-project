"""Raw dataset loading, column dropping, and label derivation for the IIoT project.

This module owns the first three steps of the preprocessing pipeline:
  1. load_raw_dataset   — read the Edge-IIoTset CSV into a DataFrame.
  2. drop_identifier_columns — remove high-cardinality / identifier columns.
  3. create_labels      — derive ``label`` and ``binary_label`` and drop the
                          native source columns (Attack_type, Attack_label).

Ownership rule (SDS Section 14.1):
  create_labels is the sole, authoritative owner of native-column removal
  (Attack_type, Attack_label) throughout the entire codebase. No other
  function, in any phase, may drop either column.
"""

import logging
import os

import numpy as np
import pandas as pd


def load_raw_dataset(raw_path: str) -> pd.DataFrame:
    """Read the raw Edge-IIoTset CSV file into a pandas DataFrame.

    Purpose:
        Open the CSV at ``raw_path`` and return its contents as a DataFrame.
        No column transformations are performed here — the caller is
        responsible for passing an already-assembled path built via
        ``os.path.join(config["paths"]["raw_data_dir"],
        config["paths"]["raw_data_file"])``.

    Args:
        raw_path: Absolute or relative path to the raw CSV file
            (``data/raw/edge_iiotset.csv`` by default).

    Returns:
        pd.DataFrame: The fully-loaded raw dataset.

    Raises:
        FileNotFoundError: If ``raw_path`` does not exist on disk.

    Dependencies:
        pandas, src.utils.logger (logger is passed by the calling script;
        this function does not construct its own logger).
    """
    if not os.path.exists(raw_path):
        raise FileNotFoundError(
            f"Raw dataset not found at: '{raw_path}'"
        )

    df = pd.read_csv(raw_path, low_memory=False)
    return df


def drop_identifier_columns(
    df: pd.DataFrame,
    columns_to_drop: list,
) -> pd.DataFrame:
    """Remove identifier and high-cardinality columns from the DataFrame.

    Purpose:
        Drop every column named in ``columns_to_drop`` from ``df``.
        Columns absent from ``df`` are silently ignored (``errors="ignore"``).
        This function NEVER drops ``Attack_type`` or ``Attack_label`` — those
        native label-source columns are reserved exclusively for
        ``create_labels`` (SDS Section 14.1, Contract Invariant A.2).

    Args:
        df: Input DataFrame (the raw loaded dataset).
        columns_to_drop: List of column names to drop, sourced from
            ``config["dataset"]["drop_columns"]``.

    Returns:
        pd.DataFrame: New DataFrame with the specified columns removed;
            the original ``df`` is not mutated in place.

    Raises:
        Nothing. Missing columns are silently ignored via ``errors="ignore"``.

    Dependencies:
        pandas.
    """
    return df.drop(columns=columns_to_drop, errors="ignore")


def create_labels(
    df: pd.DataFrame,
    target_column: str,
    raw_binary_column: str,
    normal_class_value: str,
    logger: logging.Logger = None,
) -> pd.DataFrame:
    """Derive label columns and remove native source columns from the DataFrame.

    Purpose:
        This is the **sole, authoritative** function that:
          1. Creates ``label`` — an exact copy of ``df[target_column]`` (the
             multi-class string label, e.g. ``"Normal"``, ``"DDoS_HTTP"``, ...).
          2. Creates ``binary_label`` — derived independently from
             ``target_column``:  0 where ``df[target_column] ==
             normal_class_value`` (exact, case-sensitive), else 1.
          3. Validates agreement between the derived ``binary_label`` and the
             existing ``raw_binary_column``. If any row disagrees, a WARNING
             is logged (not raised) and the target-derived value is kept.
          4. Drops both ``target_column`` (Attack_type) and
             ``raw_binary_column`` (Attack_label) from the returned DataFrame,
             so they cannot leak into model features downstream.

        No other function in the entire codebase may drop either native
        column, derive ``label``, or derive ``binary_label``.

    Args:
        df: DataFrame that must still contain both ``target_column`` and
            ``raw_binary_column`` (called immediately after
            ``drop_identifier_columns``, which never removes either).
        target_column: Name of the multi-class native label column
            (= ``config["dataset"]["target_column"]``, i.e. ``"Attack_type"``).
        raw_binary_column: Name of the native binary label column
            (= ``config["dataset"]["raw_binary_column"]``,
            i.e. ``"Attack_label"``).
        normal_class_value: String value in ``target_column`` that represents
            normal (benign) traffic
            (= ``config["dataset"]["normal_class_value"]``, i.e. ``"Normal"``).
        logger: Optional pre-constructed ``logging.Logger`` instance passed
            down from the calling ``experiments/run_*.py`` script.
            If ``None``, a fallback module-level logger is used for the
            WARNING path only.

    Returns:
        pd.DataFrame: New DataFrame with:
            - ``label`` column added (multi-class string).
            - ``binary_label`` column added (int: 0 = normal, 1 = attack).
            - ``target_column`` and ``raw_binary_column`` both dropped.

    Raises:
        KeyError: If ``target_column`` or ``raw_binary_column`` is not present
            in ``df``.

    Dependencies:
        pandas, numpy, src.utils.logger.
    """
    # Validate that required source columns are present
    if target_column not in df.columns:
        raise KeyError(
            f"target_column '{target_column}' not found in DataFrame columns."
        )
    if raw_binary_column not in df.columns:
        raise KeyError(
            f"raw_binary_column '{raw_binary_column}' not found in DataFrame columns."
        )

    _log = logger if logger is not None else logging.getLogger(__name__)

    df = df.copy()

    # Step 1: derive label (exact copy of target_column)
    df["label"] = df[target_column].values

    # Step 2: derive binary_label from target_column (not from raw_binary_column)
    df["binary_label"] = np.where(
        df[target_column] == normal_class_value, 0, 1
    )

    # Step 3: validate agreement with raw_binary_column
    agreement_mask = df["binary_label"] == df[raw_binary_column].astype(int)
    mismatch_count = (~agreement_mask).sum()
    if mismatch_count > 0:
        _log.warning(
            "binary_label agreement check: %d row(s) differ between the "
            "target-derived binary_label and '%s'. "
            "The target-derived value is kept as authoritative.",
            mismatch_count,
            raw_binary_column,
        )

    # Step 4: drop both native source columns (sole removal point in pipeline)
    df = df.drop(columns=[target_column, raw_binary_column])

    return df
