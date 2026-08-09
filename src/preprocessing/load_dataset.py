"""Raw dataset loading, column dropping, and label derivation for the IIoT project.

Ownership rule (SDS Section 14.1): ``create_labels`` is the sole, authoritative
owner of native-column removal (Attack_type, Attack_label) throughout the entire
codebase. No other function, in any phase, may drop either column.
"""

import logging
import os

import numpy as np
import pandas as pd


def load_raw_dataset(raw_path: str) -> pd.DataFrame:
    """Read the raw Edge-IIoTset CSV file into a pandas DataFrame.

    No column transformations happen here; the caller passes an
    already-assembled path.

    Args:
        raw_path: Path to the raw CSV (``data/raw/edge_iiotset.csv`` by
            default).

    Returns:
        pd.DataFrame: The fully-loaded raw dataset.

    Raises:
        FileNotFoundError: If ``raw_path`` does not exist on disk.
    """
    if not os.path.exists(raw_path):
        raise FileNotFoundError(
            f"Raw dataset not found at: '{raw_path}'"
        )

    df = pd.read_csv(raw_path, low_memory=False)

    # Downcasting halves peak RAM: the raw CSV inflates to ~3-4 GB as a
    # default-dtype frame versus ~1.5-2 GB at float32/int32. No meaningful
    # precision is lost — these are packet counts, byte lengths, ports, and
    # flow statistics under 7 significant digits, all subsequently MinMax-scaled
    # into [0, 1] for a float32 Keras graph. Object columns are left for
    # fit_categorical_transformer.
    float64_cols = df.select_dtypes("float64").columns
    if len(float64_cols):
        df[float64_cols] = df[float64_cols].astype("float32")

    int64_cols = df.select_dtypes("int64").columns
    if len(int64_cols):
        df[int64_cols] = df[int64_cols].astype("int32")

    return df



def drop_identifier_columns(
    df: pd.DataFrame,
    columns_to_drop: list,
) -> pd.DataFrame:
    """Remove identifier and high-cardinality columns from the DataFrame.

    Never drops ``Attack_type`` or ``Attack_label`` — those native label-source
    columns are reserved exclusively for ``create_labels`` (SDS Section 14.1,
    Contract Invariant A.2). Absent columns are ignored.

    Args:
        df: Input DataFrame (the raw loaded dataset).
        columns_to_drop: Column names to drop, from
            ``config["dataset"]["drop_columns"]``.

    Returns:
        pd.DataFrame: New DataFrame with the columns removed; ``df`` is not
            mutated in place.
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

    The sole, authoritative function that derives ``label`` / ``binary_label``
    and drops the two native source columns, so they cannot leak into model
    features downstream. No other function in the codebase may do either.

    ``binary_label`` is derived from ``target_column``, not copied from
    ``raw_binary_column``; the two are then cross-checked and any disagreement
    is logged as a WARNING with the target-derived value kept as authoritative.

    Args:
        df: DataFrame still containing both ``target_column`` and
            ``raw_binary_column``.
        target_column: Multi-class native label column (``"Attack_type"``).
        raw_binary_column: Native binary label column (``"Attack_label"``).
        normal_class_value: Value in ``target_column`` meaning benign traffic
            (``"Normal"``), matched exactly and case-sensitively.
        logger: Optional logger from the calling ``experiments/run_*.py``
            script. Falls back to a module logger for the WARNING path only.

    Returns:
        pd.DataFrame: New DataFrame with ``label`` (string) and ``binary_label``
            (0 = normal, 1 = attack) added, and both native columns dropped.

    Raises:
        KeyError: If either native column is missing from ``df``.
    """
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

    df["label"] = df[target_column].values

    df["binary_label"] = np.where(
        df[target_column] == normal_class_value, 0, 1
    )

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

    # Sole native-column removal point in the entire pipeline.
    df = df.drop(columns=[target_column, raw_binary_column])

    return df
