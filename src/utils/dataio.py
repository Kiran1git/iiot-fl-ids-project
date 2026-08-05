"""Processed-dataset I/O helpers for the IIoT Federated IDS project.

This module owns the single, authoritative pair of read/write functions for the
processed dataset. Every consumer of ``data/processed/`` goes through here, so
the on-disk format is a configuration decision rather than a code change spread
across four call sites.

Why this module exists
----------------------
The project originally persisted the processed dataset as a 643 MB CSV. That
format has three concrete costs:

  1. CSV is text. Every float is re-parsed from ASCII on every load, which
     dominates the start-up time of ``train_baseline``, ``server_app``, and the
     evaluation pipeline.
  2. CSV carries no dtype information, so ``pandas.read_csv`` re-infers every
     column and defaults to float64 — double the memory the model actually
     needs, since Keras casts to float32 anyway.
  3. The federated simulation loads the same file in the parent process and
     then ships slices of it into four Ray actors, so any load-time bloat is
     paid more than once.

Parquet fixes all three: it is columnar, stores dtypes in the schema, and
compresses well (expect roughly a 5-10x on-disk reduction for this dataset).

Format dispatch is by file extension, taken from
``config["paths"]["processed_data_file"]``. Setting that key back to a
``.csv`` filename restores the original behaviour with no code change, so this
is a backwards-compatible optimisation rather than a migration.
"""

import os

import pandas


__all__ = ["read_processed", "write_processed"]


_PARQUET_EXTENSIONS = (".parquet", ".pq")
_CSV_EXTENSIONS = (".csv",)


def _dispatch_extension(path: str) -> str:
    """Return the normalised file extension used to pick a storage backend.

    Args:
        path: Path to a processed-dataset file.

    Returns:
        str: The lower-cased extension including the leading dot.

    Raises:
        ValueError: If the extension is neither a Parquet nor a CSV extension.
    """
    extension = os.path.splitext(path)[1].lower()

    if extension in _PARQUET_EXTENSIONS or extension in _CSV_EXTENSIONS:
        return extension

    raise ValueError(
        f"Unsupported processed-data format '{extension}' for path '{path}'. "
        f"Supported extensions: {_PARQUET_EXTENSIONS + _CSV_EXTENSIONS}."
    )


def read_processed(path: str, columns: list = None) -> pandas.DataFrame:
    """Load the processed dataset, dispatching on the file extension.

    Purpose:
        The single read entry point for ``data/processed/``. Used by
        ``train_baseline.train_centralized_model``,
        ``server_app.run_federated_simulation``,
        ``compare_fl_vs_centralized.run_evaluation_pipeline``, and
        ``experiments/run_preprocessing.py``'s plotting step.

    Args:
        path: Path to the processed dataset file. Built by the caller via
            ``os.path.join(config["paths"]["processed_data_dir"],
            config["paths"]["processed_data_file"])``.
        columns: Optional subset of columns to read. Honoured natively by the
            Parquet reader, which skips the other columns entirely rather than
            reading and discarding them — this is the main reason the
            evaluation and plotting paths can avoid materialising the full
            frame. Ignored-but-applied for CSV (pandas reads then subsets).

    Returns:
        pandas.DataFrame: The processed dataset, or the requested column
            subset of it.

    Raises:
        FileNotFoundError: If ``path`` does not exist on disk.
        ValueError: If the file extension is not a supported format.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Processed dataset not found at: '{path}'. "
            f"Run experiments/run_preprocessing.py first."
        )

    extension = _dispatch_extension(path)

    if extension in _PARQUET_EXTENSIONS:
        # Parquet is columnar: passing `columns` avoids reading the other
        # columns off disk at all, rather than reading then discarding.
        return pandas.read_parquet(path, columns=columns)

    frame = pandas.read_csv(path, low_memory=False)
    if columns is not None:
        frame = frame[columns]
    return frame


def write_processed(df: pandas.DataFrame, path: str) -> None:
    """Persist the processed dataset, dispatching on the file extension.

    Purpose:
        The single write entry point for ``data/processed/``. Called once, by
        ``run_preprocessing_pipeline``'s persistence step.

    Args:
        df: The fully processed (encoded, scaled) DataFrame.
        path: Destination path. The parent directory is created if absent.

    Returns:
        None

    Raises:
        ValueError: If the file extension is not a supported format.
        ImportError: If a Parquet path is requested but neither ``pyarrow`` nor
            ``fastparquet`` is installed. The message names the fix explicitly
            rather than surfacing pandas' generic engine error.
    """
    extension = _dispatch_extension(path)

    parent_directory = os.path.dirname(path)
    if parent_directory:
        os.makedirs(parent_directory, exist_ok=True)

    if extension in _PARQUET_EXTENSIONS:
        try:
            # snappy: fast to decompress, which matters because this file is
            # read far more often than it is written (once per preprocessing
            # run, but once per training run, per evaluation run, and once per
            # federated simulation).
            df.to_parquet(path, index=False, compression="snappy")
        except ImportError as exc:
            raise ImportError(
                "Writing Parquet requires the 'pyarrow' package. Install it "
                "with 'pip install pyarrow', or set "
                "paths.processed_data_file in configs/config.yaml back to a "
                "'.csv' filename to use the CSV backend instead."
            ) from exc
        return

    df.to_csv(path, index=False)
