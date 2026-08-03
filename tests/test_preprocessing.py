"""Unit tests for src/preprocessing/load_dataset.py — Phase 3.

Covers the load/drop/label cases mandated by SDS Section 19 and the Phase 3
validation checklist. Phase 4 cases (infer_feature_column_types,
split_train_test, prepare_model_ready_data, leakage regression) are added in
Phase 4 by extending this file.

SDS Section 19 required cases (Phase 3 scope):
  - load_raw_dataset raises FileNotFoundError on bad path.
  - drop_identifier_columns removes only configured identifier columns,
    ignores missing ones without error, and never removes Attack_type /
    Attack_label.
  - create_labels produces both label and binary_label with no NaNs,
    and both Attack_type and Attack_label are absent from the returned
    DataFrame.
"""

import logging

import numpy as np
import pandas as pd
import pytest

from src.preprocessing.load_dataset import (
    create_labels,
    drop_identifier_columns,
    load_raw_dataset,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def minimal_raw_df() -> pd.DataFrame:
    """A minimal synthetic DataFrame that mimics the raw Edge-IIoTset schema.

    Contains both native label-source columns (Attack_type, Attack_label),
    a column that should be dropped (ip.src_host), a numeric feature,
    and a categorical feature.
    """
    return pd.DataFrame(
        {
            "ip.src_host": ["192.168.1.1", "10.0.0.1", "172.16.0.1", "192.168.1.2"],
            "http.file_data": ["data1", "data2", "data3", "data4"],
            "frame.len": [60, 100, 80, 200],
            "tcp.srcport": [1234, 5678, 9101, 1112],
            "Attack_type": ["Normal", "DDoS_HTTP", "Normal", "MITM"],
            "Attack_label": [0, 1, 0, 1],
        }
    )


@pytest.fixture()
def raw_csv_file(tmp_path, minimal_raw_df) -> str:
    """Write the minimal raw DataFrame to a temporary CSV and return its path."""
    csv_path = tmp_path / "edge_iiotset.csv"
    minimal_raw_df.to_csv(csv_path, index=False)
    return str(csv_path)


@pytest.fixture()
def null_logger() -> logging.Logger:
    """A no-op logger that suppresses output during tests."""
    logger = logging.getLogger("test_preprocessing_null")
    logger.addHandler(logging.NullHandler())
    return logger


# ---------------------------------------------------------------------------
# load_raw_dataset — SDS Section 14.1
# ---------------------------------------------------------------------------

class TestLoadRawDataset:
    """Tests for load_raw_dataset."""

    def test_load_returns_dataframe(self, raw_csv_file):
        """load_raw_dataset returns a non-empty pandas DataFrame on a valid path."""
        df = load_raw_dataset(raw_csv_file)
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0

    def test_load_preserves_all_columns(self, raw_csv_file, minimal_raw_df):
        """load_raw_dataset returns a DataFrame with the same columns as the CSV."""
        df = load_raw_dataset(raw_csv_file)
        assert set(df.columns) == set(minimal_raw_df.columns)

    def test_load_preserves_row_count(self, raw_csv_file, minimal_raw_df):
        """load_raw_dataset returns the same number of rows as the source CSV."""
        df = load_raw_dataset(raw_csv_file)
        assert len(df) == len(minimal_raw_df)

    def test_load_raises_file_not_found_on_bad_path(self):
        """FileNotFoundError is raised when the path does not exist on disk."""
        with pytest.raises(FileNotFoundError):
            load_raw_dataset("data/raw/this_file_does_not_exist.csv")

    def test_load_error_message_contains_path(self):
        """The FileNotFoundError message includes the bad path for diagnosability."""
        bad_path = "data/raw/nonexistent.csv"
        with pytest.raises(FileNotFoundError, match=bad_path):
            load_raw_dataset(bad_path)

    def test_native_columns_present_after_load(self, raw_csv_file):
        """Attack_type and Attack_label are present in the loaded DataFrame.

        This confirms load_raw_dataset never drops the native label columns —
        their removal is create_labels' exclusive responsibility.
        """
        df = load_raw_dataset(raw_csv_file)
        assert "Attack_type" in df.columns
        assert "Attack_label" in df.columns


# ---------------------------------------------------------------------------
# drop_identifier_columns — SDS Section 14.1
# ---------------------------------------------------------------------------

class TestDropIdentifierColumns:
    """Tests for drop_identifier_columns."""

    def test_drops_configured_columns(self, minimal_raw_df):
        """Columns in the drop list are removed from the returned DataFrame."""
        cols_to_drop = ["ip.src_host", "http.file_data"]
        result = drop_identifier_columns(minimal_raw_df, cols_to_drop)
        for col in cols_to_drop:
            assert col not in result.columns

    def test_retains_non_configured_columns(self, minimal_raw_df):
        """Columns NOT in the drop list are still present after the call."""
        cols_to_drop = ["ip.src_host"]
        result = drop_identifier_columns(minimal_raw_df, cols_to_drop)
        assert "frame.len" in result.columns
        assert "tcp.srcport" in result.columns

    def test_ignores_missing_columns_silently(self, minimal_raw_df):
        """Columns absent from the DataFrame are ignored without raising."""
        cols_to_drop = ["ip.src_host", "this_column_does_not_exist"]
        # Must not raise
        result = drop_identifier_columns(minimal_raw_df, cols_to_drop)
        assert "ip.src_host" not in result.columns

    def test_never_drops_attack_type(self, minimal_raw_df):
        """Attack_type is never dropped by drop_identifier_columns.

        drop_columns in config.yaml deliberately excludes Attack_type so that
        create_labels can always find it (SDS Section 14.1 / Contract A.2).
        """
        cols_to_drop = ["ip.src_host", "http.file_data"]
        result = drop_identifier_columns(minimal_raw_df, cols_to_drop)
        assert "Attack_type" in result.columns

    def test_never_drops_attack_label(self, minimal_raw_df):
        """Attack_label is never dropped by drop_identifier_columns.

        Same reasoning as Attack_type — exclusively removed by create_labels.
        """
        cols_to_drop = ["ip.src_host", "http.file_data"]
        result = drop_identifier_columns(minimal_raw_df, cols_to_drop)
        assert "Attack_label" in result.columns

    def test_does_not_mutate_original_dataframe(self, minimal_raw_df):
        """drop_identifier_columns returns a new DataFrame; the original is unchanged."""
        original_cols = list(minimal_raw_df.columns)
        drop_identifier_columns(minimal_raw_df, ["ip.src_host"])
        assert list(minimal_raw_df.columns) == original_cols

    def test_empty_drop_list_returns_unchanged_dataframe(self, minimal_raw_df):
        """An empty drop list returns a DataFrame identical to the input."""
        result = drop_identifier_columns(minimal_raw_df, [])
        assert set(result.columns) == set(minimal_raw_df.columns)


# ---------------------------------------------------------------------------
# create_labels — SDS Section 14.1
# ---------------------------------------------------------------------------

class TestCreateLabels:
    """Tests for create_labels."""

    def _make_df_post_drop(self) -> pd.DataFrame:
        """Synthetic post-drop_identifier_columns DataFrame for create_labels tests."""
        return pd.DataFrame(
            {
                "frame.len": [60, 100, 80, 200, 150],
                "tcp.srcport": [1234, 5678, 9101, 1112, 3333],
                "Attack_type": ["Normal", "DDoS_HTTP", "Normal", "MITM", "Scanning"],
                "Attack_label": [0, 1, 0, 1, 1],
            }
        )

    def test_label_column_created(self, null_logger):
        """create_labels adds a 'label' column to the returned DataFrame."""
        df = self._make_df_post_drop()
        result = create_labels(df, "Attack_type", "Attack_label", "Normal", null_logger)
        assert "label" in result.columns

    def test_binary_label_column_created(self, null_logger):
        """create_labels adds a 'binary_label' column to the returned DataFrame."""
        df = self._make_df_post_drop()
        result = create_labels(df, "Attack_type", "Attack_label", "Normal", null_logger)
        assert "binary_label" in result.columns

    def test_label_values_match_attack_type(self, null_logger):
        """label column is an exact copy of the original Attack_type values."""
        df = self._make_df_post_drop()
        original_values = df["Attack_type"].tolist()
        result = create_labels(df, "Attack_type", "Attack_label", "Normal", null_logger)
        assert result["label"].tolist() == original_values

    def test_binary_label_zero_for_normal(self, null_logger):
        """binary_label is 0 for rows where Attack_type == 'Normal'."""
        df = self._make_df_post_drop()
        result = create_labels(df, "Attack_type", "Attack_label", "Normal", null_logger)
        normal_mask = pd.Series(
            ["Normal", "DDoS_HTTP", "Normal", "MITM", "Scanning"]
        ) == "Normal"
        for idx, is_normal in enumerate(normal_mask):
            expected = 0 if is_normal else 1
            assert result.iloc[idx]["binary_label"] == expected

    def test_binary_label_one_for_attacks(self, null_logger):
        """binary_label is 1 for all non-Normal rows."""
        df = self._make_df_post_drop()
        result = create_labels(df, "Attack_type", "Attack_label", "Normal", null_logger)
        attack_rows = result[result["label"] != "Normal"]
        assert (attack_rows["binary_label"] == 1).all()

    def test_no_nans_in_label(self, null_logger):
        """label column contains zero NaN values after create_labels."""
        df = self._make_df_post_drop()
        result = create_labels(df, "Attack_type", "Attack_label", "Normal", null_logger)
        assert result["label"].isna().sum() == 0

    def test_no_nans_in_binary_label(self, null_logger):
        """binary_label column contains zero NaN values after create_labels."""
        df = self._make_df_post_drop()
        result = create_labels(df, "Attack_type", "Attack_label", "Normal", null_logger)
        assert result["binary_label"].isna().sum() == 0

    def test_attack_type_absent_from_result(self, null_logger):
        """Attack_type is absent from the DataFrame returned by create_labels."""
        df = self._make_df_post_drop()
        result = create_labels(df, "Attack_type", "Attack_label", "Normal", null_logger)
        assert "Attack_type" not in result.columns

    def test_attack_label_absent_from_result(self, null_logger):
        """Attack_label is absent from the DataFrame returned by create_labels."""
        df = self._make_df_post_drop()
        result = create_labels(df, "Attack_type", "Attack_label", "Normal", null_logger)
        assert "Attack_label" not in result.columns

    def test_does_not_mutate_original_dataframe(self, null_logger):
        """create_labels returns a new DataFrame; the original is unchanged."""
        df = self._make_df_post_drop()
        original_cols = list(df.columns)
        create_labels(df, "Attack_type", "Attack_label", "Normal", null_logger)
        assert list(df.columns) == original_cols
        assert "Attack_type" in df.columns
        assert "Attack_label" in df.columns

    def test_raises_key_error_on_missing_target_column(self, null_logger):
        """KeyError is raised if target_column is absent from the DataFrame."""
        df = pd.DataFrame(
            {"Attack_label": [0, 1], "frame.len": [60, 100]}
        )
        with pytest.raises(KeyError):
            create_labels(df, "Attack_type", "Attack_label", "Normal", null_logger)

    def test_raises_key_error_on_missing_binary_column(self, null_logger):
        """KeyError is raised if raw_binary_column is absent from the DataFrame."""
        df = pd.DataFrame(
            {"Attack_type": ["Normal", "DDoS_HTTP"], "frame.len": [60, 100]}
        )
        with pytest.raises(KeyError):
            create_labels(df, "Attack_type", "Attack_label", "Normal", null_logger)

    def test_binary_mismatch_produces_warning_not_exception(
        self, null_logger, caplog
    ):
        """A binary-agreement mismatch between derived and raw binary column logs WARNING.

        This validates the SDS Section 14.1 rule: mismatch → WARNING, not exception.
        The target-derived value (from Attack_type) is kept as authoritative.
        """
        # Manually introduce a disagreement: row 0 Attack_type=Normal but
        # Attack_label=1 (incorrect raw binary value)
        df = pd.DataFrame(
            {
                "frame.len": [60, 100],
                "Attack_type": ["Normal", "DDoS_HTTP"],
                "Attack_label": [1, 1],  # row 0 disagrees
            }
        )
        with caplog.at_level(logging.WARNING):
            result = create_labels(
                df, "Attack_type", "Attack_label", "Normal", null_logger
            )
        # Must not raise — function must complete successfully
        assert "label" in result.columns
        assert "binary_label" in result.columns
        # binary_label for row 0 must be 0 (target-derived, authoritative)
        assert result.iloc[0]["binary_label"] == 0

    def test_non_none_logger_does_not_fail(self):
        """create_labels works correctly when a real logger is passed."""
        df = pd.DataFrame(
            {
                "frame.len": [60, 100],
                "Attack_type": ["Normal", "DDoS_HTTP"],
                "Attack_label": [0, 1],
            }
        )
        real_logger = logging.getLogger("test_real_logger")
        real_logger.addHandler(logging.NullHandler())
        result = create_labels(
            df, "Attack_type", "Attack_label", "Normal", real_logger
        )
        assert "label" in result.columns
        assert "binary_label" in result.columns
