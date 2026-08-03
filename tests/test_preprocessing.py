"""Unit tests for src/preprocessing/ — Phases 3 and 4.

Phase 3 cases (load_dataset.py):
  - load_raw_dataset raises FileNotFoundError on bad path.
  - drop_identifier_columns removes only configured identifier columns,
    ignores missing ones without error, and never removes Attack_type /
    Attack_label.
  - create_labels produces both label and binary_label with no NaNs,
    and both Attack_type and Attack_label are absent from the returned
    DataFrame.

Phase 4 cases (encode_normalize.py — SDS Section 19):
  - infer_feature_column_types correctly separates a mixed dummy DataFrame
    into categorical/numeric lists using rule="dtype_object", excludes
    label/binary_label from both, and raises ValueError for any other rule.
  - split_train_test produces non-overlapping index sets whose union covers
    the full dataset.
  - prepare_model_ready_data produces X of shape (n, num_features, 1) and
    one-hot y of shape (n, num_classes) for a small dummy processed DataFrame.
  - Leakage regression: encoders/scaler fit only on a training slice produce
    different parameters than if fit on the full dataset.
"""

import logging

import numpy as np
import pandas as pd
import pytest

from sklearn.preprocessing import LabelEncoder

from src.preprocessing.load_dataset import (
    create_labels,
    drop_identifier_columns,
    load_raw_dataset,
)
from src.preprocessing.encode_normalize import (
    fit_categorical_encoder,
    fit_label_encoder,
    fit_scaler,
    infer_feature_column_types,
    prepare_model_ready_data,
    split_train_test,
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


# ===========================================================================
# Phase 4 test cases — encode_normalize.py (SDS Section 19)
# ===========================================================================

# ---------------------------------------------------------------------------
# Shared fixture for Phase 4 tests
# ---------------------------------------------------------------------------

@pytest.fixture()
def labeled_df() -> pd.DataFrame:
    """Synthetic post-create_labels DataFrame for Phase 4 tests.

    Mimics the state of the DataFrame after Phase 3 functions have run:
    - Attack_type and Attack_label are absent.
    - label and binary_label are present.
    - One categorical column (proto) and two numeric columns (frame.len, port).
    - 30 rows (10 per class) so a stratified 80/20 split yields 6 test rows,
      which satisfies sklearn's requirement of >= num_classes (3) in test set.
    """
    n_per_class = 10
    protos = (["tcp", "udp"] * 15)[:n_per_class * 3]
    frame_lens = list(range(60, 60 + n_per_class * 3))
    ports = [80, 443, 8080] * n_per_class
    labels = (
        ["Normal"] * n_per_class
        + ["DDoS_HTTP"] * n_per_class
        + ["MITM"] * n_per_class
    )
    binary_labels = [0] * n_per_class + [1] * (n_per_class * 2)
    return pd.DataFrame(
        {
            "proto": protos,
            "frame.len": frame_lens,
            "port": ports,
            "label": labels,
            "binary_label": binary_labels,
        }
    )



# ---------------------------------------------------------------------------
# infer_feature_column_types — SDS Section 14.2
# ---------------------------------------------------------------------------

class TestInferFeatureColumnTypes:
    """Tests for infer_feature_column_types."""

    def test_returns_two_lists(self, labeled_df):
        """infer_feature_column_types returns a tuple of two lists."""
        result = infer_feature_column_types(labeled_df)
        assert isinstance(result, tuple)
        assert len(result) == 2
        cat_cols, num_cols = result
        assert isinstance(cat_cols, list)
        assert isinstance(num_cols, list)

    def test_categorical_column_identified(self, labeled_df):
        """Object-dtype column 'proto' is classified as categorical."""
        cat_cols, _ = infer_feature_column_types(labeled_df)
        assert "proto" in cat_cols

    def test_numeric_columns_identified(self, labeled_df):
        """Numeric columns 'frame.len' and 'port' are classified as numeric."""
        _, num_cols = infer_feature_column_types(labeled_df)
        assert "frame.len" in num_cols
        assert "port" in num_cols

    def test_label_excluded_from_both_lists(self, labeled_df):
        """'label' is excluded from both categorical and numeric output lists."""
        cat_cols, num_cols = infer_feature_column_types(labeled_df)
        assert "label" not in cat_cols
        assert "label" not in num_cols

    def test_binary_label_excluded_from_both_lists(self, labeled_df):
        """'binary_label' is excluded from both output lists."""
        cat_cols, num_cols = infer_feature_column_types(labeled_df)
        assert "binary_label" not in cat_cols
        assert "binary_label" not in num_cols

    def test_lists_are_disjoint(self, labeled_df):
        """Categorical and numeric lists share no column names."""
        cat_cols, num_cols = infer_feature_column_types(labeled_df)
        assert len(set(cat_cols) & set(num_cols)) == 0

    def test_union_covers_all_feature_columns(self, labeled_df):
        """Union of both lists covers every non-label feature column."""
        cat_cols, num_cols = infer_feature_column_types(labeled_df)
        expected = {c for c in labeled_df.columns
                    if c not in ("label", "binary_label")}
        actual = set(cat_cols) | set(num_cols)
        assert actual == expected

    def test_raises_value_error_for_unknown_rule(self, labeled_df):
        """ValueError is raised when rule is not 'dtype_object'."""
        with pytest.raises(ValueError, match="dtype_object"):
            infer_feature_column_types(labeled_df, rule="pearson_correlation")

    def test_default_rule_is_dtype_object(self, labeled_df):
        """Default call (no rule arg) uses dtype_object and succeeds."""
        # Must not raise
        cat_cols, num_cols = infer_feature_column_types(labeled_df)
        assert isinstance(cat_cols, list)
        assert isinstance(num_cols, list)

    def test_custom_label_columns_respected(self, labeled_df):
        """Custom label_columns argument is honoured."""
        # Treat 'binary_label' as a feature instead of a label
        cat_cols, num_cols = infer_feature_column_types(
            labeled_df, label_columns=["label"]
        )
        # binary_label is numeric (int), so it should appear in num_cols
        assert "binary_label" in num_cols


# ---------------------------------------------------------------------------
# split_train_test — SDS Section 14.2
# ---------------------------------------------------------------------------

class TestSplitTrainTest:
    """Tests for split_train_test."""

    def test_returns_two_arrays(self, labeled_df):
        """split_train_test returns a tuple of two numpy arrays."""
        train_idx, test_idx = split_train_test(labeled_df, "label", 0.2, 42)
        assert isinstance(train_idx, np.ndarray)
        assert isinstance(test_idx, np.ndarray)

    def test_union_covers_full_dataset(self, labeled_df):
        """train_indices ∪ test_indices == all row indices of df."""
        train_idx, test_idx = split_train_test(labeled_df, "label", 0.2, 42)
        all_indices = set(range(len(labeled_df)))
        assert set(train_idx) | set(test_idx) == all_indices

    def test_indices_are_non_overlapping(self, labeled_df):
        """train_indices ∩ test_indices == ∅ (no row appears in both)."""
        train_idx, test_idx = split_train_test(labeled_df, "label", 0.2, 42)
        assert len(set(train_idx) & set(test_idx)) == 0

    def test_split_sizes_respect_test_size(self, labeled_df):
        """Test split size is approximately test_size of the full dataset."""
        train_idx, test_idx = split_train_test(labeled_df, "label", 0.2, 42)
        total = len(labeled_df)
        assert len(train_idx) + len(test_idx) == total
        # Allow ±1 row tolerance due to stratification rounding
        assert abs(len(test_idx) - round(total * 0.2)) <= 1

    def test_deterministic_given_same_seed(self, labeled_df):
        """Same seed produces identical index arrays across two calls."""
        train1, test1 = split_train_test(labeled_df, "label", 0.2, 42)
        train2, test2 = split_train_test(labeled_df, "label", 0.2, 42)
        assert np.array_equal(np.sort(train1), np.sort(train2))
        assert np.array_equal(np.sort(test1), np.sort(test2))

    def test_raises_value_error_on_single_sample_class(self):
        """ValueError raised if any class has fewer than 2 samples."""
        df = pd.DataFrame(
            {
                "feature": [1.0, 2.0, 3.0],
                "label": ["A", "A", "B"],  # 'B' has only 1 sample
                "binary_label": [0, 0, 1],
            }
        )
        with pytest.raises(ValueError):
            split_train_test(df, "label", 0.3, 42)


# ---------------------------------------------------------------------------
# prepare_model_ready_data — SDS Section 14.2
# ---------------------------------------------------------------------------

class TestPrepareModelReadyData:
    """Tests for prepare_model_ready_data."""

    def _make_processed_df(self) -> pd.DataFrame:
        """Synthetic fully-processed (encoded, scaled) DataFrame."""
        return pd.DataFrame(
            {
                "proto_enc": [0.0, 1.0, 0.0, 1.0, 0.0, 1.0],
                "frame_len_sc": [0.1, 0.5, 0.3, 0.9, 0.2, 0.7],
                "port_sc": [0.2, 0.8, 0.4, 0.1, 0.6, 0.3],
                "label": [
                    "Normal", "DDoS_HTTP", "Normal",
                    "MITM", "DDoS_HTTP", "MITM",
                ],
                "binary_label": [0, 1, 0, 1, 1, 1],
            }
        )

    def _make_label_encoder(self, labels) -> LabelEncoder:
        """Fit and return a LabelEncoder on given labels."""
        le = LabelEncoder()
        le.fit(labels)
        return le

    def test_x_shape(self):
        """X shape is (n_samples, n_features, 1)."""
        df = self._make_processed_df()
        feature_cols = ["proto_enc", "frame_len_sc", "port_sc"]
        le = self._make_label_encoder(df["label"])
        n_classes = len(le.classes_)
        indices = np.arange(len(df))
        X, _ = prepare_model_ready_data(df, indices, feature_cols, le, n_classes)
        assert X.shape == (len(df), len(feature_cols), 1)

    def test_y_shape_one_hot(self):
        """y shape is (n_samples, num_classes) — one-hot encoded."""
        df = self._make_processed_df()
        feature_cols = ["proto_enc", "frame_len_sc", "port_sc"]
        le = self._make_label_encoder(df["label"])
        n_classes = len(le.classes_)
        indices = np.arange(len(df))
        _, y = prepare_model_ready_data(df, indices, feature_cols, le, n_classes)
        assert y.shape == (len(df), n_classes)

    def test_y_rows_sum_to_one(self):
        """Each row of y sums to 1.0 (valid one-hot encoding)."""
        df = self._make_processed_df()
        feature_cols = ["proto_enc", "frame_len_sc", "port_sc"]
        le = self._make_label_encoder(df["label"])
        n_classes = len(le.classes_)
        indices = np.arange(len(df))
        _, y = prepare_model_ready_data(df, indices, feature_cols, le, n_classes)
        assert np.allclose(y.sum(axis=1), 1.0)

    def test_index_subset_works(self):
        """prepare_model_ready_data correctly selects a row subset via indices."""
        df = self._make_processed_df()
        feature_cols = ["proto_enc", "frame_len_sc", "port_sc"]
        le = self._make_label_encoder(df["label"])
        n_classes = len(le.classes_)
        subset_indices = np.array([0, 2, 4])
        X, y = prepare_model_ready_data(
            df, subset_indices, feature_cols, le, n_classes
        )
        assert X.shape[0] == 3
        assert y.shape[0] == 3

    def test_x_values_match_dataframe(self):
        """X values match the corresponding rows/columns in the DataFrame."""
        df = self._make_processed_df()
        feature_cols = ["proto_enc", "frame_len_sc", "port_sc"]
        le = self._make_label_encoder(df["label"])
        n_classes = len(le.classes_)
        indices = np.array([0, 1])
        X, _ = prepare_model_ready_data(df, indices, feature_cols, le, n_classes)
        expected = df.loc[indices, feature_cols].values.reshape(-1, 3, 1)
        assert np.allclose(X, expected)


# ---------------------------------------------------------------------------
# Leakage regression test — SDS Section 14.2 / Section 19
# ---------------------------------------------------------------------------

class TestLeakageRegression:
    """Regression tests guarding against train/test leakage.

    SDS Section 19 requires: 'encoders/scaler fit only on a training slice
    produce different parameters than if fit on the full dataset (regression
    test guarding against the leakage bug the pipeline order in Section 14.2
    exists to prevent).'
    """

    def _make_asymmetric_df(self) -> pd.DataFrame:
        """DataFrame where train and test rows have different value ranges.

        The training rows (indices 0-5) have numeric values in [10, 50].
        The test rows (indices 6-9) have values in [100, 900].
        If the scaler is fit on the full dataset its min/max will differ
        substantially from a train-only fit.
        """
        return pd.DataFrame(
            {
                "cat_col": ["a", "b", "a", "b", "a", "b", "c", "c", "c", "c"],
                "num_col": [10.0, 20.0, 30.0, 40.0, 50.0, 15.0,
                            100.0, 200.0, 500.0, 900.0],
                "label": [
                    "Normal", "Attack", "Normal", "Attack",
                    "Normal", "Attack", "Normal", "Attack",
                    "Normal", "Attack",
                ],
                "binary_label": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
            }
        )

    def test_scaler_train_only_differs_from_full_dataset_fit(self):
        """MinMaxScaler fit on train-only rows has different scale_ than full fit.

        This is the canonical leakage regression required by SDS Section 19.
        """
        df = self._make_asymmetric_df()
        train_indices = np.array([0, 1, 2, 3, 4, 5])   # low-range rows
        numeric_columns = ["num_col"]

        # Fit on training slice only (correct, no leakage)
        _, scaler_train_only = fit_scaler(df.loc[train_indices], numeric_columns)

        # Fit on full dataset (WRONG — leakage — what this test guards against)
        _, scaler_full = fit_scaler(df, numeric_columns)

        # The data_max_ must differ: train-only sees max=50, full sees max=900
        train_max = scaler_train_only.data_max_[0]
        full_max = scaler_full.data_max_[0]
        assert train_max != full_max, (
            "Train-only scaler and full-dataset scaler have identical data_max_ "
            "— this indicates a leakage bug in the pipeline order."
        )

    def test_categorical_encoder_train_only_differs_from_full_fit(self):
        """OrdinalEncoder fit on training rows only is unaware of test-only categories.

        Training rows contain categories 'a' and 'b'; test rows introduce 'c'.
        A train-only encoder should not have seen 'c' during fitting.
        """
        df = self._make_asymmetric_df()
        train_indices = np.array([0, 1, 2, 3, 4, 5])   # only 'a' and 'b'
        cat_cols = ["cat_col"]

        _, enc_train_only = fit_categorical_encoder(
            df.loc[train_indices], cat_cols
        )
        _, enc_full = fit_categorical_encoder(df, cat_cols)

        # Train-only encoder knows 2 categories; full encoder knows 3
        train_categories = enc_train_only.categories_[0].tolist()
        full_categories = enc_full.categories_[0].tolist()
        assert len(train_categories) != len(full_categories), (
            "Train-only and full-dataset categorical encoders have the same "
            "category list — this indicates a leakage bug."
        )
        assert "c" not in train_categories
        assert "c" in full_categories

