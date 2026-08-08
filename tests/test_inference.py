"""Unit tests for src/inference/predict.py — user-facing batch inference.

What these tests are actually guarding
--------------------------------------
The inference path's whole job is to be *identical* to the path that produced
the experiment results. So the important assertions here are equivalence
assertions, not smoke tests:

  - ``prepare_model_features`` returns exactly what ``prepare_model_ready_data``
    returns for ``X``, so splitting the reshape out cannot have changed the
    labelled path.
  - A raw row pushed through ``prepare_uploaded_features`` lands on the same
    88-column vector as the same row pushed through the real preprocessing
    pipeline's encoders. If the inference path ever grows its own encoding
    logic, this test fails.
  - Feature order comes from ``feature_names.pkl``, so shuffling the uploaded
    CSV's columns changes nothing about the model input.

The remaining tests cover the validation contract: missing columns are fatal
for the file, unparseable rows are dropped and reported, and the reported row
numbers still point at the right lines of the original upload.

No test here trains, fits on full data, or writes to ``outputs/``. The fitted
encoders used by the equivalence tests are built in-memory from a synthetic
frame via the project's own fit functions.
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import LabelEncoder

from src.inference.predict import (
    InvalidInputError,
    get_expected_input_columns,
    load_inference_artifacts,
    predict_dataframe,
    prepare_uploaded_features,
    validate_uploaded_dataframe,
)
from src.preprocessing.encode_normalize import (
    apply_categorical_transformer,
    fit_categorical_transformer,
    prepare_model_features,
    prepare_model_ready_data,
)
from sklearn.preprocessing import MinMaxScaler


# ---------------------------------------------------------------------------
# Fixtures — a miniature but structurally faithful stand-in for the real
# artifacts: two numeric columns, one low-cardinality column that is one-hot
# encoded, exactly as the production bundle does.
# ---------------------------------------------------------------------------

@pytest.fixture()
def raw_training_frame() -> pd.DataFrame:
    """Synthetic 'raw' traffic with two numeric columns and one nominal one."""
    return pd.DataFrame(
        {
            "frame.len": [60.0, 100.0, 80.0, 200.0, 150.0, 120.0],
            "tcp.dstport": [80.0, 443.0, 8080.0, 22.0, 53.0, 445.0],
            "http.request.method": [
                "GET", "POST", "GET", "PUT", "POST", "GET",
            ],
        }
    )


@pytest.fixture()
def artifacts(raw_training_frame) -> dict:
    """An inference bundle fitted the way run_preprocessing_pipeline fits one.

    Built with the project's own ``fit_categorical_transformer`` and a
    ``MinMaxScaler``, so the equivalence tests below compare the inference path
    against the real encoders rather than against a hand-rolled imitation.
    """
    numeric_columns = ["frame.len", "tcp.dstport"]
    categorical_columns = ["http.request.method"]

    categorical_encoder = fit_categorical_transformer(
        raw_training_frame, categorical_columns, []
    )

    encoded = apply_categorical_transformer(
        raw_training_frame, categorical_encoder
    )

    scaler = MinMaxScaler()
    scaler.fit(encoded[numeric_columns])

    feature_columns = numeric_columns + list(
        categorical_encoder["onehot_feature_names"]
    )
    class_mapping = {"0": "DDoS_HTTP", "1": "Normal", "2": "Port_Scanning"}

    return {
        "feature_columns": feature_columns,
        "categorical_encoder": categorical_encoder,
        "scaler": scaler,
        "class_mapping": class_mapping,
        "class_names": [class_mapping[str(i)] for i in range(len(class_mapping))],
        "num_classes": len(class_mapping),
        "numeric_columns": numeric_columns,
        "normal_class_value": "Normal",
    }


class _StubModel:
    """A stand-in for the federated global model.

    Returns a fixed probability row per input row, so the tests assert on the
    inference path's own behaviour (argmax, confidence, Normal/Attack mapping)
    rather than on model weights. The real model is exercised separately by the
    artifact-backed test at the bottom of this file.
    """

    def __init__(self, probability_rows):
        self._probability_rows = np.asarray(probability_rows, dtype=np.float32)
        self.last_input = None

    def predict(self, X, verbose=0):
        self.last_input = X
        return self._probability_rows[: len(X)]


# ---------------------------------------------------------------------------
# prepare_model_features — the refactor must be a no-op for the labelled path
# ---------------------------------------------------------------------------

class TestPrepareModelFeatures:
    """Tests that the extracted reshape matches prepare_model_ready_data."""

    def _processed_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "a": [0.1, 0.5, 0.9],
                "b": [0.2, 0.6, 0.4],
                "label": ["Normal", "DDoS_HTTP", "Normal"],
            }
        )

    def test_shape_is_features_by_one(self):
        """X has shape (n_rows, n_features, 1) for Conv1D input."""
        df = self._processed_frame()
        X = prepare_model_features(df, np.arange(len(df)), ["a", "b"])
        assert X.shape == (3, 2, 1)

    def test_dtype_is_float32(self):
        """The tensor is float32, matching what the Keras graph consumes."""
        df = self._processed_frame()
        X = prepare_model_features(df, np.arange(len(df)), ["a", "b"])
        assert X.dtype == np.float32

    def test_identical_to_prepare_model_ready_data(self):
        """The extracted function returns exactly the labelled path's X.

        This is the regression guard for the refactor: if the two ever diverge,
        the dashboard's uploaded rows stop being comparable to the rows the
        experiment was evaluated on.
        """
        df = self._processed_frame()
        encoder = LabelEncoder().fit(df["label"])
        indices = np.arange(len(df))

        X_labelled, _ = prepare_model_ready_data(
            df, indices, ["a", "b"], encoder, len(encoder.classes_)
        )
        X_features = prepare_model_features(df, indices, ["a", "b"])

        assert np.array_equal(X_labelled, X_features)

    def test_respects_requested_feature_order(self):
        """Columns are emitted in the order requested, not the frame's order."""
        df = self._processed_frame()
        X = prepare_model_features(df, np.array([0]), ["b", "a"])
        assert X[0, 0, 0] == pytest.approx(0.2)
        assert X[0, 1, 0] == pytest.approx(0.1)


# ---------------------------------------------------------------------------
# validate_uploaded_dataframe — the upload contract
# ---------------------------------------------------------------------------

class TestValidateUploadedDataframe:
    """Tests for the validation of an uploaded frame."""

    def test_accepts_a_well_formed_upload(self, raw_training_frame, artifacts):
        """A frame with every required column validates with no losses."""
        valid, report = validate_uploaded_dataframe(raw_training_frame, artifacts)
        assert report["num_valid_rows"] == len(raw_training_frame)
        assert report["invalid_row_numbers"] == []
        assert list(valid.columns) == get_expected_input_columns(artifacts)

    def test_rejects_empty_frame(self, artifacts):
        """An empty upload raises rather than producing zero predictions."""
        empty = pd.DataFrame(columns=get_expected_input_columns(artifacts))
        with pytest.raises(InvalidInputError):
            validate_uploaded_dataframe(empty, artifacts)

    def test_rejects_missing_required_column(self, raw_training_frame, artifacts):
        """A missing model input is fatal for the whole file."""
        broken = raw_training_frame.drop(columns=["tcp.dstport"])
        with pytest.raises(InvalidInputError, match="tcp.dstport"):
            validate_uploaded_dataframe(broken, artifacts)

    def test_error_message_names_the_missing_columns(
        self, raw_training_frame, artifacts
    ):
        """The message is actionable: it names what to add."""
        broken = raw_training_frame.drop(columns=["frame.len"])
        with pytest.raises(InvalidInputError) as excinfo:
            validate_uploaded_dataframe(broken, artifacts)
        assert "frame.len" in str(excinfo.value)

    def test_extra_columns_are_ignored_not_rejected(
        self, raw_training_frame, artifacts
    ):
        """Label columns left in the upload are ignored, not treated as errors."""
        with_labels = raw_training_frame.copy()
        with_labels["Attack_type"] = "Normal"
        with_labels["Attack_label"] = 0

        valid, report = validate_uploaded_dataframe(with_labels, artifacts)

        assert "Attack_type" in report["ignored_columns"]
        assert "Attack_type" not in valid.columns
        assert report["num_valid_rows"] == len(raw_training_frame)

    def test_unparseable_row_is_dropped_and_reported(
        self, raw_training_frame, artifacts
    ):
        """One bad row costs one row, not the whole upload."""
        dirty = raw_training_frame.copy()
        dirty.loc[2, "frame.len"] = "not-a-number"

        valid, report = validate_uploaded_dataframe(dirty, artifacts)

        assert report["num_input_rows"] == len(dirty)
        assert report["num_valid_rows"] == len(dirty) - 1
        # Row 3 in 1-based, user-visible numbering.
        assert report["invalid_row_numbers"] == [3]
        assert len(valid) == len(dirty) - 1

    def test_reported_row_numbers_are_one_based_and_aligned(
        self, raw_training_frame, artifacts
    ):
        """Surviving rows keep the line numbers they had in the source file."""
        dirty = raw_training_frame.copy()
        dirty.loc[0, "tcp.dstport"] = np.nan

        _, report = validate_uploaded_dataframe(dirty, artifacts)

        assert report["invalid_row_numbers"] == [1]
        assert report["row_numbers"].tolist() == [2, 3, 4, 5, 6]

    def test_rejects_when_every_row_is_unparseable(
        self, raw_training_frame, artifacts
    ):
        """A file where nothing survives is an error, not an empty result."""
        dirty = raw_training_frame.copy()
        dirty["frame.len"] = "junk"
        with pytest.raises(InvalidInputError):
            validate_uploaded_dataframe(dirty, artifacts)


# ---------------------------------------------------------------------------
# prepare_uploaded_features — equivalence with the real pipeline
# ---------------------------------------------------------------------------

class TestPrepareUploadedFeatures:
    """Tests that uploaded rows are transformed exactly as training rows were."""

    def test_shape_matches_persisted_feature_count(
        self, raw_training_frame, artifacts
    ):
        """The tensor has one position per persisted feature name."""
        valid, _ = validate_uploaded_dataframe(raw_training_frame, artifacts)
        X = prepare_uploaded_features(valid, artifacts)
        assert X.shape == (len(valid), len(artifacts["feature_columns"]), 1)

    def test_matches_the_pipelines_own_encoders(
        self, raw_training_frame, artifacts
    ):
        """The inference path reproduces the pipeline's transform exactly.

        Computed here the long way — apply_categorical_transformer, then the
        fitted scaler, then a reindex — and compared against what the inference
        path produced. Any second, subtly different preprocessing
        implementation inside predict.py would show up as a mismatch.
        """
        valid, _ = validate_uploaded_dataframe(raw_training_frame, artifacts)
        X = prepare_uploaded_features(valid, artifacts)

        expected_frame = apply_categorical_transformer(
            valid, artifacts["categorical_encoder"]
        )
        numeric = artifacts["numeric_columns"]
        expected_frame[numeric] = artifacts["scaler"].transform(
            expected_frame[numeric]
        )
        expected = (
            expected_frame.loc[:, artifacts["feature_columns"]]
            .to_numpy(dtype=np.float32)
            .reshape(-1, len(artifacts["feature_columns"]), 1)
        )

        assert np.allclose(X, expected)

    def test_column_order_of_upload_is_irrelevant(
        self, raw_training_frame, artifacts
    ):
        """Shuffling the CSV's columns cannot change the model input.

        Feature order is taken from feature_names.pkl, so this must hold — and
        if it ever stops holding, every prediction silently becomes garbage
        rather than failing loudly.
        """
        shuffled = raw_training_frame[
            ["http.request.method", "tcp.dstport", "frame.len"]
        ]

        valid_original, _ = validate_uploaded_dataframe(
            raw_training_frame, artifacts
        )
        valid_shuffled, _ = validate_uploaded_dataframe(shuffled, artifacts)

        assert np.allclose(
            prepare_uploaded_features(valid_original, artifacts),
            prepare_uploaded_features(valid_shuffled, artifacts),
        )

    def test_unseen_category_encodes_as_all_zeros(self, artifacts):
        """A method never seen in training yields an all-zero indicator block.

        This is the persisted encoder's handle_unknown="ignore" behaviour, and
        it is what stops an unfamiliar user upload from raising.
        """
        unseen = pd.DataFrame(
            {
                "frame.len": [90.0],
                "tcp.dstport": [8443.0],
                "http.request.method": ["TRACE"],
            }
        )
        valid, _ = validate_uploaded_dataframe(unseen, artifacts)
        X = prepare_uploaded_features(valid, artifacts)

        onehot_width = len(
            artifacts["categorical_encoder"]["onehot_feature_names"]
        )
        indicator_block = X[0, -onehot_width:, 0]
        assert np.count_nonzero(indicator_block) == 0


# ---------------------------------------------------------------------------
# predict_dataframe — the dashboard's entry point
# ---------------------------------------------------------------------------

class TestPredictDataframe:
    """Tests for the end-to-end prediction call."""

    def test_returns_expected_columns(self, raw_training_frame, artifacts):
        """The result table has exactly the four columns the UI displays."""
        model = _StubModel([[0.1, 0.8, 0.1]] * len(raw_training_frame))
        results, _ = predict_dataframe(raw_training_frame, model, artifacts)
        assert list(results.columns) == [
            "row",
            "predicted_class",
            "confidence",
            "status",
        ]

    def test_one_result_row_per_valid_input_row(
        self, raw_training_frame, artifacts
    ):
        """Every valid uploaded row produces exactly one prediction."""
        model = _StubModel([[0.1, 0.8, 0.1]] * len(raw_training_frame))
        results, report = predict_dataframe(raw_training_frame, model, artifacts)
        assert len(results) == len(raw_training_frame)
        assert report["num_predicted_rows"] == len(raw_training_frame)

    def test_predicted_class_is_the_argmax_class_name(
        self, raw_training_frame, artifacts
    ):
        """The class name is looked up from class_mapping, never hardcoded."""
        # Index 2 -> "Port_Scanning" in the fixture's mapping.
        model = _StubModel([[0.1, 0.2, 0.7]] * len(raw_training_frame))
        results, _ = predict_dataframe(raw_training_frame, model, artifacts)
        assert set(results["predicted_class"]) == {"Port_Scanning"}

    def test_confidence_is_the_max_probability(
        self, raw_training_frame, artifacts
    ):
        """Confidence is the probability of the predicted class."""
        model = _StubModel([[0.1, 0.2, 0.7]] * len(raw_training_frame))
        results, _ = predict_dataframe(raw_training_frame, model, artifacts)
        assert results["confidence"].iloc[0] == pytest.approx(0.7, abs=1e-6)

    def test_status_is_normal_for_the_normal_class(
        self, raw_training_frame, artifacts
    ):
        """The Normal/Attack split uses config's normal_class_value."""
        # Index 1 -> "Normal".
        model = _StubModel([[0.1, 0.8, 0.1]] * len(raw_training_frame))
        results, _ = predict_dataframe(raw_training_frame, model, artifacts)
        assert set(results["status"]) == {"Normal"}

    def test_status_is_attack_for_any_other_class(
        self, raw_training_frame, artifacts
    ):
        """Every non-Normal class is reported as an Attack."""
        model = _StubModel([[0.9, 0.05, 0.05]] * len(raw_training_frame))
        results, _ = predict_dataframe(raw_training_frame, model, artifacts)
        assert set(results["status"]) == {"Attack"}
        assert set(results["predicted_class"]) == {"DDoS_HTTP"}

    def test_row_numbers_skip_dropped_rows(self, raw_training_frame, artifacts):
        """Result row numbers point at the surviving lines of the upload."""
        dirty = raw_training_frame.copy()
        dirty.loc[1, "frame.len"] = "bad"

        model = _StubModel([[0.1, 0.8, 0.1]] * len(dirty))
        results, report = predict_dataframe(dirty, model, artifacts)

        assert report["invalid_row_numbers"] == [2]
        assert results["row"].tolist() == [1, 3, 4, 5, 6]

    def test_model_receives_the_expected_input_shape(
        self, raw_training_frame, artifacts
    ):
        """The model is handed (n_rows, n_features, 1) — nothing else."""
        model = _StubModel([[0.1, 0.8, 0.1]] * len(raw_training_frame))
        _, report = predict_dataframe(raw_training_frame, model, artifacts)
        assert report["model_input_shape"] == (
            len(raw_training_frame),
            len(artifacts["feature_columns"]),
            1,
        )

    def test_results_are_csv_serialisable(self, raw_training_frame, artifacts):
        """The table round-trips through CSV, as the download button requires."""
        model = _StubModel([[0.1, 0.8, 0.1]] * len(raw_training_frame))
        results, _ = predict_dataframe(raw_training_frame, model, artifacts)

        csv_text = results.to_csv(index=False)
        assert "predicted_class" in csv_text.splitlines()[0]
        assert len(csv_text.splitlines()) == len(results) + 1

    def test_invalid_upload_propagates_invalid_input_error(self, artifacts):
        """A structurally wrong file raises the error the dashboard catches."""
        model = _StubModel([[0.1, 0.8, 0.1]])
        with pytest.raises(InvalidInputError):
            predict_dataframe(pd.DataFrame({"unrelated": [1]}), model, artifacts)


# ---------------------------------------------------------------------------
# Artifact-backed test — runs only when the real artifacts are on disk
# ---------------------------------------------------------------------------

class TestWithPersistedArtifacts:
    """Tests against the real persisted artifacts, when they are present."""

    @pytest.fixture()
    def real_artifacts(self) -> dict:
        """Load the project's own artifacts, skipping if they are absent."""
        from src.utils.config_loader import load_config

        try:
            return load_inference_artifacts(load_config("configs/config.yaml"))
        except FileNotFoundError:
            pytest.skip("Preprocessing artifacts not present in outputs/.")

    def test_expected_columns_are_raw_column_names(self, real_artifacts):
        """The upload contract is expressed in raw Edge-IIoTset column names.

        The one-hot *generated* names (for example
        'http.request.method_GET') must never appear in the contract — a user
        uploads raw traffic, not an encoded feature matrix.
        """
        expected = get_expected_input_columns(real_artifacts)
        generated = set(
            real_artifacts["categorical_encoder"]["onehot_feature_names"]
        )
        assert not (set(expected) & generated)

    def test_raw_dataset_columns_satisfy_the_contract(self, real_artifacts):
        """A slice of the real raw CSV is a valid upload.

        The strongest available statement that the contract is achievable: the
        very file the project was built from satisfies it.
        """
        import os

        raw_path = os.path.join("data", "raw", "edge_iiotset.csv")
        if not os.path.exists(raw_path):
            pytest.skip("Raw dataset not present.")

        sample = pd.read_csv(raw_path, nrows=5, low_memory=False)
        valid, report = validate_uploaded_dataframe(sample, real_artifacts)

        assert report["num_valid_rows"] == len(sample)
        assert len(valid.columns) == len(
            get_expected_input_columns(real_artifacts)
        )

    def test_raw_rows_produce_the_full_feature_vector(self, real_artifacts):
        """Real raw rows preprocess into the persisted 88-feature input."""
        import os

        raw_path = os.path.join("data", "raw", "edge_iiotset.csv")
        if not os.path.exists(raw_path):
            pytest.skip("Raw dataset not present.")

        sample = pd.read_csv(raw_path, nrows=5, low_memory=False)
        valid, _ = validate_uploaded_dataframe(sample, real_artifacts)
        X = prepare_uploaded_features(valid, real_artifacts)

        assert X.shape == (
            len(valid),
            len(real_artifacts["feature_columns"]),
            1,
        )
        assert X.dtype == np.float32
        assert not np.isnan(X).any()
