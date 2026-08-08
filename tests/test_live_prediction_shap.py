"""Tests for SHAP explanation of Live Prediction rows.

Covers ``get_top_feature_contributions`` and the contract it relies on:
that ``predict_dataframe(..., return_features=True)`` hands back the exact
tensor the model scored, so the explanation and the prediction describe the
same input.

The unit tests use a tiny stub explainer rather than a real
``shap.GradientExplainer``, so ranking/sign/ordering logic is verified
deterministically and in milliseconds. The artifact-backed class at the end
runs the genuine SHAP path against the persisted background and is skipped
when the artifacts are absent.
"""

import os
import unittest

import numpy as np
import pandas as pd

from src.explainability.shap_utils import (
    explain_single_prediction,
    get_top_feature_contributions,
)


ARTIFACTS_DIR = os.path.join("outputs", "artifacts")
MODELS_DIR = os.path.join("outputs", "models")
BACKGROUND_PATH = os.path.join(ARTIFACTS_DIR, "shap_background.npy")


def _make_shap_values(num_classes: int, num_features: int, seed: int = 0):
    """Build a per-class SHAP structure shaped like the real one."""
    rng = np.random.RandomState(seed)
    return [
        rng.uniform(-1.0, 1.0, size=(1, num_features))
        for _ in range(num_classes)
    ]


class TestGetTopFeatureContributions(unittest.TestCase):
    """Ranking, sign handling and alignment of the contribution table."""

    def setUp(self):
        self.num_features = 8
        self.num_classes = 4
        self.feature_names = [f"f{i}" for i in range(self.num_features)]
        self.sample = np.arange(self.num_features, dtype=np.float32).reshape(
            1, self.num_features, 1
        )

    def test_returns_requested_number_of_features(self):
        shap_values = _make_shap_values(self.num_classes, self.num_features)
        result = get_top_feature_contributions(
            shap_values, self.sample, self.feature_names, 0, top_n=5
        )
        self.assertEqual(len(result), 5)

    def test_top_n_is_capped_at_feature_count(self):
        """Asking for more features than exist returns all of them, not an error."""
        shap_values = _make_shap_values(self.num_classes, self.num_features)
        result = get_top_feature_contributions(
            shap_values, self.sample, self.feature_names, 0, top_n=999
        )
        self.assertEqual(len(result), self.num_features)

    def test_ordered_by_descending_absolute_contribution(self):
        shap_values = _make_shap_values(self.num_classes, self.num_features)
        result = get_top_feature_contributions(
            shap_values, self.sample, self.feature_names, 2, top_n=8
        )
        magnitudes = [row["abs_shap_value"] for row in result]
        self.assertEqual(magnitudes, sorted(magnitudes, reverse=True))

    def test_uses_the_predicted_class_row_not_an_average(self):
        """The explanation must come from the predicted class's own SHAP row.

        Class 1 is given a single dominant feature that is near-zero in every
        other class. If the implementation averaged across classes, that
        feature would be diluted and would not rank first.
        """
        shap_values = [
            np.full((1, self.num_features), 0.001) for _ in range(self.num_classes)
        ]
        shap_values[1][0, 5] = 9.0

        result = get_top_feature_contributions(
            shap_values, self.sample, self.feature_names, 1, top_n=3
        )

        self.assertEqual(result[0]["feature"], "f5")
        self.assertAlmostEqual(result[0]["shap_value"], 9.0)

    def test_different_class_index_gives_different_explanation(self):
        shap_values = _make_shap_values(self.num_classes, self.num_features)
        first = get_top_feature_contributions(
            shap_values, self.sample, self.feature_names, 0, top_n=8
        )
        second = get_top_feature_contributions(
            shap_values, self.sample, self.feature_names, 1, top_n=8
        )
        self.assertNotEqual(
            [row["shap_value"] for row in first],
            [row["shap_value"] for row in second],
        )

    def test_sign_is_preserved_and_direction_matches(self):
        """Negative contributions stay negative — the sign is the whole point."""
        shap_values = [np.zeros((1, self.num_features)) for _ in range(2)]
        shap_values[0][0, 0] = 5.0
        shap_values[0][0, 1] = -3.0

        result = get_top_feature_contributions(
            shap_values, self.sample, self.feature_names, 0, top_n=2
        )

        by_feature = {row["feature"]: row for row in result}
        self.assertGreater(by_feature["f0"]["shap_value"], 0)
        self.assertEqual(by_feature["f0"]["direction"], "increases")
        self.assertLess(by_feature["f1"]["shap_value"], 0)
        self.assertEqual(by_feature["f1"]["direction"], "decreases")

    def test_negative_contribution_can_outrank_positive_one(self):
        """Ranking is by magnitude, so strong negative evidence is not hidden."""
        shap_values = [np.zeros((1, self.num_features)) for _ in range(2)]
        shap_values[0][0, 3] = 0.5
        shap_values[0][0, 6] = -7.0

        result = get_top_feature_contributions(
            shap_values, self.sample, self.feature_names, 0, top_n=1
        )

        self.assertEqual(result[0]["feature"], "f6")

    def test_feature_values_come_from_the_explained_sample(self):
        shap_values = _make_shap_values(self.num_classes, self.num_features)
        result = get_top_feature_contributions(
            shap_values, self.sample, self.feature_names, 0, top_n=8
        )
        for row in result:
            position = self.feature_names.index(row["feature"])
            self.assertAlmostEqual(
                row["feature_value"], float(self.sample[0, position, 0])
            )

    def test_every_expected_key_is_present(self):
        shap_values = _make_shap_values(self.num_classes, self.num_features)
        result = get_top_feature_contributions(
            shap_values, self.sample, self.feature_names, 0, top_n=3
        )
        for row in result:
            self.assertEqual(
                set(row.keys()),
                {
                    "feature",
                    "feature_value",
                    "shap_value",
                    "abs_shap_value",
                    "direction",
                },
            )

    def test_values_are_plain_python_floats(self):
        """Streamlit/pandas formatting expects scalars, not numpy types."""
        shap_values = _make_shap_values(self.num_classes, self.num_features)
        result = get_top_feature_contributions(
            shap_values, self.sample, self.feature_names, 0, top_n=3
        )
        for row in result:
            self.assertIsInstance(row["shap_value"], float)
            self.assertIsInstance(row["feature_value"], float)
            self.assertIsInstance(row["feature"], str)

    def test_out_of_range_class_index_raises(self):
        shap_values = _make_shap_values(self.num_classes, self.num_features)
        with self.assertRaises(IndexError):
            get_top_feature_contributions(
                shap_values, self.sample, self.feature_names, 99, top_n=3
            )

    def test_negative_class_index_raises(self):
        shap_values = _make_shap_values(self.num_classes, self.num_features)
        with self.assertRaises(IndexError):
            get_top_feature_contributions(
                shap_values, self.sample, self.feature_names, -1, top_n=3
            )

    def test_feature_name_count_mismatch_raises(self):
        """A misaligned name list would silently mislabel the explanation."""
        shap_values = _make_shap_values(self.num_classes, self.num_features)
        with self.assertRaises(ValueError):
            get_top_feature_contributions(
                shap_values, self.sample, ["only", "three", "names"], 0, top_n=3
            )

    def test_result_is_dataframe_ready(self):
        shap_values = _make_shap_values(self.num_classes, self.num_features)
        result = get_top_feature_contributions(
            shap_values, self.sample, self.feature_names, 0, top_n=4
        )
        frame = pd.DataFrame(result)
        self.assertEqual(len(frame), 4)
        self.assertIn("shap_value", frame.columns)


class _StubExplainer:
    """Minimal stand-in for shap.GradientExplainer.

    Returns per-class arrays in the same ``(n, num_features, 1)`` shape the
    real GradientExplainer produces, so the squeeze in
    ``explain_single_prediction`` is genuinely exercised.
    """

    def __init__(self, num_classes: int, num_features: int):
        self.num_classes = num_classes
        self.num_features = num_features
        self.call_count = 0

    def shap_values(self, sample):
        self.call_count += 1
        rng = np.random.RandomState(42)
        return [
            rng.uniform(-1, 1, size=(sample.shape[0], self.num_features, 1))
            for _ in range(self.num_classes)
        ]


class TestExplainSingleWithStub(unittest.TestCase):
    """The existing single-sample entry point feeds the new ranking function."""

    def test_explain_then_rank_produces_a_full_table(self):
        num_features, num_classes = 12, 5
        explainer = _StubExplainer(num_classes, num_features)
        sample = np.random.RandomState(1).rand(1, num_features, 1)

        shap_values = explain_single_prediction(None, explainer, sample)

        self.assertEqual(len(shap_values), num_classes)
        self.assertEqual(np.asarray(shap_values[0]).shape, (1, num_features))

        result = get_top_feature_contributions(
            shap_values,
            sample,
            [f"f{i}" for i in range(num_features)],
            3,
            top_n=10,
        )
        self.assertEqual(len(result), 10)

    def test_explainer_is_called_once_per_explanation(self):
        """No hidden re-computation inside the explain path."""
        explainer = _StubExplainer(3, 6)
        sample = np.zeros((1, 6, 1))
        explain_single_prediction(None, explainer, sample)
        self.assertEqual(explainer.call_count, 1)


@unittest.skipUnless(
    os.path.exists(BACKGROUND_PATH),
    "Persisted SHAP background not available.",
)
class TestPersistedBackground(unittest.TestCase):
    """Checks against the real persisted artifacts."""

    def test_background_has_the_model_input_shape(self):
        background = np.load(BACKGROUND_PATH)
        self.assertEqual(background.ndim, 3)
        self.assertEqual(background.shape[2], 1)

    def test_background_matches_the_persisted_feature_count(self):
        """The background must describe the same 88 features the model takes."""
        import pickle

        feature_names_path = os.path.join(ARTIFACTS_DIR, "feature_names.pkl")
        if not os.path.exists(feature_names_path):
            self.skipTest("feature_names.pkl not available.")

        with open(feature_names_path, "rb") as handle:
            feature_names = pickle.load(handle)

        background = np.load(BACKGROUND_PATH)
        self.assertEqual(background.shape[1], len(feature_names))

    def test_loading_the_background_does_not_modify_it(self):
        """Reuse must be read-only: no resampling, no rewriting."""
        before_mtime = os.path.getmtime(BACKGROUND_PATH)
        first = np.load(BACKGROUND_PATH)
        second = np.load(BACKGROUND_PATH)
        after_mtime = os.path.getmtime(BACKGROUND_PATH)

        np.testing.assert_array_equal(first, second)
        self.assertEqual(before_mtime, after_mtime)


@unittest.skipUnless(
    os.path.exists(BACKGROUND_PATH)
    and os.path.exists(
        os.path.join(MODELS_DIR, "federated_global_model_keras215_v2.h5")
    )
    and os.path.exists(os.path.join(ARTIFACTS_DIR, "feature_names.pkl")),
    "Model or SHAP artifacts not available.",
)
class TestEndToEndExplanation(unittest.TestCase):
    """The real SHAP path over the persisted background and real model."""

    @classmethod
    def setUpClass(cls):
        import tensorflow as tf
        import shap

        from src.inference.predict import (
            FEDERATED_GLOBAL_MODEL_FILENAME,
            load_inference_artifacts,
        )
        from src.utils.config_loader import load_config

        cls.config = load_config("configs/config.yaml")
        cls.artifacts = load_inference_artifacts(cls.config)
        cls.model = tf.keras.models.load_model(
            os.path.join(MODELS_DIR, FEDERATED_GLOBAL_MODEL_FILENAME),
            compile=False,
        )
        cls.background = np.load(BACKGROUND_PATH)
        cls.explainer = shap.GradientExplainer(cls.model, cls.background)

    def test_explanation_covers_every_class_and_feature(self):
        sample = self.background[:1]
        shap_values = explain_single_prediction(
            self.model, self.explainer, sample
        )

        self.assertEqual(len(shap_values), self.artifacts["num_classes"])
        self.assertEqual(
            np.asarray(shap_values[0]).shape,
            (1, len(self.artifacts["feature_columns"])),
        )

    def test_top_features_are_real_model_feature_names(self):
        sample = self.background[:1]
        probabilities = self.model.predict(sample, verbose=0)
        predicted_index = int(np.argmax(probabilities[0]))

        shap_values = explain_single_prediction(
            self.model, self.explainer, sample
        )
        result = get_top_feature_contributions(
            shap_values,
            sample,
            self.artifacts["feature_columns"],
            predicted_index,
            top_n=15,
        )

        self.assertEqual(len(result), 15)
        for row in result:
            self.assertIn(row["feature"], self.artifacts["feature_columns"])

    def test_explanation_is_not_uniformly_zero(self):
        """An all-zero explanation would mean the SHAP path is inert."""
        sample = self.background[:1]
        probabilities = self.model.predict(sample, verbose=0)
        predicted_index = int(np.argmax(probabilities[0]))

        shap_values = explain_single_prediction(
            self.model, self.explainer, sample
        )
        result = get_top_feature_contributions(
            shap_values,
            sample,
            self.artifacts["feature_columns"],
            predicted_index,
            top_n=15,
        )

        self.assertGreater(
            max(row["abs_shap_value"] for row in result), 0.0
        )


@unittest.skipUnless(
    os.path.exists(os.path.join(ARTIFACTS_DIR, "feature_names.pkl"))
    and os.path.exists(
        os.path.join(MODELS_DIR, "federated_global_model_keras215_v2.h5")
    ),
    "Model or preprocessing artifacts not available.",
)
class TestReturnFeaturesContract(unittest.TestCase):
    """``return_features`` must expose the very tensor that was scored."""

    @classmethod
    def setUpClass(cls):
        import tensorflow as tf

        from src.inference.predict import (
            FEDERATED_GLOBAL_MODEL_FILENAME,
            load_inference_artifacts,
        )
        from src.utils.config_loader import load_config

        cls.config = load_config("configs/config.yaml")
        cls.artifacts = load_inference_artifacts(cls.config)
        cls.model = tf.keras.models.load_model(
            os.path.join(MODELS_DIR, FEDERATED_GLOBAL_MODEL_FILENAME),
            compile=False,
        )

        raw_path = os.path.join(
            cls.config["paths"]["raw_data_dir"],
            cls.config["paths"]["raw_data_file"],
        )
        if not os.path.exists(raw_path):
            raise unittest.SkipTest("Raw dataset not available.")
        cls.frame = pd.read_csv(raw_path, nrows=8, low_memory=False)

    def test_default_call_still_returns_two_values(self):
        """Existing callers must be unaffected by the new parameter."""
        from src.inference.predict import predict_dataframe

        outcome = predict_dataframe(self.frame, self.model, self.artifacts)
        self.assertEqual(len(outcome), 2)

    def test_return_features_returns_three_values(self):
        from src.inference.predict import predict_dataframe

        outcome = predict_dataframe(
            self.frame, self.model, self.artifacts, return_features=True
        )
        self.assertEqual(len(outcome), 3)

    def test_returned_tensor_matches_the_reported_input_shape(self):
        from src.inference.predict import predict_dataframe

        results, report, X = predict_dataframe(
            self.frame, self.model, self.artifacts, return_features=True
        )
        self.assertEqual(X.shape, report["model_input_shape"])
        self.assertEqual(len(X), len(results))

    def test_returned_tensor_reproduces_the_reported_predictions(self):
        """Re-scoring the returned tensor must give back the same answers.

        This is what proves the explanation is describing the input that was
        actually predicted on, rather than a lookalike recomputation.
        """
        from src.inference.predict import predict_dataframe

        results, _, X = predict_dataframe(
            self.frame, self.model, self.artifacts, return_features=True
        )

        probabilities = self.model.predict(X, verbose=0)
        indices = np.argmax(probabilities, axis=1)
        classes = [
            self.artifacts["class_names"][int(index)] for index in indices
        ]

        self.assertEqual(classes, results["predicted_class"].tolist())
        np.testing.assert_allclose(
            probabilities[np.arange(len(indices)), indices],
            results["confidence"].to_numpy(),
            rtol=1e-6,
            atol=1e-6,
        )

    def test_row_alignment_holds_for_every_row(self):
        """results.iloc[i] must describe X[i] — the SHAP lookup depends on it."""
        from src.inference.predict import predict_dataframe

        results, _, X = predict_dataframe(
            self.frame, self.model, self.artifacts, return_features=True
        )

        for position in range(len(results)):
            single = self.model.predict(
                X[position: position + 1], verbose=0
            )[0]
            expected = self.artifacts["class_names"][int(np.argmax(single))]
            self.assertEqual(
                expected, results.iloc[position]["predicted_class"]
            )


if __name__ == "__main__":
    unittest.main()
