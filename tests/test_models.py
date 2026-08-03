"""Unit tests for src/models/cnn_gru.py — Phase 5.

Covers every case mandated by SDS Section 19's test-case table for this file
and the Phase 5 validation checklist:
  1. build_cnn_gru returns a compiled tf.keras.Model.
  2. A forward pass on a dummy batch of shape (batch_size, num_features, 1)
     produces output shape (batch_size, num_classes).
  3. ValueError is raised on num_classes < 2.
  4. Conv1D uses padding="same" and GRU uses return_sequences=False, confirmed
     via layer-config inspection.

num_classes is never hardcoded in src/. In these tests it is derived from a
small synthetic class mapping fixture, mirroring how production code derives it
as len(class_mapping) from outputs/artifacts/class_mapping.json.
"""

import numpy
import pytest
import tensorflow
from tensorflow.keras.layers import Conv1D, Dense, Dropout, GRU, MaxPooling1D

from src.models.cnn_gru import (
    build_cnn_gru,
    get_model_summary_string,
    save_model_architecture_diagram,
)
from src.utils.config_loader import load_config


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def config() -> dict:
    """The production config, the single source of every model hyperparameter."""
    return load_config("configs/config.yaml")


@pytest.fixture(scope="module")
def model_config(config: dict) -> dict:
    """The config["model"] sub-dict passed to build_cnn_gru."""
    return config["model"]


@pytest.fixture(scope="module")
def training_config(config: dict) -> dict:
    """The config["training"] sub-dict passed to build_cnn_gru."""
    return config["training"]


@pytest.fixture(scope="module")
def class_mapping() -> dict:
    """A small synthetic class mapping, standing in for class_mapping.json.

    Keys are stringified integers, exactly as fit_label_encoder persists them,
    so num_classes is derived here the same way production code derives it.
    """
    return {"0": "Normal", "1": "DDoS", "2": "Injection", "3": "MITM"}


@pytest.fixture(scope="module")
def num_classes(class_mapping: dict) -> int:
    """num_classes derived at runtime as len(class_mapping) — never hardcoded."""
    return len(class_mapping)


@pytest.fixture(scope="module")
def num_features() -> int:
    """A small synthetic feature count for dummy-batch forward passes."""
    return 20


@pytest.fixture(scope="module")
def input_shape(num_features: int) -> tuple[int, int]:
    """The model input shape, (num_features, 1)."""
    return (num_features, 1)


@pytest.fixture(scope="module")
def model(input_shape, num_classes, model_config, training_config):
    """A single built model reused across the read-only inspection tests."""
    return build_cnn_gru(
        input_shape=input_shape,
        num_classes=num_classes,
        model_config=model_config,
        training_config=training_config,
    )


# ---------------------------------------------------------------------------
# Case 1 — build_cnn_gru returns a compiled tf.keras.Model
# ---------------------------------------------------------------------------

def test_build_cnn_gru_returns_keras_model(model):
    """build_cnn_gru returns a tensorflow.keras.Model instance."""
    assert isinstance(model, tensorflow.keras.Model)


def test_build_cnn_gru_returns_compiled_model(model):
    """The returned model is already compiled (optimizer and loss are set)."""
    assert model.optimizer is not None, "Model was returned uncompiled (no optimizer)"
    assert model.loss is not None, "Model was returned uncompiled (no loss)"


def test_build_cnn_gru_uses_adam_optimizer(model):
    """Adam is the only optimizer ever instantiated (SDS Section 6 / 14.4)."""
    assert isinstance(model.optimizer, tensorflow.keras.optimizers.Adam)


def test_build_cnn_gru_uses_configured_learning_rate(model, training_config):
    """The Adam learning rate comes from config["training"]["learning_rate"]."""
    learning_rate = float(
        tensorflow.keras.backend.get_value(model.optimizer.learning_rate)
    )
    assert learning_rate == pytest.approx(training_config["learning_rate"])


def test_build_cnn_gru_uses_configured_loss(model, training_config):
    """The compiled loss comes from config["training"]["loss"]."""
    assert training_config["loss"] in str(model.loss)


def test_build_cnn_gru_compiles_accuracy_metric(model, num_features, num_classes):
    """The model is compiled with the ["accuracy"] metric.

    In Keras 2.x, compiled metrics are lazily instantiated: model.metrics_names
    is empty until the model has actually been run. A single evaluate() call on
    a dummy batch materialises them, which is why this test evaluates rather
    than inspecting the freshly compiled model.

    Targets are one-hot encoded because the configured loss is
    categorical_crossentropy, which expects one-hot rather than integer labels.
    """
    batch_size = 4
    dummy_x = numpy.random.rand(batch_size, num_features, 1).astype("float32")
    integer_labels = numpy.arange(batch_size) % num_classes
    dummy_y = numpy.eye(num_classes, dtype="float32")[integer_labels]
    model.evaluate(dummy_x, dummy_y, verbose=0)
    assert any("accuracy" in name for name in model.metrics_names)


# ---------------------------------------------------------------------------
# Case 2 — forward pass output shape
# ---------------------------------------------------------------------------

def test_forward_pass_output_shape(model, num_features, num_classes):
    """A dummy batch of (batch, num_features, 1) yields (batch, num_classes)."""
    batch_size = 8
    dummy_batch = numpy.random.rand(batch_size, num_features, 1).astype("float32")
    predictions = model.predict(dummy_batch, verbose=0)
    assert predictions.shape == (batch_size, num_classes)


def test_model_declared_output_shape(model, num_classes):
    """The model's declared output shape is (None, num_classes)."""
    assert model.output_shape == (None, num_classes)


def test_model_declared_input_shape(model, num_features):
    """The model's declared input shape is (None, num_features, 1)."""
    assert model.input_shape == (None, num_features, 1)


def test_forward_pass_outputs_are_softmax_probabilities(model, num_features):
    """Softmax output rows sum to 1.0, confirming the output activation."""
    batch_size = 4
    dummy_batch = numpy.random.rand(batch_size, num_features, 1).astype("float32")
    predictions = model.predict(dummy_batch, verbose=0)
    row_sums = predictions.sum(axis=1)
    assert numpy.allclose(row_sums, 1.0, atol=1e-5)


# ---------------------------------------------------------------------------
# Case 3 — ValueError on num_classes < 2
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("bad_num_classes", [1, 0, -1])
def test_build_cnn_gru_raises_value_error_on_num_classes_below_two(
    bad_num_classes, input_shape, model_config, training_config
):
    """ValueError is raised for any num_classes < 2 (SDS Section 14.4)."""
    with pytest.raises(ValueError):
        build_cnn_gru(
            input_shape=input_shape,
            num_classes=bad_num_classes,
            model_config=model_config,
            training_config=training_config,
        )


def test_value_error_message_mentions_num_classes(
    input_shape, model_config, training_config
):
    """The ValueError message names num_classes for diagnosability."""
    with pytest.raises(ValueError, match="num_classes"):
        build_cnn_gru(
            input_shape=input_shape,
            num_classes=1,
            model_config=model_config,
            training_config=training_config,
        )


# ---------------------------------------------------------------------------
# Case 4 — layer-config inspection: padding="same", return_sequences=False
# ---------------------------------------------------------------------------

def test_conv1d_padding_is_same(model):
    """Conv1D.padding is always "same" (locked, AI Coding Contract Part C)."""
    conv_layers = [layer for layer in model.layers if isinstance(layer, Conv1D)]
    assert len(conv_layers) == 1, "Expected exactly one Conv1D layer"
    assert conv_layers[0].get_config()["padding"] == "same"


def test_gru_return_sequences_is_false(model):
    """GRU.return_sequences is always False (locked, AI Coding Contract Part C)."""
    gru_layers = [layer for layer in model.layers if isinstance(layer, GRU)]
    assert len(gru_layers) == 1, "Expected exactly one GRU layer"
    assert gru_layers[0].get_config()["return_sequences"] is False


def test_layer_sequence_matches_sds_section_14_4(model):
    """The layer sequence matches SDS Section 14.4 exactly, in order."""
    expected = [Conv1D, MaxPooling1D, GRU, Dropout, Dense, Dense]
    actual = [type(layer) for layer in model.layers]
    assert actual == expected


def test_conv1d_uses_configured_filters_and_kernel(model, model_config):
    """Conv1D filters/kernel_size/activation are sourced from config["model"]."""
    conv_config = [
        layer for layer in model.layers if isinstance(layer, Conv1D)
    ][0].get_config()
    assert conv_config["filters"] == model_config["conv_filters"]
    assert conv_config["kernel_size"] == (model_config["conv_kernel_size"],)
    assert conv_config["activation"] == model_config["conv_activation"]


def test_maxpooling_uses_configured_pool_size(model, model_config):
    """MaxPooling1D pool_size is sourced from config["model"]."""
    pool_config = [
        layer for layer in model.layers if isinstance(layer, MaxPooling1D)
    ][0].get_config()
    assert pool_config["pool_size"] == (model_config["pool_size"],)


def test_gru_uses_configured_units(model, model_config):
    """GRU units are sourced from config["model"]."""
    gru_config = [
        layer for layer in model.layers if isinstance(layer, GRU)
    ][0].get_config()
    assert gru_config["units"] == model_config["gru_units"]


def test_dropout_uses_configured_rate(model, model_config):
    """Dropout rate is sourced from config["model"]."""
    dropout_config = [
        layer for layer in model.layers if isinstance(layer, Dropout)
    ][0].get_config()
    assert dropout_config["rate"] == pytest.approx(model_config["dropout_rate"])


def test_dense_layers_use_configured_units_and_activations(
    model, model_config, num_classes
):
    """The hidden Dense and output Dense layers match config and num_classes."""
    dense_layers = [layer for layer in model.layers if isinstance(layer, Dense)]
    assert len(dense_layers) == 2, "Expected exactly two Dense layers"

    hidden_config = dense_layers[0].get_config()
    assert hidden_config["units"] == model_config["dense_units"]
    assert hidden_config["activation"] == model_config["dense_activation"]

    output_config = dense_layers[1].get_config()
    assert output_config["units"] == num_classes
    assert output_config["activation"] == model_config["output_activation"]


def test_no_flatten_layer_present(model):
    """No Flatten layer exists — return_sequences=False makes one unnecessary."""
    layer_types = [type(layer).__name__ for layer in model.layers]
    assert "Flatten" not in layer_types


# ---------------------------------------------------------------------------
# get_model_summary_string
# ---------------------------------------------------------------------------

def test_get_model_summary_string_returns_non_empty_string(model):
    """get_model_summary_string returns a non-empty string."""
    summary = get_model_summary_string(model)
    assert isinstance(summary, str)
    assert len(summary) > 0


def test_get_model_summary_string_contains_layer_names(model):
    """The captured summary text mentions the architecture's layer types."""
    summary = get_model_summary_string(model)
    assert "conv1d" in summary.lower()
    assert "gru" in summary.lower()
    assert "dense" in summary.lower()


# ---------------------------------------------------------------------------
# save_model_architecture_diagram
# ---------------------------------------------------------------------------

def test_save_model_architecture_diagram_never_raises(model, tmp_path):
    """Diagram generation is best-effort and never raises, per SDS 14.4."""
    output_path = str(tmp_path / "architecture.png")
    save_model_architecture_diagram(model, output_path)  # must not raise


def test_save_model_architecture_diagram_returns_none(model, tmp_path):
    """save_model_architecture_diagram returns None."""
    output_path = str(tmp_path / "architecture_returns_none.png")
    assert save_model_architecture_diagram(model, output_path) is None


# ---------------------------------------------------------------------------
# Lightweight-model soft guideline (SDS Section 21, Milestone 3)
# ---------------------------------------------------------------------------

def test_model_parameter_count_is_lightweight(model):
    """Parameter count stays under the SDS Milestone 3 soft guideline of 500K."""
    assert model.count_params() < 500_000
