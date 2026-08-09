"""CNN-GRU model definition for the IIoT Federated IDS project.

``build_cnn_gru`` is the sole model-construction function (SDS Section 6 /
14.4): it builds both the centralized baseline and the reconstructed federated
global model. No second model builder may be introduced anywhere.
"""

import io

import tensorflow
from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import (
    Conv1D,
    Dense,
    Dropout,
    GRU,
    Input,
    MaxPooling1D,
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import plot_model


def build_cnn_gru(
    input_shape: tuple[int, int],
    num_classes: int,
    model_config: dict,
    training_config: dict,
) -> Model:
    """Construct and compile the CNN-GRU model per SDS Section 14.4.

    The 7-entry layer sequence (Input, Conv1D, MaxPooling1D, GRU, Dropout,
    Dense, Dense) is frozen — no layer may be added, removed, or reordered.

    Two locked config values (AI Coding Contract Part C) carry shape
    guarantees: ``conv_padding="same"`` keeps the Conv1D output length equal to
    ``num_features`` regardless of ``kernel_size``, and
    ``gru_return_sequences=False`` makes the GRU emit a single
    ``(batch, gru_units)`` vector, which is why no Flatten layer exists.

    Args:
        input_shape: The model input shape as ``(num_features, 1)``.
        num_classes: The number of output classes. Always derived at runtime as
            ``len(class_mapping)`` by the caller — never a hardcoded literal.
        model_config: The ``config["model"]`` sub-dict.
        training_config: The ``config["training"]`` sub-dict, supplying
            ``learning_rate`` and ``loss``.

    Returns:
        A compiled ``tensorflow.keras.Model``.

    Raises:
        ValueError: If ``num_classes`` is less than 2.
    """
    if num_classes < 2:
        raise ValueError(
            f"num_classes must be >= 2 for multi-class classification, got {num_classes}"
        )

    model = Sequential(
        [
            Input(shape=input_shape),
            Conv1D(
                filters=model_config["conv_filters"],
                kernel_size=model_config["conv_kernel_size"],
                padding=model_config["conv_padding"],
                activation=model_config["conv_activation"],
            ),
            MaxPooling1D(pool_size=model_config["pool_size"]),
            GRU(
                units=model_config["gru_units"],
                return_sequences=model_config["gru_return_sequences"],
            ),
            Dropout(rate=model_config["dropout_rate"]),
            Dense(
                units=model_config["dense_units"],
                activation=model_config["dense_activation"],
            ),
            Dense(
                units=num_classes,
                activation=model_config["output_activation"],
            ),
        ]
    )

    # Adam is the only optimizer ever instantiated in this project.
    # training_config["optimizer"] is documentation-only and is never read here.
    model.compile(
        optimizer=Adam(learning_rate=training_config["learning_rate"]),
        loss=training_config["loss"],
        metrics=["accuracy"],
    )

    return model


def get_model_summary_string(model: Model) -> str:
    """Capture ``model.summary()`` output as a string.

    Used for DEBUG-level logging and for persisting ``*_model_summary.txt``
    alongside every saved model.

    Args:
        model: The Keras model whose summary should be captured.

    Returns:
        The full ``model.summary()`` output as a single string.
    """
    buffer = io.StringIO()
    model.summary(print_fn=lambda line: buffer.write(line + "\n"))
    return buffer.getvalue()


def save_model_architecture_diagram(model: Model, output_path: str) -> None:
    """Save a visual diagram of the model architecture as a PNG.

    Best-effort, never build-blocking (SDS Section 14.4 / 15): a missing
    ``pydot``/``graphviz`` install, or any other diagram failure, is logged as
    a WARNING and swallowed, since the diagram is a non-essential artifact.

    Args:
        model: The Keras model to diagram.
        output_path: Destination PNG path.
    """
    try:
        plot_model(
            model,
            to_file=output_path,
            show_shapes=True,
            show_layer_names=True,
        )
    except Exception as exc:  # noqa: BLE001 - best-effort, never build-blocking
        tensorflow.get_logger().warning(
            "Could not save model architecture diagram to %s: %s. "
            "This is non-fatal; pydot/graphviz may not be installed.",
            output_path,
            exc,
        )
