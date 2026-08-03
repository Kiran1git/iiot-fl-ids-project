"""CNN-GRU model definition for the IIoT Federated IDS project.

This module owns the single, frozen model architecture for the entire project
(SDS Section 6 / Section 14.4). ``build_cnn_gru`` is the sole model-construction
function: it builds both the centralized baseline model and the reconstructed
federated global model. No second model-building function may be introduced
anywhere in the codebase.

Per the SDS Section 11 import graph, this layer imports nothing from
src/preprocessing/, src/centralized/, or src/federated/ — models are consumed
by the training layers, never the reverse.
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

    Purpose:
        Build the project's single, frozen CNN-GRU architecture and return it
        already compiled with the Adam optimizer, the configured loss, and the
        ``["accuracy"]`` metric. The exact 7-entry layer sequence (Input,
        Conv1D, MaxPooling1D, GRU, Dropout, Dense, Dense) is fixed — no layer
        may be added, removed, or reordered.

        ``conv_padding`` is always ``"same"`` and ``gru_return_sequences`` is
        always ``False`` in ``configs/config.yaml``; both are locked values
        (AI Coding Contract Part C) read from ``model_config`` exactly as the
        SDS Section 14.4 layer specification prescribes. ``padding="same"``
        guarantees the Conv1D output sequence length equals ``num_features``
        regardless of ``kernel_size``, so the downstream MaxPooling1D/GRU
        shapes never depend on an unstated padding choice.
        ``return_sequences=False`` guarantees the GRU emits a single
        final-state vector of shape ``(batch, gru_units)``, so no Flatten
        layer is used or needed.

    Args:
        input_shape: The model input shape as ``(num_features, 1)``.
        num_classes: The number of output classes. Always derived at runtime as
            ``len(class_mapping)`` from ``outputs/artifacts/class_mapping.json``
            by the caller — never a hardcoded literal.
        model_config: The ``config["model"]`` sub-dict, supplying layer sizes
            and activations.
        training_config: The ``config["training"]`` sub-dict, supplying
            ``learning_rate`` and ``loss`` for the compile call.

    Returns:
        A compiled ``tensorflow.keras.Model`` implementing the CNN-GRU
        architecture.

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

    Purpose:
        Return the model's Keras summary as text, for DEBUG-level logging by
        the calling training script and for persisting to
        ``*_model_summary.txt`` alongside every saved model.

    Args:
        model: The Keras model whose summary should be captured.

    Returns:
        The full ``model.summary()`` output as a single string.

    Raises:
        Nothing under normal operation.
    """
    buffer = io.StringIO()
    model.summary(print_fn=lambda line: buffer.write(line + "\n"))
    return buffer.getvalue()


def save_model_architecture_diagram(model: Model, output_path: str) -> None:
    """Save a visual diagram of the model architecture as a PNG.

    Purpose:
        Write an architecture diagram alongside a saved model. Diagram
        generation is best-effort, not build-blocking: if ``pydot`` or
        ``graphviz`` is unavailable in the environment, a WARNING is logged
        and the function returns normally rather than raising (SDS Section
        14.4 / Section 15).

    Args:
        model: The Keras model to diagram.
        output_path: Destination PNG path, constructed by the caller via
            ``os.path.join``.

    Returns:
        None

    Raises:
        Nothing — a missing ``pydot``/``graphviz`` installation or any other
        diagram-generation failure is logged as a WARNING and swallowed, since
        the architecture diagram is a non-essential audit artifact.
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
