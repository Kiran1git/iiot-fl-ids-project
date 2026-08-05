"""Centralized CNN-GRU baseline training for the IIoT Federated IDS project.

This module owns the centralized training path (SDS Section 14.7). It trains the
single, frozen CNN-GRU architecture on the full training split (no partitioning),
with early stopping and best-model checkpointing, times only the ``.fit()`` call,
and persists every centralized artifact named in SDS Section 9.

Ownership and reuse rules:
  - The model is built exclusively via ``src.models.cnn_gru.build_cnn_gru`` —
    no second model-construction function exists anywhere in the project.
  - Tensors are built exclusively via
    ``src.preprocessing.encode_normalize.prepare_model_ready_data`` — the
    reshape/one-hot logic is never reimplemented here.
  - ``num_classes`` is derived at runtime as ``len(class_mapping)`` from
    ``outputs/artifacts/class_mapping.json`` — never a hardcoded literal.
  - ``centralized_best_model.h5`` (lowest ``val_loss``) is the only centralized
    model any downstream code reads; ``centralized_last_model.h5`` is audit-only.
  - ``time.time()`` wraps only the ``.fit()`` call — never surrounding I/O,
    artifact persistence, or diagram generation.

Per the SDS Section 11 import graph, this layer imports only from src/utils/,
src/preprocessing/, and src/models/. It never imports src/federated/,
src/partitioning/, src/evaluation/, or src/explainability/.
"""

import json
import logging
import os
import pickle
import time

import numpy as np
import pandas as pd
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

from src.models.cnn_gru import (
    build_cnn_gru,
    get_model_summary_string,
    save_model_architecture_diagram,
)
from src.preprocessing.encode_normalize import prepare_model_ready_data
from src.utils.dataio import read_processed



def train_centralized_model(
    config: dict,
    logger: logging.Logger = None,
) -> None:
    """Train the centralized CNN-GRU baseline and persist all artifacts.

    Purpose:
        Load the processed dataset and preprocessing artifacts, build the
        CNN-GRU model via ``build_cnn_gru``, train it on the full training
        split with ``EarlyStopping`` and ``ModelCheckpoint`` callbacks, measure
        wall-clock training duration around the ``.fit()`` call only, and
        persist the best/last models, the model summary, the architecture
        diagram, the per-epoch history CSV, and the training-time file
        (SDS Section 14.7).

        ``validation_split`` is applied by Keras internally to
        ``(X_train, y_train)`` — no separate held-out validation file is
        produced or read.

    Args:
        config: Fully loaded config dict from
            ``src.utils.config_loader.load_config()``.
        logger: Optional pre-constructed ``logging.Logger`` passed down from
            ``experiments/run_centralized.py``. Modules under ``src/`` never
            construct their own log file (SDS Section 13), so when this is
            omitted a plain handler-less logger is used and no log file beyond
            the five fixed names in SDS Section 13 is ever created.

    Returns:
        None

    Raises:
        Propagates any Keras training exception (and any I/O exception raised
        while loading artifacts or persisting outputs) after logging the full
        stack trace at ERROR level.
    """
    if logger is None:
        logger = logging.getLogger("centralized_training")

    try:
        artifacts_dir = config["paths"]["artifacts_dir"]
        models_dir = config["paths"]["models_dir"]
        results_dir = config["paths"]["results_dir"]

        os.makedirs(models_dir, exist_ok=True)
        os.makedirs(results_dir, exist_ok=True)

        logger.info("=== train_centralized_model starting ===")
        logger.info(
            "Training config summary — epochs=%s | batch_size=%s | "
            "validation_split=%s | learning_rate=%s | loss=%s | "
            "early_stopping_monitor=%s | early_stopping_patience=%s",
            config["training"]["centralized_epochs"],
            config["training"]["batch_size"],
            config["training"]["validation_split"],
            config["training"]["learning_rate"],
            config["training"]["loss"],
            config["training"]["early_stopping_monitor"],
            config["training"]["early_stopping_patience"],
        )

        # ---- Load processed dataset -------------------------------------
        processed_csv_path = os.path.join(
            config["paths"]["processed_data_dir"],
            config["paths"]["processed_data_file"],
        )
        logger.info("Loading processed dataset from '%s'", processed_csv_path)
        df = read_processed(processed_csv_path)

        logger.info(
            "Processed dataset loaded: %d rows x %d columns",
            df.shape[0],
            df.shape[1],
        )
        logger.info(
            "Class distribution: %s", df["label"].value_counts().to_dict()
        )

        # ---- Load preprocessing artifacts -------------------------------
        def _load_pickle(filename: str):
            path = os.path.join(artifacts_dir, filename)
            with open(path, "rb") as handle:
                return pickle.load(handle)

        train_indices = _load_pickle("train_indices.pkl")
        test_indices = _load_pickle("test_indices.pkl")
        feature_columns = _load_pickle("feature_names.pkl")
        label_encoder = _load_pickle("label_encoder.pkl")

        class_mapping_path = os.path.join(artifacts_dir, "class_mapping.json")
        with open(class_mapping_path, "r", encoding="utf-8") as handle:
            class_mapping = json.load(handle)

        # num_classes is always derived, never hardcoded (SDS Section 6).
        num_classes = len(class_mapping)
        num_features = len(feature_columns)
        logger.info(
            "Artifacts loaded — train=%d rows | test=%d rows | features=%d | "
            "classes=%d",
            len(train_indices),
            len(test_indices),
            num_features,
            num_classes,
        )

        # ---- Build model-ready tensors (sole owner: prepare_model_ready_data)
        X_train, y_train = prepare_model_ready_data(
            df, train_indices, feature_columns, label_encoder, num_classes
        )
        X_test, y_test = prepare_model_ready_data(
            df, test_indices, feature_columns, label_encoder, num_classes
        )
        logger.info(
            "Tensors prepared — X_train=%s, y_train=%s, X_test=%s, y_test=%s",
            X_train.shape,
            y_train.shape,
            X_test.shape,
            y_test.shape,
        )

        # ---- Build the model (sole owner: build_cnn_gru) -----------------
        model = build_cnn_gru(
            input_shape=(num_features, 1),
            num_classes=num_classes,
            model_config=config["model"],
            training_config=config["training"],
        )
        summary_text = get_model_summary_string(model)
        logger.debug("Model summary:\n%s", summary_text)

        # ---- Callbacks ---------------------------------------------------
        best_model_path = os.path.join(models_dir, "centralized_best_model.h5")
        last_model_path = os.path.join(models_dir, "centralized_last_model.h5")

        callbacks = [
            EarlyStopping(
                monitor=config["training"]["early_stopping_monitor"],
                patience=config["training"]["early_stopping_patience"],
                restore_best_weights=True,
            ),
            ModelCheckpoint(
                filepath=best_model_path,
                monitor=config["training"]["early_stopping_monitor"],
                save_best_only=True,
            ),
        ]

        # ---- Class weights ------------------------------------------------
        # Edge-IIoTset is imbalanced by a factor of ~1600:1 (Normal vs
        # Fingerprinting). Unweighted categorical crossentropy lets the model
        # reach ~96% accuracy while never once predicting the rarest attack
        # classes, because ignoring a 1,001-row class costs 0.045% accuracy.
        # For an IDS, a missed attack class is the failure mode that matters,
        # so the loss is rebalanced by inverse class frequency.
        #
        # Weights are computed from TRAINING labels only — the test split must
        # not influence any fitted quantity, and a class weight derived from
        # test-set frequencies would be exactly that.
        class_weight = None
        if config["training"].get("use_class_weights", False):
            y_train_int = np.argmax(y_train, axis=1)
            present_classes = np.unique(y_train_int)
            balanced_weights = compute_class_weight(
                class_weight="balanced",
                classes=present_classes,
                y=y_train_int,
            )
            class_weight = {
                int(cls): float(weight)
                for cls, weight in zip(present_classes, balanced_weights)
            }
            logger.info(
                "Class weights enabled (balanced, computed on training rows "
                "only): %s",
                {
                    class_mapping[str(cls)]: round(weight, 4)
                    for cls, weight in class_weight.items()
                },
            )
        else:
            logger.warning(
                "Class weights are disabled. With a %d:1 imbalance the model "
                "can score high accuracy while ignoring rare attack classes.",
                int(
                    df["label"].value_counts().max()
                    / max(df["label"].value_counts().min(), 1)
                ),
            )

        # ---- Train (time.time() wraps ONLY the .fit() call) --------------
        logger.info("Starting centralized training ...")
        start_time = time.time()
        history = model.fit(
            X_train,
            y_train,
            epochs=config["training"]["centralized_epochs"],
            batch_size=config["training"]["batch_size"],
            validation_split=config["training"]["validation_split"],
            class_weight=class_weight,
            callbacks=callbacks,
        )
        training_time_seconds = time.time() - start_time


        # ---- Per-epoch logging ------------------------------------------
        history_df = pd.DataFrame(history.history)
        history_df.insert(0, "epoch", range(1, len(history_df) + 1))
        for row in history_df.itertuples(index=False):
            logger.info(
                "Epoch %d — loss=%.6f accuracy=%.6f val_loss=%.6f "
                "val_accuracy=%.6f",
                row.epoch,
                row.loss,
                row.accuracy,
                row.val_loss,
                row.val_accuracy,
            )

        best_val_loss = float(history_df["val_loss"].min())
        logger.info(
            "Training complete — epochs_run=%d | total_training_time=%.4f "
            "seconds | best val_loss=%.6f",
            len(history_df),
            training_time_seconds,
            best_val_loss,
        )

        # ---- Persist the last-epoch model (audit-only) -------------------
        # restore_best_weights=True means the in-memory model holds the best
        # weights at this point; it is saved for audit purposes and is never
        # loaded by any downstream evaluation, comparison, or dashboard code.
        model.save(last_model_path)
        logger.info("Saved last model (audit-only) to '%s'", last_model_path)
        logger.info(
            "Best model (lowest val_loss) saved by ModelCheckpoint to '%s'",
            best_model_path,
        )

        # ---- Persist model summary and architecture diagram --------------
        summary_path = os.path.join(models_dir, "centralized_model_summary.txt")
        with open(summary_path, "w", encoding="utf-8") as handle:
            handle.write(summary_text)
        logger.info("Saved model summary to '%s'", summary_path)

        diagram_path = os.path.join(models_dir, "centralized_architecture.png")
        save_model_architecture_diagram(model, diagram_path)
        if os.path.exists(diagram_path):
            logger.info("Saved architecture diagram to '%s'", diagram_path)
        else:
            logger.warning(
                "Architecture diagram was not produced at '%s' "
                "(pydot/graphviz likely unavailable; non-fatal).",
                diagram_path,
            )

        # ---- Persist history CSV ----------------------------------------
        history_csv_path = os.path.join(results_dir, "centralized_history.csv")
        history_df.to_csv(history_csv_path, index=False)
        logger.info(
            "Saved training history (%d rows) to '%s'",
            len(history_df),
            history_csv_path,
        )

        # ---- Persist training time ---------------------------------------
        training_time_path = os.path.join(
            results_dir, "centralized_training_time.txt"
        )
        with open(training_time_path, "w", encoding="utf-8") as handle:
            handle.write(str(training_time_seconds))
        logger.info(
            "Saved training time (%.4f s) to '%s'",
            training_time_seconds,
            training_time_path,
        )

        # ---- Held-out test-set evaluation (logged, not persisted here) ----
        # Milestone 4's validation checks require the achieved test-set
        # accuracy to be logged, so this step is mandatory. It runs only after
        # every artifact above is already safely on disk, so an interruption
        # here can never cost the completed training run.
        #
        # batch_size is passed explicitly from config: Keras would otherwise
        # default to 32, which halves the throughput of this pass for no
        # reason. verbose=1 is deliberate — a silent multi-minute evaluation
        # over 443k samples is indistinguishable from a hung process, which
        # invites an interrupt. The progress bar goes to stdout only and never
        # pollutes the log file.
        logger.info(
            "Evaluating on held-out test set — %d samples, batch_size=%s "
            "(~%d batches); this pass is compute-bound and may take a few "
            "minutes. All artifacts above are already saved.",
            len(X_test),
            config["training"]["batch_size"],
            -(-len(X_test) // config["training"]["batch_size"]),
        )
        eval_start = time.time()
        test_loss, test_accuracy = model.evaluate(
            X_test,
            y_test,
            batch_size=config["training"]["batch_size"],
            verbose=1,
        )
        logger.info(
            "Held-out test-set evaluation — loss=%.6f accuracy=%.6f "
            "(random-guess baseline for %d classes ~= %.6f) | "
            "evaluation wall-clock=%.2f seconds",
            test_loss,
            test_accuracy,
            num_classes,
            1.0 / num_classes,
            time.time() - eval_start,
        )

        logger.info("=== train_centralized_model complete ===")

    except KeyboardInterrupt:
        # KeyboardInterrupt derives from BaseException, not Exception, so the
        # handler below would never see it. Logging it explicitly means a
        # manual Ctrl+C leaves a visible record instead of a log that simply
        # stops mid-run with no explanation.
        logger.warning(
            "train_centralized_model interrupted by user (KeyboardInterrupt). "
            "Any artifact logged as saved above is complete and valid on disk."
        )
        raise
    except Exception:
        logger.error(
            "train_centralized_model failed — full traceback:", exc_info=True
        )
        raise
