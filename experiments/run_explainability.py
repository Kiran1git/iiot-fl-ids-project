"""Explainability entry point for the IIoT Federated IDS project.

Orchestration script that:
  1. Loads config via the authoritative config loader.
  2. Sets the global random seed (SDS Section 12).
  3. Constructs the "explainability" logger (SDS Section 13's fixed
     logger-name table).
  4. Loads the federated global model plus the preprocessing artifacts, builds
     the train/test tensors via prepare_model_ready_data (the sole tensor
     preparation path), and calls generate_shap_explanations (SDS Section
     14.10), which owns all three SHAP artifacts.

The federated global model is the explained model per SDS Section 21
Milestone 9. ``num_classes`` is derived as ``len(class_mapping)`` — never
hardcoded (SDS Section 6/22).

Prerequisites: experiments/run_preprocessing.py and run_federated.py must have
completed.

Usage:
    python experiments/run_explainability.py

Dependencies:
    src.utils.config_loader, src.utils.seed, src.utils.logger,
    src.utils.dataio, src.preprocessing.encode_normalize,
    src.explainability.shap_utils.
"""

import json
import os
import pickle
import sys
import time

# Ensure project root is on the path regardless of invocation style
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import tensorflow

from src.utils.config_loader import load_config
from src.utils.seed import set_global_seed
from src.utils.logger import get_logger
from src.utils.dataio import read_processed
from src.preprocessing.encode_normalize import prepare_model_ready_data
from src.explainability.shap_utils import generate_shap_explanations


def main() -> None:
    """Entry point: seed, load artifacts, generate SHAP explanations.

    Returns:
        None

    Raises:
        FileNotFoundError: If the federated global model has not been produced.
        Propagates any exception from ``generate_shap_explanations`` after
        logging the full traceback at ERROR level.
    """
    config = load_config("configs/config.yaml")

    # Step 0: Set global seed — must be first executable statement (SDS §12)
    set_global_seed(config["seed"])

    logger = get_logger("explainability", config["paths"]["logs_dir"])
    script_start = time.time()

    logger.info(
        "experiments/run_explainability.py started — seed=%s | "
        "background_size=%s | sample_size=%s",
        config["seed"],
        config["explainability"]["background_size"],
        config["explainability"]["sample_size"],
    )

    try:
        artifacts_dir = config["paths"]["artifacts_dir"]
        models_dir = config["paths"]["models_dir"]
        shap_plots_dir = config["paths"]["shap_plots_dir"]

        # ---- Load the explained model --------------------------------
        model_path = os.path.join(models_dir, "federated_global_model.h5")
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Federated global model not found: '{model_path}'. "
                f"Run experiments/run_federated.py first."
            )
        logger.info("Loading federated global model from '%s'", model_path)
        model = tensorflow.keras.models.load_model(model_path)

        # ---- Load processed data and preprocessing artifacts ---------
        processed_path = os.path.join(
            config["paths"]["processed_data_dir"],
            config["paths"]["processed_data_file"],
        )
        logger.info("Loading processed dataset from '%s'", processed_path)
        df = read_processed(processed_path)

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

        # Derived, never hardcoded (SDS §6/§22).
        num_classes = len(class_mapping)
        class_names = [class_mapping[str(i)] for i in range(num_classes)]

        # ---- Build tensors via the sole tensor-preparation path -------
        X_train, _ = prepare_model_ready_data(
            df, train_indices, feature_columns, label_encoder, num_classes
        )
        X_test, _ = prepare_model_ready_data(
            df, test_indices, feature_columns, label_encoder, num_classes
        )
        logger.info(
            "Tensors prepared — X_train=%s | X_test=%s | classes=%d",
            X_train.shape,
            X_test.shape,
            num_classes,
        )

        # The frame is no longer needed once both tensors exist; SHAP holds the
        # model plus the background set, so releasing it here keeps peak
        # memory to the tensors alone.
        del df

        background_output_path = os.path.join(
            artifacts_dir, "shap_background.npy"
        )

        generate_shap_explanations(
            model=model,
            X_train=X_train,
            test_samples_full=X_test,
            class_names=class_names,
            output_dir=shap_plots_dir,
            background_output_path=background_output_path,
            seed=config["seed"],
            background_size=config["explainability"]["background_size"],
            sample_size=config["explainability"]["sample_size"],
            feature_names=list(feature_columns),
            logger=logger,
        )

    except Exception:
        logger.error(
            "run_explainability.py failed — full traceback:", exc_info=True
        )
        raise

    elapsed = time.time() - script_start
    logger.info(
        "experiments/run_explainability.py complete — total wall-clock: "
        "%.2f seconds",
        elapsed,
    )


if __name__ == "__main__":
    main()
