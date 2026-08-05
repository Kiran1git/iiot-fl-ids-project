"""Preprocessing pipeline entry point for the IIoT Federated IDS project.

Orchestration script that:
  1. Sets the global random seed (first executable statement — SDS Section 12).
  2. Loads config via the authoritative config loader.
  3. Calls run_preprocessing_pipeline(config) to execute all 11 preprocessing
     steps (SDS Section 14.2).
  4. Calls generate_class_distribution_plot to produce
     outputs/results/class_distribution.png.

Usage:
    python experiments/run_preprocessing.py

Dependencies:
    src.utils.config_loader, src.utils.seed, src.utils.logger,
    src.preprocessing.encode_normalize, src.evaluation.metrics.
"""

import os
import sys
import time

# Ensure project root is on the path regardless of invocation style
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.config_loader import load_config
from src.utils.seed import set_global_seed
from src.utils.logger import get_logger
from src.preprocessing.encode_normalize import run_preprocessing_pipeline
from src.evaluation.metrics import generate_class_distribution_plot
from src.utils.dataio import read_processed



def main() -> None:
    """Entry point: load config, seed, run pipeline, produce class-dist plot."""
    config = load_config("configs/config.yaml")

    # Step 0: Set global seed — must be first executable statement (SDS §12)
    set_global_seed(config["seed"])

    logger = get_logger("preprocessing", config["paths"]["logs_dir"])
    script_start = time.time()

    logger.info(
        "experiments/run_preprocessing.py started — seed=%s", config["seed"]
    )

    try:
        # Step 1: Run the full 11-step preprocessing pipeline
        run_preprocessing_pipeline(config)

        # Step 2: Generate class distribution plot (orchestration layer — §14.2)
        processed_csv_path = os.path.join(
            config["paths"]["processed_data_dir"],
            config["paths"]["processed_data_file"],
        )
        results_dir = config["paths"]["results_dir"]
        os.makedirs(results_dir, exist_ok=True)
        plot_path = os.path.join(results_dir, "class_distribution.png")

        logger.info(
            "Generating class distribution plot from '%s' -> '%s'",
            processed_csv_path,
            plot_path,
        )
        # Only the label column is needed for the distribution plot. Parquet is
        # columnar, so this reads one column instead of materialising the whole
        # ~1.9M x 95 frame purely to call value_counts on it.
        df_processed = read_processed(processed_csv_path, columns=["label"])
        generate_class_distribution_plot(df_processed, "label", plot_path)

        logger.info("Class distribution plot saved to '%s'", plot_path)

    except Exception:
        logger.error(
            "run_preprocessing.py failed — full traceback:", exc_info=True
        )
        raise

    elapsed = time.time() - script_start
    logger.info(
        "experiments/run_preprocessing.py complete — total wall-clock: "
        "%.2f seconds",
        elapsed,
    )


if __name__ == "__main__":
    main()
