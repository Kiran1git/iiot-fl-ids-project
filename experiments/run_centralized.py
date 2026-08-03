"""Centralized baseline training entry point for the IIoT Federated IDS project.

Orchestration script that:
  1. Sets the global random seed (first executable statement — SDS Section 12).
  2. Loads config via the authoritative config loader.
  3. Constructs the "centralized_training" logger (SDS Section 13's fixed
     logger-name table) and passes it down into the training module.
  4. Calls train_centralized_model(config, logger) to train the CNN-GRU
     baseline and persist all centralized artifacts (SDS Section 14.7).

Usage:
    python experiments/run_centralized.py

Dependencies:
    src.utils.config_loader, src.utils.seed, src.utils.logger,
    src.centralized.train_baseline.
"""

import os
import sys
import time

# Ensure project root is on the path regardless of invocation style
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.config_loader import load_config
from src.utils.seed import set_global_seed
from src.utils.logger import get_logger
from src.centralized.train_baseline import train_centralized_model


def main() -> None:
    """Entry point: seed, load config, run centralized training."""
    config = load_config("configs/config.yaml")

    # Step 0: Set global seed — must be first executable statement (SDS §12)
    set_global_seed(config["seed"])

    logger = get_logger("centralized_training", config["paths"]["logs_dir"])
    script_start = time.time()

    logger.info(
        "experiments/run_centralized.py started — seed=%s", config["seed"]
    )

    try:
        train_centralized_model(config, logger)
    except Exception:
        logger.error(
            "run_centralized.py failed — full traceback:", exc_info=True
        )
        raise

    elapsed = time.time() - script_start
    logger.info(
        "experiments/run_centralized.py complete — total wall-clock: "
        "%.2f seconds",
        elapsed,
    )


if __name__ == "__main__":
    main()
