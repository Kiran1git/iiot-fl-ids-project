"""Evaluation entry point for the IIoT Federated IDS project.

Thin orchestration script that:
  1. Loads config via the authoritative config loader.
  2. Sets the global random seed (SDS Section 12).
  3. Constructs the "evaluation" logger (SDS Section 13's fixed logger-name
     table) and passes it down into the evaluation module.
  4. Calls run_evaluation_pipeline(config, logger), which owns every
     evaluation-stage artifact (SDS Section 14.9).

Prerequisites: experiments/run_preprocessing.py, run_centralized.py and
run_federated.py must all have completed, since this script reads both trained
models and both persisted training-time files.

Usage:
    python experiments/run_evaluation.py

Dependencies:
    src.utils.config_loader, src.utils.seed, src.utils.logger,
    src.evaluation.compare_fl_vs_centralized.
"""

import os
import sys
import time

# Ensure project root is on the path regardless of invocation style
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.config_loader import load_config
from src.utils.seed import set_global_seed
from src.utils.logger import get_logger
from src.evaluation.compare_fl_vs_centralized import run_evaluation_pipeline


def main() -> None:
    """Entry point: seed, load config, run the evaluation pipeline.

    Returns:
        None

    Raises:
        Propagates any exception from ``run_evaluation_pipeline`` after
        logging the full traceback at ERROR level.
    """
    config = load_config("configs/config.yaml")

    # Step 0: Set global seed — must be first executable statement (SDS §12)
    set_global_seed(config["seed"])

    logger = get_logger("evaluation", config["paths"]["logs_dir"])
    script_start = time.time()

    logger.info(
        "experiments/run_evaluation.py started — seed=%s", config["seed"]
    )

    try:
        run_evaluation_pipeline(config, logger)
    except Exception:
        logger.error(
            "run_evaluation.py failed — full traceback:", exc_info=True
        )
        raise

    elapsed = time.time() - script_start
    logger.info(
        "experiments/run_evaluation.py complete — total wall-clock: "
        "%.2f seconds",
        elapsed,
    )


if __name__ == "__main__":
    main()
