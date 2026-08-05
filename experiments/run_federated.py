"""Federated training entry point for the IIoT Federated IDS project.

Orchestration script that:
  1. Loads config via the authoritative config loader.
  2. Sets the global random seed (first executable step — SDS Section 12).
  3. Constructs the "federated_training" logger (SDS Section 13's fixed
     logger-name table) and passes it down into the federated server module.
  4. Calls run_federated_simulation(config, logger) to run the production
     4-client / 15-round simulation and persist all federated artifacts
     (SDS Section 14.6).

This script introduces no simulation logic of its own — it is a thin
orchestrator. The single simulation entry point in the project is
``src.federated.server_app.run_federated_simulation``; no second one exists.

Usage:
    python experiments/run_federated.py

Dependencies:
    src.utils.config_loader, src.utils.seed, src.utils.logger,
    src.federated.server_app.
"""

import os
import sys
import time

# Ensure project root is on the path regardless of invocation style
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.config_loader import load_config
from src.utils.seed import set_global_seed
from src.utils.logger import get_logger
from src.federated.server_app import run_federated_simulation


def main() -> None:
    """Entry point: seed, load config, run the federated simulation.

    Purpose:
        Wire the production config, the global seed, and the
        ``federated_training`` logger into ``run_federated_simulation``.
        Production federated values are used exactly as configured
        (``num_clients=4``, ``num_rounds=15``, ``local_epochs=2``) — the smoke
        test's reduced values are never read here.

    Returns:
        None

    Raises:
        Propagates any exception from ``run_federated_simulation`` after
        logging the full traceback at ERROR level.
    """
    config = load_config("configs/config.yaml")

    # Step 0: Set global seed — must be first executable statement (SDS §12)
    set_global_seed(config["seed"])

    logger = get_logger("federated_training", config["paths"]["logs_dir"])
    script_start = time.time()

    logger.info(
        "experiments/run_federated.py started — seed=%s | num_clients=%s | "
        "num_rounds=%s | local_epochs=%s",
        config["seed"],
        config["federated"]["num_clients"],
        config["federated"]["num_rounds"],
        config["federated"]["local_epochs"],
    )

    try:
        run_federated_simulation(config, logger)
    except Exception:
        logger.error(
            "run_federated.py failed — full traceback:", exc_info=True
        )
        raise

    elapsed = time.time() - script_start
    logger.info(
        "experiments/run_federated.py complete — total wall-clock: "
        "%.2f seconds",
        elapsed,
    )


if __name__ == "__main__":
    main()
