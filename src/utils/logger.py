"""Logger factory for the IIoT Federated IDS project.

This module provides the single, authoritative logger-construction function.
No other module may create its own log files or logger-setup logic. The five
fixed logger names (one per experiments/ script) are documented in the naming
table below; callers are responsible for passing the correct literal name.

Fixed logger-name table (SDS Section 13):
    "preprocessing"         -> experiments/run_preprocessing.py
    "centralized_training"  -> experiments/run_centralized.py
    "federated_training"    -> experiments/run_federated.py
    "evaluation"            -> experiments/run_evaluation.py
    "explainability"        -> experiments/run_explainability.py
"""

import logging
import os
from datetime import datetime


def get_logger(name: str, logs_dir: str) -> logging.Logger:
    """Create and return a configured Logger writing to console and a log file.

    Purpose:
        Construct a logging.Logger that emits INFO-level messages to the
        console (stdout) and DEBUG-level messages to a new, uniquely
        timestamped file under logs_dir. Each call produces exactly one new
        log file; files are never overwritten — every run appends a fresh
        timestamped file to preserve full run history.

    Args:
        name: Fixed literal identifier for the calling script. Must be one of
            the five names from SDS Section 13's naming table:
            "preprocessing", "centralized_training", "federated_training",
            "evaluation", "explainability". This is NOT the Python module's
            __name__ — it is a fixed project-level string.
        logs_dir: Directory path where log files are stored. Sourced from
            config["paths"]["logs_dir"] by the calling script.

    Returns:
        logging.Logger: Fully configured logger instance with both a
            StreamHandler (INFO) and a FileHandler (DEBUG) attached.

    Raises:
        OSError: If logs_dir cannot be created on disk.
    """
    os.makedirs(logs_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = os.path.join(logs_dir, f"{name}_{timestamp}.log")

    # Use a unique internal logger name per call (name + timestamp) so that
    # repeated calls from tests never share a cached logger object and
    # accumulate duplicate handlers.
    logger = logging.getLogger(f"{name}_{timestamp}")
    logger.setLevel(logging.DEBUG)

    _fmt = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler — INFO level
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(_fmt)

    # File handler — DEBUG level; never overwrites an existing file
    file_handler = logging.FileHandler(log_filename, mode="a", encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(_fmt)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    return logger
