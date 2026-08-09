"""Global random seed management for the IIoT Federated IDS project.

This module provides the single, authoritative seed-setting function. No other
module may implement its own seed-setting logic. set_global_seed() must be
called as the first executable statement in every experiments/*.py script,
before any data loading, model building, or partitioning.
"""

import os
import random

import numpy
import tensorflow


def set_global_seed(seed: int = 42) -> None:
    """Set all relevant random seeds to guarantee full run-to-run determinism.

    Seeds the Python stdlib, NumPy, TensorFlow, and PYTHONHASHSEED so that
    preprocessing, weight initialisation, training, IID partitioning, and SHAP
    sampling all produce byte-identical results across runs on the same machine.
    The seed is also written to ``outputs/artifacts/random_seed.txt`` for audit
    (SDS Section 12).

    Args:
        seed: The integer seed value to set everywhere. Defaults to 42.
    """
    random.seed(seed)
    numpy.random.seed(seed)
    tensorflow.random.set_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    artifacts_dir = os.path.join("outputs", "artifacts")
    os.makedirs(artifacts_dir, exist_ok=True)
    seed_path = os.path.join(artifacts_dir, "random_seed.txt")
    with open(seed_path, "w", encoding="utf-8") as fh:
        fh.write(str(seed))
