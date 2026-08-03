"""Configuration loader for the IIoT Federated IDS project.

This module provides the single, authoritative config-loading function for the
entire project. No other module may implement its own YAML-loading or config-
validation logic.
"""

import os

import yaml


_REQUIRED_TOP_LEVEL_KEYS = (
    "seed",
    "paths",
    "dataset",
    "model",
    "training",
    "federated",
    "explainability",
    "dashboard",
)


def load_config(config_path: str = "configs/config.yaml") -> dict:
    """Load and parse the project YAML configuration file.

    Purpose:
        Read configs/config.yaml into a Python dictionary and validate that
        every required top-level key is present. This is the single
        authoritative config-loading call for the entire project — no other
        module may re-implement YAML loading or key validation.

    Args:
        config_path: Path to the YAML configuration file.
            Defaults to "configs/config.yaml".

    Returns:
        dict: Fully parsed YAML content as a Python dictionary.

    Raises:
        FileNotFoundError: If config_path does not exist on disk.
        KeyError: If any of the required top-level keys
            (seed, paths, dataset, model, training, federated,
            explainability, dashboard) is absent from the parsed config.
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(
            f"Configuration file not found: '{config_path}'"
        )

    with open(config_path, "r", encoding="utf-8") as fh:
        config = yaml.safe_load(fh)

    for key in _REQUIRED_TOP_LEVEL_KEYS:
        if key not in config:
            raise KeyError(
                f"Required top-level configuration key missing: '{key}'"
            )

    return config
