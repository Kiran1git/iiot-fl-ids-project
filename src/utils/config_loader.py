"""Configuration loader for the IIoT Federated IDS project.

This module provides the single, authoritative config-loading function for the
entire project. No other module may implement its own YAML-loading or config-
validation logic.

Run profiles
------------
The project supports two run profiles, selected without editing any code:

  ``FULL_EXPERIMENT`` (default)
      ``configs/config.yaml`` exactly as written — 4 clients, 15 rounds,
      2 local epochs, batch size 64. This is the profile intended for RunPod,
      Colab, Kaggle, or any machine with headroom.

  ``LAPTOP_MODE``
      ``configs/config.yaml`` with ``configs/config_laptop.yaml`` deep-merged
      on top — 2 clients, 5 rounds, 1 local epoch, batch size 32, plus Ray
      object-store and concurrency caps. This profile exists because Ray on
      Windows backs its object store with a memory-mapped file and fails with
      ``CreateFileMapping() failed. GetLastError() = 1450`` once the machine
      runs out of committable memory.

The profile is chosen by, in order of precedence:
  1. the ``mode`` argument to :func:`load_config`;
  2. the ``IIOT_MODE`` environment variable;
  3. the default, ``FULL_EXPERIMENT``.

The overlay is applied as a **deep merge**, so a mode file only needs to state
the keys it changes. Every unstated key — all of ``dataset``, ``model``,
``paths``, ``explainability`` — keeps its production value, which is what
guarantees the two profiles share identical preprocessing, an identical
CNN-GRU architecture, and an identical evaluation pipeline.
"""

import copy
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

FULL_EXPERIMENT = "FULL_EXPERIMENT"
LAPTOP_MODE = "LAPTOP_MODE"

#: Mode name -> overlay file applied on top of the base config. A mode mapped
#: to ``None`` uses the base config untouched.
_MODE_OVERLAY_FILES = {
    FULL_EXPERIMENT: None,
    LAPTOP_MODE: "configs/config_laptop.yaml",
}


def resolve_mode(mode: str = None) -> str:
    """Resolve the active run profile name.

    Applies the fixed precedence — explicit argument, then the ``IIOT_MODE``
    environment variable, then ``FULL_EXPERIMENT`` — so a laptop run can be
    selected with ``set IIOT_MODE=LAPTOP_MODE`` and nothing else.

    Args:
        mode: Explicit profile name, or ``None`` to fall back to the
            environment variable / default. Case-insensitive.

    Returns:
        str: One of ``FULL_EXPERIMENT`` or ``LAPTOP_MODE``.

    Raises:
        ValueError: If the resolved name is not a known profile. Failing here
            is deliberate: silently running the 4-client profile because of a
            typo like ``LAPTOP-MODE`` is exactly the crash this feature exists
            to prevent.
    """
    resolved = mode or os.environ.get("IIOT_MODE") or FULL_EXPERIMENT
    resolved = str(resolved).strip().upper()

    if resolved not in _MODE_OVERLAY_FILES:
        raise ValueError(
            f"Unknown run mode '{resolved}'. Valid modes: "
            f"{sorted(_MODE_OVERLAY_FILES)}."
        )

    return resolved


def _deep_merge(base: dict, overlay: dict) -> dict:
    """Recursively merge ``overlay`` into a copy of ``base``.

    Lets an overlay file restate only the keys it changes. A shallow
    ``dict.update`` would replace the whole ``federated`` block, silently
    deleting ``partition_strategy`` and every other key the overlay did not
    happen to repeat — so the merge must recurse.

    Args:
        base: The base config (never mutated).
        overlay: The overriding values.

    Returns:
        dict: A new dict; nested dicts present in both are merged key-by-key,
            and any non-dict value in ``overlay`` wins outright.
    """
    merged = copy.deepcopy(base)

    for key, overlay_value in overlay.items():
        base_value = merged.get(key)
        if isinstance(base_value, dict) and isinstance(overlay_value, dict):
            merged[key] = _deep_merge(base_value, overlay_value)
        else:
            merged[key] = copy.deepcopy(overlay_value)

    return merged



def load_config(
    config_path: str = "configs/config.yaml",
    mode: str = None,
) -> dict:
    """Load, merge, and validate the project YAML configuration.

    Reads the base config, deep-merges the active run profile's overlay on top,
    then validates that every required top-level key is present. This is the
    single authoritative config-loading call for the entire project — no other
    module may re-implement YAML loading, overlay merging, or key validation.

    Args:
        config_path: Path to the base YAML configuration file.
            Defaults to "configs/config.yaml".
        mode: Optional run profile (``"FULL_EXPERIMENT"`` or
            ``"LAPTOP_MODE"``). When ``None``, the ``IIOT_MODE`` environment
            variable is consulted, then the ``FULL_EXPERIMENT`` default. The
            resolved name is recorded at ``config["run_mode"]`` so downstream
            code and the log file both show which profile actually ran.

    Returns:
        dict: The fully merged configuration, with ``config["run_mode"]`` set.

    Raises:
        FileNotFoundError: If ``config_path``, or the selected mode's overlay
            file, does not exist on disk.
        ValueError: If ``mode`` (or ``IIOT_MODE``) names an unknown profile.
        KeyError: If any required top-level key is absent after merging.
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(
            f"Configuration file not found: '{config_path}'"
        )

    with open(config_path, "r", encoding="utf-8") as fh:
        config = yaml.safe_load(fh)

    # ---- Apply the run-profile overlay ----------------------------------
    # FULL_EXPERIMENT maps to None, so the production config is returned
    # completely untouched — the cloud path is unaffected by this feature.
    resolved_mode = resolve_mode(mode)
    overlay_path = _MODE_OVERLAY_FILES[resolved_mode]

    if overlay_path is not None:
        if not os.path.exists(overlay_path):
            raise FileNotFoundError(
                f"Run mode '{resolved_mode}' requires the overlay file "
                f"'{overlay_path}', which was not found."
            )
        with open(overlay_path, "r", encoding="utf-8") as fh:
            overlay = yaml.safe_load(fh) or {}
        config = _deep_merge(config, overlay)

    config["run_mode"] = resolved_mode

    # Validated after merging, so an overlay can never remove a required key.
    for key in _REQUIRED_TOP_LEVEL_KEYS:
        if key not in config:
            raise KeyError(
                f"Required top-level configuration key missing: '{key}'"
            )

    return config

