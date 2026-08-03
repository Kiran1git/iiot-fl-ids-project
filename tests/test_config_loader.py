"""Unit tests for src/utils/config_loader.py — Phase 2.

Covers all three cases mandated by SDS Section 19 and the Phase 2 validation
checklist:
  1. A valid config.yaml loads successfully and returns a dict with all
     required top-level keys.
  2. FileNotFoundError is raised for a non-existent config path.
  3. KeyError is raised when a required top-level key is absent.
"""

import pytest
import yaml

from src.utils.config_loader import load_config


# ---------------------------------------------------------------------------
# Case 1 — valid config loads and all required keys are present
# ---------------------------------------------------------------------------

def test_load_valid_config_returns_dict():
    """load_config returns a dict with all 8 required top-level keys."""
    config = load_config("configs/config.yaml")
    assert isinstance(config, dict)


def test_load_valid_config_contains_all_required_keys():
    """load_config returns a dict containing every SDS-mandated top-level key."""
    required_keys = [
        "seed",
        "paths",
        "dataset",
        "model",
        "training",
        "federated",
        "explainability",
        "dashboard",
    ]
    config = load_config("configs/config.yaml")
    for key in required_keys:
        assert key in config, f"Required key missing from loaded config: '{key}'"


def test_load_valid_config_seed_is_42():
    """load_config returns the correct seed value from the production config."""
    config = load_config("configs/config.yaml")
    assert config["seed"] == 42


# ---------------------------------------------------------------------------
# Case 2 — FileNotFoundError on a non-existent path
# ---------------------------------------------------------------------------

def test_load_config_missing_file_raises_file_not_found():
    """FileNotFoundError is raised when config_path does not exist on disk."""
    with pytest.raises(FileNotFoundError):
        load_config("configs/this_file_does_not_exist.yaml")


def test_load_config_missing_file_message_contains_path():
    """The FileNotFoundError message includes the bad path for diagnosability."""
    bad_path = "configs/no_such_config.yaml"
    with pytest.raises(FileNotFoundError, match=bad_path):
        load_config(bad_path)


# ---------------------------------------------------------------------------
# Case 3 — KeyError on a config that is missing a required top-level key
# ---------------------------------------------------------------------------

def test_load_config_missing_top_level_key_raises_key_error(tmp_path):
    """KeyError is raised when a required top-level key is absent from the YAML."""
    # Construct a minimal YAML that omits "dataset" intentionally
    incomplete = {
        "seed": 42,
        "paths": {},
        # "dataset" intentionally omitted
        "model": {},
        "training": {},
        "federated": {},
        "explainability": {},
        "dashboard": {},
    }
    config_file = tmp_path / "incomplete_config.yaml"
    config_file.write_text(yaml.dump(incomplete), encoding="utf-8")

    with pytest.raises(KeyError):
        load_config(str(config_file))


def test_load_config_missing_key_error_message_names_key(tmp_path):
    """The KeyError message identifies which specific key is absent."""
    incomplete = {
        "seed": 42,
        "paths": {},
        "dataset": {},
        "model": {},
        "training": {},
        "federated": {},
        # "explainability" intentionally omitted
        "dashboard": {},
    }
    config_file = tmp_path / "incomplete_config2.yaml"
    config_file.write_text(yaml.dump(incomplete), encoding="utf-8")

    with pytest.raises(KeyError, match="explainability"):
        load_config(str(config_file))
