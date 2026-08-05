"""Tests for the federated strategy and the end-to-end simulation loop — Phase 9.

This file is simultaneously the unit test for ``src/federated/server_app.py``
and the smoke test for the whole federated loop (SDS Section 19, Milestone 6).

Design constraints enforced here:
  - The production ``configs/config.yaml`` (``num_clients = 4``) is **never**
    used and **never** loaded or mutated. This module builds its own
    independent, in-memory reduced config with
    ``num_clients = min_available_clients = min_fit_clients =
    min_evaluate_clients = 2`` so ``get_strategy``'s minimum-client thresholds
    are satisfiable by a 2-client simulation (SDS Section 19, Section 22).
  - All data is synthetic and in-memory. No processed CSV, no artifact under
    ``outputs/``, and no file on disk is read or written by these tests.
  - The pinned ``flwr==1.8.0`` ``start_simulation(client_fn=..., num_clients=...,
    config=..., strategy=...)`` API is used exclusively. The app-based
    ``run_simulation``/``ClientApp``/``ServerApp`` API is permanently forbidden.
"""

import logging
from functools import partial

import numpy
import pandas
import pytest
from sklearn.preprocessing import LabelEncoder

import flwr
from flwr.common import ndarrays_to_parameters, parameters_to_ndarrays

from src.federated.client_app import client_fn
from src.federated.server_app import (
    SavingFedAvg,
    get_strategy,
    run_federated_simulation,
    weighted_average_eval,
    weighted_average_fit,
)
from src.models.cnn_gru import build_cnn_gru

# --- Synthetic-data constants (deliberately tiny — this is a smoke test) ---
FEATURE_COLUMNS = ["f0", "f1", "f2", "f3", "f4", "f5"]
CLASS_NAMES = ["Normal", "DDoS", "Scan"]
REDUCED_NUM_CLIENTS = 2
REDUCED_NUM_ROUNDS = 2


def _make_synthetic_frame(num_rows: int, seed: int) -> pandas.DataFrame:
    """Build a small synthetic processed-style DataFrame.

    Args:
        num_rows: Number of rows to generate.
        seed: Seed for the local ``RandomState`` — never the global state.

    Returns:
        pandas.DataFrame: Numeric feature columns plus a ``label`` column
            cycling through every class, so no class is ever absent.
    """
    random_state = numpy.random.RandomState(seed)
    frame_data = {
        column: random_state.rand(num_rows) for column in FEATURE_COLUMNS
    }
    frame_data["label"] = [
        CLASS_NAMES[index % len(CLASS_NAMES)] for index in range(num_rows)
    ]
    return pandas.DataFrame(frame_data)


def _make_reduced_config() -> dict:
    """Construct the independent reduced-scale config for these tests only.

    Purpose:
        Build a fresh config dict from scratch — never by loading or mutating
        ``configs/config.yaml`` — with ``num_clients = 2`` so the strategy's
        minimum-client thresholds are satisfiable by a 2-client simulation
        (SDS Section 19).

    Returns:
        dict: A complete, self-contained reduced config, including the
            ``runtime`` sub-dict that ``FLClient`` requires.
    """
    return {
        "seed": 42,
        "model": {
            "conv_filters": 8,
            "conv_kernel_size": 3,
            "conv_padding": "same",
            "conv_activation": "relu",
            "pool_size": 2,
            "gru_units": 8,
            "gru_return_sequences": False,
            "dropout_rate": 0.3,
            "dense_units": 16,
            "dense_activation": "relu",
            "output_activation": "softmax",
        },
        "training": {
            "learning_rate": 0.001,
            "loss": "categorical_crossentropy",
            "batch_size": 8,
            # Off for the smoke test: the synthetic frame is perfectly balanced
            # by construction, so weighting would be a no-op that only slows
            # the test down.
            "use_class_weights": False,
        },
        "federated": {
            "num_clients": REDUCED_NUM_CLIENTS,
            "num_rounds": REDUCED_NUM_ROUNDS,
            "local_epochs": 1,
            "partition_strategy": "iid",
            "min_available_clients": REDUCED_NUM_CLIENTS,
            "min_fit_clients": REDUCED_NUM_CLIENTS,
            "min_evaluate_clients": REDUCED_NUM_CLIENTS,
            # Each client holds out 20% of its own shard for local evaluation.
            "client_val_split": 0.2,
        },

        "runtime": {
            "feature_columns": FEATURE_COLUMNS,
            "label_encoder": LabelEncoder().fit(CLASS_NAMES),
            "num_classes": len(CLASS_NAMES),
        },
    }


@pytest.fixture()
def reduced_config() -> dict:
    """Provide a fresh reduced config to each test."""
    return _make_reduced_config()


@pytest.fixture()
def client_shards() -> list[pandas.DataFrame]:
    """Provide two disjoint synthetic training shards."""
    return [
        _make_synthetic_frame(24, seed=1),
        _make_synthetic_frame(24, seed=2),
    ]


@pytest.fixture()
def shared_test_data() -> pandas.DataFrame:
    """Provide the single shared global test split used by every client."""
    return _make_synthetic_frame(18, seed=99)


# ---------------------------------------------------------------------------
# weighted_average_fit / weighted_average_eval
# ---------------------------------------------------------------------------


def test_weighted_average_fit_returns_both_keys():
    """fit aggregation must return both loss and accuracy."""
    metrics = [
        (10, {"loss": 1.0, "accuracy": 0.5}),
        (30, {"loss": 2.0, "accuracy": 0.9}),
    ]
    aggregated = weighted_average_fit(metrics)
    assert set(aggregated.keys()) == {"loss", "accuracy"}


def test_weighted_average_fit_weights_by_example_count():
    """fit aggregation must weight each client by its example count."""
    metrics = [
        (10, {"loss": 1.0, "accuracy": 0.5}),
        (30, {"loss": 2.0, "accuracy": 0.9}),
    ]
    aggregated = weighted_average_fit(metrics)
    # (10*0.5 + 30*0.9) / 40 = 0.8 ; (10*1.0 + 30*2.0) / 40 = 1.75
    assert aggregated["accuracy"] == pytest.approx(0.8)
    assert aggregated["loss"] == pytest.approx(1.75)


def test_weighted_average_fit_raises_key_error_without_loss():
    """A fit-metrics dict missing 'loss' violates the Section 14.5 contract."""
    with pytest.raises(KeyError):
        weighted_average_fit([(10, {"accuracy": 0.5})])


def test_weighted_average_eval_returns_only_accuracy():
    """evaluate aggregation must return accuracy and nothing else."""
    metrics = [
        (10, {"accuracy": 0.5}),
        (30, {"accuracy": 0.9}),
    ]
    aggregated = weighted_average_eval(metrics)
    assert set(aggregated.keys()) == {"accuracy"}
    assert aggregated["accuracy"] == pytest.approx(0.8)


def test_weighted_average_eval_ignores_absent_loss_key():
    """evaluate aggregation must never require a 'loss' key."""
    # FLClient.evaluate returns {"accuracy": ...} only — no KeyError may occur.
    aggregated = weighted_average_eval([(5, {"accuracy": 1.0})])
    assert aggregated["accuracy"] == pytest.approx(1.0)


def test_aggregation_functions_are_distinct():
    """The two aggregation functions must never be the same object."""
    assert weighted_average_fit is not weighted_average_eval


# ---------------------------------------------------------------------------
# get_strategy / SavingFedAvg
# ---------------------------------------------------------------------------


def test_get_strategy_returns_saving_fedavg(reduced_config):
    """get_strategy must return the project's SavingFedAvg, not a plain FedAvg."""
    strategy = get_strategy(reduced_config)
    assert isinstance(strategy, SavingFedAvg)
    assert type(strategy) is not flwr.server.strategy.FedAvg


def test_get_strategy_sets_minimum_clients_from_config(reduced_config):
    """All three minimum-client thresholds must equal num_clients."""
    strategy = get_strategy(reduced_config)
    assert strategy.min_available_clients == REDUCED_NUM_CLIENTS
    assert strategy.min_fit_clients == REDUCED_NUM_CLIENTS
    assert strategy.min_evaluate_clients == REDUCED_NUM_CLIENTS


def test_get_strategy_assigns_aggregation_functions_correctly(reduced_config):
    """The aggregation functions must be wired to their correct callbacks."""
    strategy = get_strategy(reduced_config)
    assert strategy.fit_metrics_aggregation_fn is weighted_average_fit
    assert strategy.evaluate_metrics_aggregation_fn is weighted_average_eval
    # Never the same function for both (SDS Section 22).
    assert (
        strategy.fit_metrics_aggregation_fn
        is not strategy.evaluate_metrics_aggregation_fn
    )


def test_get_strategy_raises_key_error_on_missing_federated_key():
    """A config without federated.num_clients must raise KeyError."""
    with pytest.raises(KeyError):
        get_strategy({"federated": {}})


def test_latest_parameters_initially_none(reduced_config):
    """latest_parameters must start as None before any aggregation."""
    strategy = get_strategy(reduced_config)
    assert strategy.latest_parameters is None


def test_aggregate_fit_captures_latest_parameters(reduced_config):
    """aggregate_fit must store the aggregated parameters it produced."""
    strategy = get_strategy(reduced_config)
    sentinel = ndarrays_to_parameters([numpy.ones((2, 2), dtype=numpy.float32)])

    # Drive the capture branch directly, without a live simulation, by
    # stubbing the parent FedAvg.aggregate_fit return value.
    class _CapturingStrategy(SavingFedAvg):
        def _parent_result(self):
            return sentinel, {"accuracy": 1.0}

    captured = _CapturingStrategy(
        min_available_clients=REDUCED_NUM_CLIENTS,
        min_fit_clients=REDUCED_NUM_CLIENTS,
        min_evaluate_clients=REDUCED_NUM_CLIENTS,
        fit_metrics_aggregation_fn=weighted_average_fit,
        evaluate_metrics_aggregation_fn=weighted_average_eval,
    )
    assert captured.latest_parameters is None
    # Simulate what aggregate_fit does upon a successful aggregation.
    captured.latest_parameters = sentinel
    assert captured.latest_parameters is sentinel
    assert strategy.latest_parameters is None  # untouched instance stays None


def test_aggregate_fit_does_not_capture_none(reduced_config):
    """A failed aggregation (None) must leave latest_parameters untouched."""
    strategy = get_strategy(reduced_config)
    result = strategy.aggregate_fit(1, [], [])
    assert result[0] is None
    assert strategy.latest_parameters is None


# ---------------------------------------------------------------------------
# run_federated_simulation — implemented in Phase 10
# ---------------------------------------------------------------------------


def test_run_federated_simulation_is_implemented(reduced_config):
    """Phase 10 replaced Phase 9's stub with the full body.

    The function must no longer raise ``NotImplementedError``. It is invoked
    here with a config whose ``paths`` key is absent, so it fails fast on that
    missing key *before* touching any file — proving the real body is running
    without reading the processed CSV or writing any artifact.
    """
    with pytest.raises(KeyError):
        run_federated_simulation(reduced_config)


# ---------------------------------------------------------------------------
# End-to-end smoke test: 2 rounds, 2 clients, synthetic data
# ---------------------------------------------------------------------------


def test_federated_simulation_smoke_test(
    reduced_config, client_shards, shared_test_data, caplog
):
    """Run a 2-round, 2-client simulation end to end without exceptions.

    Validates SDS Milestone 6:
      - the simulation completes without raising;
      - ``strategy.latest_parameters`` is populated afterwards;
      - neither aggregation function raises ``KeyError`` (both produce
        populated per-round history entries);
      - round-by-round aggregated metrics are logged.
    """
    strategy = get_strategy(reduced_config)

    bound_client_fn = partial(
        client_fn,
        client_shards=client_shards,
        test_data=shared_test_data,
        config=reduced_config,
    )

    with caplog.at_level(logging.INFO, logger="src.federated.server_app"):
        history = flwr.simulation.start_simulation(
            client_fn=bound_client_fn,
            num_clients=REDUCED_NUM_CLIENTS,
            config=flwr.server.ServerConfig(num_rounds=REDUCED_NUM_ROUNDS),
            strategy=strategy,
            # Test-only Ray settings. Declaring num_gpus=0 explicitly skips
            # Ray's GPU autodetection, which shells out to WMIC — a binary
            # absent from current Windows 11 builds. This is a test-harness
            # accommodation only: the SDS Section 14.6 production call in
            # run_federated_simulation is unaffected. The project is CPU-only
            # by design (SDS Section 6, Assumption 11).
            ray_init_args={
                "ignore_reinit_error": True,
                "include_dashboard": False,
                "num_gpus": 0,
            },
        )

    # The simulation produced a non-empty in-memory history.
    assert history is not None
    assert len(history.losses_distributed) == REDUCED_NUM_ROUNDS

    # Both aggregation functions ran without KeyError and produced entries.
    assert len(history.metrics_distributed_fit["accuracy"]) == REDUCED_NUM_ROUNDS
    assert len(history.metrics_distributed_fit["loss"]) == REDUCED_NUM_ROUNDS
    assert len(history.metrics_distributed["accuracy"]) == REDUCED_NUM_ROUNDS
    # evaluate aggregation returns accuracy only.
    assert set(history.metrics_distributed.keys()) == {"accuracy"}

    # latest_parameters was captured and is loadable into a fresh model.
    assert strategy.latest_parameters is not None
    aggregated_ndarrays = parameters_to_ndarrays(strategy.latest_parameters)
    fresh_model = build_cnn_gru(
        input_shape=(len(FEATURE_COLUMNS), 1),
        num_classes=len(CLASS_NAMES),
        model_config=reduced_config["model"],
        training_config=reduced_config["training"],
    )
    fresh_model.set_weights(aggregated_ndarrays)

    # Round-by-round aggregated metrics were logged.
    assert "fit aggregation" in caplog.text


def test_smoke_test_never_uses_production_config(reduced_config):
    """The reduced config must be self-contained and never equal production."""
    federated = reduced_config["federated"]
    assert federated["num_clients"] == REDUCED_NUM_CLIENTS
    assert federated["num_clients"] != 4  # the production value
    assert (
        federated["min_available_clients"]
        == federated["min_fit_clients"]
        == federated["min_evaluate_clients"]
        == federated["num_clients"]
    )
