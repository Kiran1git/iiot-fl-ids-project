"""Federated server, strategy, and simulation wiring — Phase 9.

This module owns the project's single federated aggregation strategy and its
two distinct metrics-aggregation functions (SDS Section 14.6).

Ownership and reuse rules:
  - ``SavingFedAvg`` is the **sole** strategy class. A plain
    ``flwr.server.strategy.FedAvg`` must never be instantiated directly for
    production use anywhere, because a stock ``FedAvg`` does not expose the
    final round's aggregated parameters after the simulation ends.
  - ``weighted_average_fit`` and ``weighted_average_eval`` are **two distinct
    functions** and must never be collapsed into one. ``weighted_average_fit``
    reads both ``loss`` and ``accuracy`` (the keys ``FLClient.fit`` always
    returns); ``weighted_average_eval`` reads only ``accuracy`` (the single key
    ``FLClient.evaluate`` returns). Their assignment to
    ``fit_metrics_aggregation_fn`` vs. ``evaluate_metrics_aggregation_fn`` must
    never be swapped (SDS Section 14.6, Section 22).
  - The pinned Flower API is ``flwr==1.8.0``'s
    client-function/strategy-based ``start_simulation``. The newer app-based
    ``flwr.simulation.run_simulation(server_app=..., client_app=...,
    num_supernodes=...)`` entry point is permanently forbidden project-wide
    (SDS Section 5, Section 14.6, Section 22).
  - Every simulated client evaluates against the identical shared global test
    split — never a per-client test partition (SDS Section 14.3).

Phase scope: ``run_federated_simulation`` is intentionally a signature-only
stub in this phase. Its full body — data loading, sharding, the
``start_simulation`` call, the ``parameters_to_ndarrays`` →
``build_cnn_gru`` → ``set_weights`` → save sequence, and all artifact
persistence — is deferred to Phase 10, which is the only phase permitted to
modify this file and may add that body ONLY.

Per the SDS Section 11 import graph, ``src/federated/`` imports only from
``src/utils/``, ``src/preprocessing/``, ``src/partitioning/``, and
``src/models/``. It never imports ``src/centralized/``, ``src/evaluation/``,
or ``src/explainability/``.
"""

import logging

from flwr.server.strategy import FedAvg

# Module-level logger only. Modules under src/ never create log files; the five
# fixed log files are created exclusively by experiments/ scripts via
# src.utils.logger.get_logger (SDS Section 13).
_logger = logging.getLogger(__name__)

__all__ = [
    "SavingFedAvg",
    "get_strategy",
    "weighted_average_fit",
    "weighted_average_eval",
    "run_federated_simulation",
]


class SavingFedAvg(FedAvg):
    """A ``FedAvg`` subclass that retains the latest aggregated parameters.

    Purpose:
        Neither ``flwr.simulation.start_simulation``'s return value nor a stock
        ``FedAvg`` instance exposes the final round's aggregated global
        parameters once the simulation ends. This minimal subclass captures
        them on every successful aggregation so Phase 10 can rebuild and
        persist the federated global model from ``latest_parameters``
        (SDS Section 14.6).

        This is the project's only strategy class — a plain ``FedAvg`` is never
        instantiated directly for production use.

    Attributes:
        latest_parameters: The most recently aggregated global parameters as a
            ``flwr.common.Parameters`` object, or ``None`` before the first
            successful ``aggregate_fit`` call.
    """

    def __init__(self, *args, **kwargs) -> None:
        """Initialize the strategy and the parameter-capture slot.

        Purpose:
            Forward every argument untouched to ``FedAvg.__init__`` and add the
            ``latest_parameters`` capture slot, initialised to ``None``.

        Args:
            *args: Positional arguments forwarded verbatim to ``FedAvg``.
            **kwargs: Keyword arguments forwarded verbatim to ``FedAvg``.

        Returns:
            None

        Raises:
            Nothing beyond what ``FedAvg.__init__`` itself raises.
        """
        super().__init__(*args, **kwargs)
        self.latest_parameters = None

    def aggregate_fit(self, server_round, results, failures):
        """Aggregate client weights for one round and capture the result.

        Purpose:
            Delegate aggregation to ``FedAvg.aggregate_fit`` and store the
            aggregated parameters in ``self.latest_parameters`` whenever the
            aggregation succeeds, so the final global model survives the end of
            the simulation.

        Args:
            server_round: The 1-based federated round number.
            results: Successful ``(ClientProxy, FitRes)`` pairs from this round.
            failures: Failed client results/exceptions from this round.

        Returns:
            tuple: ``(aggregated_parameters, aggregated_metrics)`` exactly as
                returned by ``FedAvg.aggregate_fit`` — the return value is
                never altered, only observed.

        Raises:
            Nothing beyond what ``FedAvg.aggregate_fit`` itself raises.
        """
        aggregated_parameters, aggregated_metrics = super().aggregate_fit(
            server_round, results, failures
        )

        if aggregated_parameters is not None:
            self.latest_parameters = aggregated_parameters

        _logger.info(
            "Round %s fit aggregation — clients=%d failures=%d metrics=%s",
            server_round,
            len(results),
            len(failures),
            aggregated_metrics,
        )

        return aggregated_parameters, aggregated_metrics


def weighted_average_fit(metrics: list[tuple[int, dict]]) -> dict:
    """Aggregate fit-time client metrics weighted by example count.

    Purpose:
        The ``fit_metrics_aggregation_fn`` for ``SavingFedAvg``. Operates on
        ``FLClient.fit()``'s metrics dicts, which always contain **both**
        ``"loss"`` and ``"accuracy"`` (SDS Section 14.5). This function is
        never used for evaluate-time aggregation — that is
        ``weighted_average_eval``'s exclusive role.

    Args:
        metrics: One ``(num_examples, metrics_dict)`` pair per participating
            client, as supplied by Flower.

    Returns:
        dict: ``{"accuracy": weighted_accuracy, "loss": weighted_loss}``, each
            value weighted by the contributing client's example count.

    Raises:
        Nothing under normal operation. A ``KeyError`` would indicate a client
        returned a fit-metrics dict missing ``loss`` or ``accuracy``, which
        violates the Section 14.5 contract.
    """
    total_examples = sum(num_examples for num_examples, _ in metrics)
    weighted_accuracy = (
        sum(num_examples * m["accuracy"] for num_examples, m in metrics)
        / total_examples
    )
    weighted_loss = (
        sum(num_examples * m["loss"] for num_examples, m in metrics)
        / total_examples
    )
    return {"accuracy": weighted_accuracy, "loss": weighted_loss}


def weighted_average_eval(metrics: list[tuple[int, dict]]) -> dict:
    """Aggregate evaluate-time client metrics weighted by example count.

    Purpose:
        The ``evaluate_metrics_aggregation_fn`` for ``SavingFedAvg``. Operates
        on ``FLClient.evaluate()``'s metrics dicts, which contain only
        ``"accuracy"`` (SDS Section 14.5). Evaluate-time loss aggregation is
        handled natively by Flower via each client's returned ``loss`` tuple
        element (into ``history.losses_distributed``) and requires no custom
        function here. This function is never used for fit-time aggregation —
        that is ``weighted_average_fit``'s exclusive role.

    Args:
        metrics: One ``(num_examples, metrics_dict)`` pair per participating
            client, as supplied by Flower.

    Returns:
        dict: ``{"accuracy": weighted_accuracy}``, weighted by each
            contributing client's example count.

    Raises:
        Nothing under normal operation. A ``KeyError`` would indicate a client
        returned an evaluate-metrics dict missing ``accuracy``, which violates
        the Section 14.5 contract.
    """
    total_examples = sum(num_examples for num_examples, _ in metrics)
    weighted_accuracy = (
        sum(num_examples * m["accuracy"] for num_examples, m in metrics)
        / total_examples
    )
    return {"accuracy": weighted_accuracy}


def get_strategy(config: dict) -> SavingFedAvg:
    """Construct the federated aggregation strategy from the config.

    Purpose:
        Build the project's ``SavingFedAvg`` strategy with every minimum-client
        threshold pinned to ``config["federated"]["num_clients"]`` and the two
        distinct aggregation functions wired to their correct callback sites
        (SDS Section 14.6).

        A plain ``flwr.server.strategy.FedAvg`` is never returned: it does not
        expose the final round's aggregated parameters, which Phase 10 requires
        to build ``federated_global_model.h5``.

    Args:
        config: The loaded config dict. Only ``config["federated"]
            ["num_clients"]`` is read — the production value is 4; the smoke
            test supplies its own independently-constructed reduced config with
            2 (SDS Section 19). Pinning all three minimums to that same
            ``num_clients`` value guarantees the thresholds are always
            satisfiable by exactly the number of clients being simulated.

    Returns:
        SavingFedAvg: The configured strategy, with
            ``fit_metrics_aggregation_fn=weighted_average_fit`` and
            ``evaluate_metrics_aggregation_fn=weighted_average_eval`` — never
            swapped, never the same function for both.

    Raises:
        KeyError: If ``config`` lacks ``federated.num_clients``.
    """
    num_clients = config["federated"]["num_clients"]

    strategy = SavingFedAvg(
        min_available_clients=num_clients,
        min_fit_clients=num_clients,
        min_evaluate_clients=num_clients,
        fit_metrics_aggregation_fn=weighted_average_fit,
        evaluate_metrics_aggregation_fn=weighted_average_eval,
    )

    _logger.info(
        "SavingFedAvg strategy constructed — min_available_clients=%d "
        "min_fit_clients=%d min_evaluate_clients=%d "
        "fit_metrics_aggregation_fn=%s evaluate_metrics_aggregation_fn=%s",
        num_clients,
        num_clients,
        num_clients,
        weighted_average_fit.__name__,
        weighted_average_eval.__name__,
    )

    return strategy


def run_federated_simulation(config: dict) -> None:
    """Run the full federated simulation and persist the global model.

    Purpose:
        The project's single top-level federated orchestrator: partition the
        training split, launch ``flwr.simulation.start_simulation`` for
        ``config["federated"]["num_rounds"]`` rounds using the pinned
        ``flwr==1.8.0`` API, capture the final aggregated global model from
        ``SavingFedAvg.latest_parameters``, time the run, and persist all
        federated artifacts (SDS Section 14.6).

        **Phase 9 scope:** this is a signature-only stub. The full body is
        deferred to Phase 10, which is the only phase permitted to modify this
        file and may add this body ONLY — no other function here may be
        altered. No second simulation entry point may ever be introduced.

    Args:
        config: The loaded config dict.

    Returns:
        None

    Raises:
        NotImplementedError: Always, at this phase. The implementation lands in
            Phase 10 once the reduced-scale smoke test in
            ``tests/test_federated_loop.py`` is green.
    """
    raise NotImplementedError(
        "run_federated_simulation's body is deferred to Phase 10 (SDS Section "
        "14.6). Phase 9 defines only its signature, alongside SavingFedAvg, "
        "get_strategy, weighted_average_fit, and weighted_average_eval."
    )
