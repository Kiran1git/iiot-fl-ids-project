"""Federated server, strategy, and simulation wiring.

Owns the project's single federated aggregation strategy and its two distinct
metrics-aggregation functions (SDS Section 14.6).

Ownership and reuse rules:
  - ``SavingFedAvg`` is the **sole** strategy class. A stock
    ``flwr.server.strategy.FedAvg`` must never be used in production, because
    it does not expose the final round's aggregated parameters after the
    simulation ends.
  - ``weighted_average_fit`` and ``weighted_average_eval`` are **two distinct
    functions** and must never be collapsed or swapped. Fit reads both ``loss``
    and ``accuracy`` (the keys ``FLClient.fit`` returns); eval reads only
    ``accuracy`` (the single key ``FLClient.evaluate`` returns).
  - The pinned API is ``flwr==1.8.0``'s ``start_simulation``. The newer
    app-based ``flwr.simulation.run_simulation`` is permanently forbidden
    project-wide (SDS Section 5 / 14.6 / 22).

``run_federated_simulation`` is the single top-level federated orchestrator: it
shards the training split via ``partition_iid``, runs the simulation, and
rebuilds the global model from ``SavingFedAvg.latest_parameters`` — never from
a per-client local model.
"""

import gc
import json
import logging
import os
import pickle
import time
from functools import partial


import flwr
import pandas
from flwr.common import parameters_to_ndarrays
from flwr.server.strategy import FedAvg

from src.federated.client_app import client_fn
from src.models.cnn_gru import (
    build_cnn_gru,
    get_model_summary_string,
    save_model_architecture_diagram,
)
from src.partitioning.partition_data import partition_iid
from src.preprocessing.encode_normalize import prepare_model_ready_data
from src.utils.dataio import read_processed


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

    Neither ``start_simulation``'s return value nor a stock ``FedAvg`` exposes
    the final round's aggregated global parameters once the simulation ends.
    This subclass captures them on every successful aggregation so the
    federated global model can be rebuilt from ``latest_parameters``
    (SDS Section 14.6). It is the project's only strategy class.

    Attributes:
        latest_parameters: The most recently aggregated global parameters as a
            ``flwr.common.Parameters`` object, or ``None`` before the first
            successful ``aggregate_fit`` call.
    """

    def __init__(self, *args, **kwargs) -> None:
        """Initialize the strategy and the parameter-capture slot.

        Args:
            *args: Positional arguments forwarded verbatim to ``FedAvg``.
            **kwargs: Keyword arguments forwarded verbatim to ``FedAvg``.
        """
        super().__init__(*args, **kwargs)
        self.latest_parameters = None

    def aggregate_fit(self, server_round, results, failures):
        """Aggregate client weights for one round and capture the result.

        Delegates to ``FedAvg.aggregate_fit`` and stores the aggregated
        parameters whenever aggregation succeeds, so the final global model
        survives the end of the simulation.

        Args:
            server_round: The 1-based federated round number.
            results: Successful ``(ClientProxy, FitRes)`` pairs from this round.
            failures: Failed client results/exceptions from this round.

        Returns:
            tuple: ``(aggregated_parameters, aggregated_metrics)`` exactly as
                returned by ``FedAvg.aggregate_fit`` — never altered, only
                observed.
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

    The ``fit_metrics_aggregation_fn`` for ``SavingFedAvg``, operating on
    ``FLClient.fit()``'s dicts, which always contain **both** ``"loss"`` and
    ``"accuracy"`` (SDS Section 14.5). Never used for evaluate-time
    aggregation — that is ``weighted_average_eval``'s exclusive role.

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

    The ``evaluate_metrics_aggregation_fn`` for ``SavingFedAvg``, operating on
    ``FLClient.evaluate()``'s dicts, which contain only ``"accuracy"``
    (SDS Section 14.5). Evaluate-time loss needs no custom function: Flower
    aggregates each client's returned ``loss`` natively into
    ``history.losses_distributed``. Never used for fit-time aggregation.

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


def build_ray_init_args(config: dict) -> dict:
    """Assemble the ``ray_init_args`` dict for ``start_simulation``.

    Ray on Windows backs its object store with a memory-mapped file sized by
    default at ~30% of physical RAM. On a 16 GB laptop that is a ~4.8 GB
    mapping requested up front, and once TensorFlow's client actors have taken
    their share Windows can no longer commit it::

        CreateFileMapping() failed. GetLastError() = 1450

    which is ``ERROR_NO_SYSTEM_RESOURCES`` and kills the raylet mid-run.
    Capping ``object_store_memory`` keeps the mapping small enough to succeed.

    When ``federated.ray`` is absent — the untouched production
    ``configs/config.yaml`` case — the returned settings are exactly the
    project's originals, so FULL_EXPERIMENT is bit-for-bit unchanged.

    Args:
        config: The loaded config dict. Reads the optional
            ``federated.ray`` sub-dict: ``object_store_memory_mb``,
            ``memory_mb``, ``num_cpus``.

    Returns:
        dict: Arguments forwarded verbatim to ``ray.init`` by Flower.
    """
    # Baseline, identical to the pre-existing production behaviour.
    # num_gpus=0 skips Ray's GPU autodetection, which shells out to WMIC — a
    # binary absent from current Windows 11 builds (SDS Section 6,
    # Assumption 11).
    ray_init_args = {
        "ignore_reinit_error": True,
        "include_dashboard": False,
        "num_gpus": 0,
    }

    ray_config = config["federated"].get("ray") or {}

    megabyte = 1024 * 1024
    if ray_config.get("object_store_memory_mb"):
        ray_init_args["object_store_memory"] = int(
            ray_config["object_store_memory_mb"] * megabyte
        )
    if ray_config.get("memory_mb"):
        ray_init_args["_memory"] = int(ray_config["memory_mb"] * megabyte)
    if ray_config.get("num_cpus"):
        ray_init_args["num_cpus"] = int(ray_config["num_cpus"])

    return ray_init_args


def build_client_resources(config: dict):
    """Assemble Flower's per-client Ray actor resource request.

    Flower derives the number of *concurrent* client actors as
    ``floor(ray_num_cpus / client_num_cpus)``. Every concurrent actor holds its
    own TensorFlow runtime (~350-500 MB), CNN-GRU graph, and cached tensors, so
    raising ``client_num_cpus`` *lowers* peak RAM by serialising the clients.
    In LAPTOP_MODE the overlay makes exactly one client train at a time.

    This changes nothing mathematically: clients within a FedAvg round are
    independent, so sequential execution yields an identical aggregate.

    Args:
        config: The loaded config dict. Reads the optional
            ``federated.client_num_cpus`` / ``federated.client_num_gpus``.

    Returns:
        dict | None: The ``client_resources`` mapping, or ``None`` to keep
            Flower's default (1 CPU per client) when the keys are absent — the
            FULL_EXPERIMENT behaviour.
    """
    federated_config = config["federated"]

    if "client_num_cpus" not in federated_config:
        return None

    return {
        "num_cpus": float(federated_config["client_num_cpus"]),
        "num_gpus": float(federated_config.get("client_num_gpus", 0)),
    }


def subsample_for_server_eval(
    test_data: pandas.DataFrame,
    config: dict,
    logger: logging.Logger,
) -> pandas.DataFrame:
    """Optionally shrink the server-side centralized evaluation split.

    These tensors stay resident for the whole simulation: ~443k test rows x 95
    features x 4 bytes is roughly 168 MB of float32 held alongside every client
    actor. ``runtime_profile.server_eval_max_rows`` caps that (50k rows,
    ~19 MB, in LAPTOP_MODE).

    The subsample is stratified on ``label`` and drawn with the project seed,
    so the estimate stays representative and reproducible; at 50k rows the
    sampling error on an accuracy figure is well under +/-0.5%. When the key is
    absent — the FULL_EXPERIMENT case — the full split is returned unchanged.

    Args:
        test_data: The shared global test split.
        config: The loaded config dict.
        logger: Logger used to record whether subsampling occurred.

    Returns:
        pandas.DataFrame: Either ``test_data`` itself or a stratified sample.
    """
    max_rows = (config.get("runtime_profile") or {}).get(
        "server_eval_max_rows", 0
    )

    if not max_rows or len(test_data) <= max_rows:
        logger.info(
            "Server-side evaluation uses the FULL shared global test split "
            "(%d rows).",
            len(test_data),
        )
        return test_data

    fraction = max_rows / len(test_data)
    sampled = (
        test_data.groupby("label", group_keys=False)
        .apply(
            lambda group: group.sample(
                # At least one row per class, so a rare class is never dropped
                # from the evaluation entirely.
                n=max(1, int(round(len(group) * fraction))),
                random_state=config["seed"],
            )
        )
    )
    logger.info(
        "Server-side evaluation uses a stratified subsample of the shared "
        "global test split: %d of %d rows (runtime_profile."
        "server_eval_max_rows=%d). Frees roughly %.0f MB for the whole run.",
        len(sampled),
        len(test_data),
        max_rows,
        (len(test_data) - len(sampled))
        * len(test_data.columns)
        * 4
        / (1024 * 1024),
    )
    return sampled


def make_server_side_evaluate_fn(

    config: dict,
    X_test,
    y_test,
    num_features: int,
    num_classes: int,
    logger: logging.Logger,
):
    """Build the server-side centralized evaluation callback.

    Distributed evaluation answers "how does the global model do on each
    client's local data?"; this hook answers the more authoritative question of
    how the *aggregated* model scores on the shared global test split, once, on
    the server, with no per-client weighting artefacts. Both matter, because a
    divergence between them is the standard symptom of aggregation going wrong
    (weights averaged in the wrong order, a client returning stale parameters),
    and with only distributed evaluation that failure is invisible.

    The returned closure binds ``X_test``/``y_test`` once, so the test tensors
    are built a single time for the whole simulation rather than per round.

    Args:
        config: Loaded config dict; ``training.batch_size`` is read.
        X_test: Shared global test features, shape ``(n, num_features, 1)``.
        y_test: Shared global test labels, one-hot, shape ``(n, num_classes)``.
        num_features: Feature count for rebuilding the model shell.
        num_classes: Class count for rebuilding the model shell.
        logger: Logger to record per-round centralized metrics.

    Returns:
        Callable: A Flower ``evaluate_fn`` with signature
            ``(server_round, parameters_ndarrays, config) ->
            (loss, {"accuracy": acc})``.
    """
    # The model shell is built once and reused across rounds; only the weights
    # change. Rebuilding a Keras model 15 times would add graph-construction
    # overhead to every round for no benefit.
    eval_model = build_cnn_gru(
        input_shape=(num_features, 1),
        num_classes=num_classes,
        model_config=config["model"],
        training_config=config["training"],
    )
    batch_size = config["training"]["batch_size"]

    def evaluate_fn(server_round, parameters_ndarrays, eval_config):
        eval_model.set_weights(parameters_ndarrays)
        loss, accuracy = eval_model.evaluate(
            X_test, y_test, batch_size=batch_size, verbose=0
        )
        logger.info(
            "Round %d — SERVER-SIDE centralized evaluation on the shared "
            "global test split: loss=%.6f accuracy=%.6f (%d samples)",
            server_round,
            loss,
            accuracy,
            len(X_test),
        )
        return float(loss), {"accuracy": float(accuracy)}

    return evaluate_fn


def get_strategy(config: dict, evaluate_fn=None) -> SavingFedAvg:

    """Construct the federated aggregation strategy from the config.

    Builds ``SavingFedAvg`` with every minimum-client threshold pinned to
    ``num_clients`` and the two distinct aggregation functions wired to their
    correct callback sites (SDS Section 14.6). A stock ``FedAvg`` is never
    returned: it does not expose the final round's aggregated parameters,
    which building ``federated_global_model.h5`` requires.

    Args:
        config: The loaded config dict. Only ``federated.num_clients`` is read;
            pinning all three minimums to it guarantees the thresholds are
            always satisfiable by exactly the number of clients simulated.

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
        # Server-side centralized evaluation. Optional so the smoke test and
        # the unit tests can construct a strategy without building tensors.
        evaluate_fn=evaluate_fn,
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


def run_federated_simulation(
    config: dict,
    logger: logging.Logger = None,
) -> None:
    """Run the full federated simulation and persist the global model.

    The single top-level federated orchestrator (SDS Section 14.6): shards the
    training split via ``partition_iid``, runs ``start_simulation`` for
    ``federated.num_rounds`` rounds under the pinned ``flwr==1.8.0`` API, times
    only that call, and persists every federated artifact in SDS Section 9.

    ``federated_global_model.h5`` is *always* built by converting
    ``strategy.latest_parameters`` via ``parameters_to_ndarrays`` into a fresh
    ``build_cnn_gru`` model. A per-client local model is never saved under that
    filename (SDS Section 14.6 / 22). ``partition_iid`` is applied to training
    rows exclusively (SDS Section 14.3).

    Args:
        config: Fully loaded config dict from ``load_config()``.
        logger: Optional logger from ``experiments/run_federated.py``. Modules
            under ``src/`` never construct their own log file (SDS Section 13),
            so omitting it yields a handler-less logger.

    Raises:
        RuntimeError: If ``strategy.latest_parameters`` is still ``None`` after
            the simulation, meaning no round ever aggregated successfully and
            no global model can be built.
        Propagates any Flower simulation exception (and any I/O exception
        raised while loading artifacts or persisting outputs) after logging the
        full stack trace at ERROR level.
    """
    if logger is None:
        logger = logging.getLogger("federated_training")

    try:
        artifacts_dir = config["paths"]["artifacts_dir"]
        models_dir = config["paths"]["models_dir"]
        results_dir = config["paths"]["results_dir"]

        os.makedirs(models_dir, exist_ok=True)
        os.makedirs(results_dir, exist_ok=True)

        num_clients = config["federated"]["num_clients"]
        num_rounds = config["federated"]["num_rounds"]

        logger.info("=== run_federated_simulation starting ===")
        logger.info(
            "Run profile: %s", config.get("run_mode", "FULL_EXPERIMENT")
        )
        logger.info(
            "Federated config summary — num_clients=%s | num_rounds=%s | "
            "local_epochs=%s | partition_strategy=%s | batch_size=%s",
            num_clients,
            num_rounds,
            config["federated"]["local_epochs"],
            config["federated"]["partition_strategy"],
            config["training"]["batch_size"],
        )


        # ---- Load processed dataset -------------------------------------
        processed_csv_path = os.path.join(
            config["paths"]["processed_data_dir"],
            config["paths"]["processed_data_file"],
        )
        logger.info("Loading processed dataset from '%s'", processed_csv_path)
        df = read_processed(processed_csv_path)

        logger.info(
            "Processed dataset loaded: %d rows x %d columns",
            df.shape[0],
            df.shape[1],
        )

        # ---- Load preprocessing artifacts -------------------------------
        def _load_pickle(filename: str):
            path = os.path.join(artifacts_dir, filename)
            with open(path, "rb") as handle:
                return pickle.load(handle)

        train_indices = _load_pickle("train_indices.pkl")
        test_indices = _load_pickle("test_indices.pkl")
        feature_columns = _load_pickle("feature_names.pkl")
        label_encoder = _load_pickle("label_encoder.pkl")

        class_mapping_path = os.path.join(artifacts_dir, "class_mapping.json")
        with open(class_mapping_path, "r", encoding="utf-8") as handle:
            class_mapping = json.load(handle)

        # num_classes is always derived, never hardcoded (SDS Section 6).
        num_classes = len(class_mapping)
        num_features = len(feature_columns)
        logger.info(
            "Artifacts loaded — train=%d rows | test=%d rows | features=%d | "
            "classes=%d",
            len(train_indices),
            len(test_indices),
            num_features,
            num_classes,
        )

        # ---- Shard the TRAINING split only (sole owner: partition_iid) ---
        # The shared global test split is never partitioned per client
        # (SDS Section 14.3).
        client_shards = partition_iid(
            df.loc[train_indices], num_clients, config["seed"]
        )
        test_data_global = df.loc[test_indices]
        logger.info(
            "Partitioned training split into %d IID shards — sizes=%s | "
            "shared global test split=%d rows",
            len(client_shards),
            [len(shard) for shard in client_shards],
            len(test_data_global),
        )

        # ---- Bind the client factory's closure inputs --------------------
        # The shards, the shared test split, and the config are bound once so
        # Flower's engine only ever supplies `cid`; nothing is re-read from
        # disk per client (SDS Section 14.5).
        client_runtime_config = dict(config)
        client_runtime_config["runtime"] = {
            "feature_columns": feature_columns,
            "label_encoder": label_encoder,
            "num_classes": num_classes,
        }
        bound_client_fn = partial(
            client_fn,
            client_shards=client_shards,
            test_data=test_data_global,
            config=client_runtime_config,
        )

        # ---- Server-side centralized evaluation --------------------------
        # Built once, outside the timed region, so the tensor construction
        # cost is not attributed to federated training time.
        #
        # In LAPTOP_MODE this uses a stratified subsample of the test split,
        # because these tensors stay resident for the entire simulation and
        # the full split is ~168 MB of float32. In FULL_EXPERIMENT the split
        # is returned untouched.
        server_eval_data = subsample_for_server_eval(
            test_data_global, config, logger
        )
        X_test_global, y_test_global = prepare_model_ready_data(
            server_eval_data,
            server_eval_data.index.values,
            feature_columns,
            label_encoder,
            num_classes,
        )

        server_evaluate_fn = make_server_side_evaluate_fn(
            config,
            X_test_global,
            y_test_global,
            num_features,
            num_classes,
            logger,
        )

        # ---- Run the simulation (time.time() wraps ONLY this call) -------
        strategy = get_strategy(config, evaluate_fn=server_evaluate_fn)

        # Ray/actor sizing. In FULL_EXPERIMENT both helpers return the
        # project's original settings (and client_resources is None, i.e.
        # Flower's default), so the cloud path is unchanged. In LAPTOP_MODE
        # they cap the object store and serialise the client actors.
        ray_init_args = build_ray_init_args(config)
        client_resources = build_client_resources(config)

        logger.info(
            "Starting Flower simulation — %d clients x %d rounds | "
            "ray_init_args=%s | client_resources=%s",
            num_clients,
            num_rounds,
            ray_init_args,
            client_resources,
        )
        if client_resources:
            concurrent = int(
                ray_init_args.get("num_cpus", os.cpu_count() or 1)
                // client_resources["num_cpus"]
            )
            logger.info(
                "At most %d client actor(s) will run concurrently — peak RAM "
                "is driven by this number, not by num_clients.",
                max(concurrent, 1),
            )

        start_time = time.time()
        history = flwr.simulation.start_simulation(
            client_fn=bound_client_fn,
            num_clients=num_clients,
            config=flwr.server.ServerConfig(num_rounds=num_rounds),
            strategy=strategy,
            client_resources=client_resources,
            ray_init_args=ray_init_args,
        )
        training_time_seconds = time.time() - start_time

        # ---- Release the simulation's memory before writing artifacts ----
        # Ray keeps its actors (and their TensorFlow runtimes, ~350-500 MB
        # each) alive until shutdown. Saving the .h5, rendering the
        # architecture diagram, and writing the CSVs all allocate, and on a
        # 16 GB laptop doing that while the actors are still resident is what
        # pushes the machine into the CreateFileMapping failure. Shutting Ray
        # down first is safe here because `history` and
        # `strategy.latest_parameters` are plain in-process Python objects
        # that no longer depend on the Ray cluster.
        if (config.get("runtime_profile") or {}).get(
            "release_memory_after_rounds", False
        ):
            try:
                import ray

                if ray.is_initialized():
                    ray.shutdown()
                    logger.info(
                        "Ray shut down after the final round — client actors "
                        "and the object store are released before artifacts "
                        "are written."
                    )
            except Exception:
                # Never let cleanup failure destroy a completed training run.
                logger.warning(
                    "Ray shutdown after the final round failed; continuing to "
                    "artifact persistence.",
                    exc_info=True,
                )

            # The shards and the server-side eval tensors are the largest
            # remaining objects in the parent process and nothing below needs
            # them.
            del client_shards, bound_client_fn
            del X_test_global, y_test_global
            collected = gc.collect()
            logger.info(
                "Released client shards and server-side evaluation tensors "
                "(gc collected %d objects).",
                collected,
            )


        # ---- Per-round logging + history CSV -----------------------------
        # Columns are fixed by SDS Section 14.6:
        # round, aggregated_loss, aggregated_accuracy.
        losses = dict(history.losses_distributed)
        accuracies = dict(history.metrics_distributed.get("accuracy", []))
        # Server-side centralized metrics, recorded alongside the distributed
        # ones so a divergence between the two is visible in the CSV rather
        # than only in the log.
        central_losses = dict(history.losses_centralized)
        central_accuracies = dict(
            history.metrics_centralized.get("accuracy", [])
        )
        history_rows = []
        for server_round in sorted(losses):
            aggregated_loss = losses[server_round]
            aggregated_accuracy = accuracies.get(server_round)
            history_rows.append(
                {
                    "round": server_round,
                    "aggregated_loss": aggregated_loss,
                    "aggregated_accuracy": aggregated_accuracy,
                    "centralized_loss": central_losses.get(server_round),
                    "centralized_accuracy": central_accuracies.get(
                        server_round
                    ),
                }
            )
            logger.info(
                "Round %d — distributed: loss=%.6f accuracy=%.6f | "
                "centralized: loss=%s accuracy=%s",
                server_round,
                aggregated_loss,
                aggregated_accuracy,
                central_losses.get(server_round),
                central_accuracies.get(server_round),
            )


        history_df = pandas.DataFrame(history_rows)
        logger.info(
            "Simulation complete — rounds_run=%d | total_training_time=%.4f "
            "seconds",
            len(history_df),
            training_time_seconds,
        )

        # ---- Global-weight capture (mandatory — SDS Section 14.6) --------
        if strategy.latest_parameters is None:
            raise RuntimeError(
                "strategy.latest_parameters is None after the simulation — no "
                "round aggregated successfully, so no federated global model "
                "can be built."
            )

        aggregated_ndarrays = parameters_to_ndarrays(strategy.latest_parameters)
        global_model = build_cnn_gru(
            input_shape=(num_features, 1),
            num_classes=num_classes,
            model_config=config["model"],
            training_config=config["training"],
        )
        global_model.set_weights(aggregated_ndarrays)
        logger.info(
            "Global model rebuilt from strategy.latest_parameters "
            "(%d weight tensors) — never from a per-client local model.",
            len(aggregated_ndarrays),
        )

        global_model_path = os.path.join(
            models_dir, "federated_global_model.h5"
        )
        global_model.save(global_model_path)
        logger.info(
            "Saved federated global model to '%s'", global_model_path
        )

        # ---- Persist model summary and architecture diagram --------------
        summary_text = get_model_summary_string(global_model)
        summary_path = os.path.join(models_dir, "federated_model_summary.txt")
        with open(summary_path, "w", encoding="utf-8") as handle:
            handle.write(summary_text)
        logger.info("Saved model summary to '%s'", summary_path)

        diagram_path = os.path.join(models_dir, "federated_architecture.png")
        save_model_architecture_diagram(global_model, diagram_path)
        if os.path.exists(diagram_path):
            logger.info("Saved architecture diagram to '%s'", diagram_path)
        else:
            logger.warning(
                "Architecture diagram was not produced at '%s' "
                "(pydot/graphviz likely unavailable; non-fatal).",
                diagram_path,
            )

        # ---- Persist history CSV -----------------------------------------
        history_csv_path = os.path.join(results_dir, "federated_history.csv")
        history_df.to_csv(history_csv_path, index=False)
        logger.info(
            "Saved federated history (%d rows) to '%s'",
            len(history_df),
            history_csv_path,
        )

        # ---- Persist training time ---------------------------------------
        training_time_path = os.path.join(
            results_dir, "federated_training_time.txt"
        )
        with open(training_time_path, "w", encoding="utf-8") as handle:
            handle.write(str(training_time_seconds))
        logger.info(
            "Saved training time (%.4f s) to '%s'",
            training_time_seconds,
            training_time_path,
        )

        logger.info("=== run_federated_simulation complete ===")

    except KeyboardInterrupt:
        # KeyboardInterrupt derives from BaseException, not Exception, so the
        # handler below would never see it. Logging it explicitly means a
        # manual Ctrl+C leaves a visible record instead of a log that simply
        # stops mid-run with no explanation.
        logger.warning(
            "run_federated_simulation interrupted by user (KeyboardInterrupt). "
            "Any artifact logged as saved above is complete and valid on disk."
        )
        raise
    except Exception:
        logger.error(
            "run_federated_simulation failed — full traceback:", exc_info=True
        )
        raise
