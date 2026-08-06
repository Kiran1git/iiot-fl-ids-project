"""Federated server, strategy, and simulation wiring.

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

``run_federated_simulation`` is the project's single top-level federated
orchestrator. It loads the processed data and preprocessing artifacts, shards
the training split via ``partition_iid``, runs the production-scale simulation
through the pinned ``start_simulation`` API, and rebuilds the global model from
``SavingFedAvg.latest_parameters`` — never from a per-client local model.

Per the SDS Section 11 import graph, ``src/federated/`` imports only from
``src/utils/``, ``src/preprocessing/``, ``src/partitioning/``, and
``src/models/``. It never imports ``src/centralized/``, ``src/evaluation/``,
or ``src/explainability/``.
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


def build_ray_init_args(config: dict) -> dict:
    """Assemble the ``ray_init_args`` dict for ``start_simulation``.

    Purpose:
        Ray on Windows backs its object store with a memory-mapped file sized,
        by default, at roughly 30% of physical RAM. On a 16 GB laptop that is a
        ~4.8 GB mapping requested up front, and once TensorFlow's four client
        actors have taken their share, Windows can no longer commit it:

            CreateFileMapping() failed. GetLastError() = 1450

        which is ``ERROR_NO_SYSTEM_RESOURCES`` and kills the raylet mid-run.
        Capping ``object_store_memory`` keeps that mapping small enough to
        always succeed.

        When ``federated.ray`` is absent from the config — which is the case
        for the untouched production ``configs/config.yaml`` — this function
        returns exactly the settings the project used before, so the
        FULL_EXPERIMENT path is bit-for-bit unchanged.

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

    Purpose:
        Flower derives the number of *concurrent* client actors from this dict:

            concurrent_actors = floor(ray_num_cpus / client_num_cpus)

        Every concurrent actor holds its own TensorFlow runtime (~350-500 MB),
        its own copy of the CNN-GRU graph, and its own cached feature tensors.
        Raising ``client_num_cpus`` therefore *lowers* peak RAM by serialising
        the clients. In LAPTOP_MODE the overlay sets ``num_cpus: 2`` and
        ``client_num_cpus: 2``, so exactly one client trains at a time.

        This changes nothing mathematically: clients within a FedAvg round are
        independent, so running them sequentially yields the identical
        aggregate as running them in parallel.

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

    Purpose:
        The server-side evaluation tensors stay resident for the whole
        simulation. At ~443k test rows x 95 features x 4 bytes that is roughly
        168 MB of float32 held alongside every client actor. In LAPTOP_MODE
        ``runtime_profile.server_eval_max_rows`` caps that at 50k rows (~19 MB).

        The subsample is stratified on ``label`` and drawn with the project
        seed, so the accuracy estimate stays representative and reproducible;
        at 50k rows the sampling error on an accuracy figure is well under
        +/-0.5%.

        When the key is absent — the FULL_EXPERIMENT case — the full split is
        returned unchanged.

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

    Purpose:
        Client-side distributed evaluation answers "how does the global model
        do on each client's copy of the test split?"; it is aggregated by
        example count and is the only signal the project previously recorded.
        Flower's ``evaluate_fn`` hook answers a different and more
        authoritative question: how does the *aggregated* model score on the
        shared global test split, evaluated exactly once, on the server, with
        no per-client weighting artefacts.

        Having both matters here because a divergence between the two is the
        standard symptom of aggregation going wrong (e.g. weights averaged in
        the wrong order, or a client returning stale parameters). With only
        distributed evaluation, that failure is invisible.

        The returned closure keeps ``X_test``/``y_test`` bound once, so the
        test tensors are built a single time for the whole simulation rather
        than per round.

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

    Purpose:
        The project's single top-level federated orchestrator (SDS Section
        14.6): load the processed dataset and preprocessing artifacts, shard
        the training split via ``partition_iid``, launch
        ``flwr.simulation.start_simulation`` for
        ``config["federated"]["num_rounds"]`` rounds using the pinned
        ``flwr==1.8.0`` client-function/strategy API, time only that call,
        rebuild the global model from ``SavingFedAvg.latest_parameters``, and
        persist every federated artifact named in SDS Section 9.

        The saved ``federated_global_model.h5`` is *always* built by converting
        ``strategy.latest_parameters`` via ``parameters_to_ndarrays`` and
        loading those weights into a freshly constructed ``build_cnn_gru``
        model. A per-client local model is never saved under that filename
        (SDS Section 14.6, Section 22).

        Only the shared global test split is used for client-side evaluation —
        ``partition_iid`` is applied to the training rows exclusively
        (SDS Section 14.3).

    Args:
        config: Fully loaded config dict from
            ``src.utils.config_loader.load_config()``. Production federated
            values are read as-is: ``federated.num_clients`` (4),
            ``federated.num_rounds`` (15), ``federated.local_epochs`` (2).
        logger: Optional pre-constructed ``logging.Logger`` passed down from
            ``experiments/run_federated.py``. Modules under ``src/`` never
            construct their own log file (SDS Section 13), so when this is
            omitted a plain handler-less logger is used.

    Returns:
        None

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
