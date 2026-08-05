"""Federated client implementation for the IIoT Federated IDS project — Phase 8.

This module owns the single, authoritative Flower client for the entire project
(SDS Section 14.5). ``FLClient`` wraps one virtual client's local model and data
shard for Flower's simulation engine; ``client_fn`` is the factory the pinned
``flwr==1.8.0`` ``start_simulation`` API calls to instantiate a client from a
client-ID string. No alternate client class or second factory may be introduced
anywhere in the codebase.

Ownership and reuse rules:
  - Each ``FLClient`` receives its own disjoint training shard produced by
    ``partition_iid`` (SDS Section 14.3) plus the **same shared global test
    split** — identical for every client, never a per-client test partition
    (SDS Section 14.3, authoritative note).
  - The local model is built exclusively via
    ``src.models.cnn_gru.build_cnn_gru`` — the project's only model builder.
  - Tensors are built exclusively via
    ``src.preprocessing.encode_normalize.prepare_model_ready_data`` — the
    reshape/one-hot logic is never reimplemented here (SDS Section 22).
  - ``set_global_seed(config["seed"])`` is called before model instantiation,
    because Flower's simulation engine may run clients in separate processes
    (SDS Section 12, Section 14.5 "Critical requirement").
  - Clients never persist anything to disk. Only the server persists the final
    global model (SDS Section 14.5: "Writes: Nothing").
  - This module performs **no file I/O at all**: every value it needs arrives
    through the constructor arguments or through the ``config`` dict prepared
    by the orchestrator, so the same client works unchanged against synthetic
    in-memory data (Phase 9's smoke test) and against the real processed
    dataset (Phase 10's full run).

Runtime values supplied by the orchestrator through ``config`` (see
``FLClient.__init__``): because SDS Section 14.5 fixes the constructor to
exactly ``client_id``/``train_data``/``test_data``/``config`` and forbids this
module from reading any file, the fitted ``LabelEncoder``, the ordered feature
column list, and the derived class count are passed inside
``config["runtime"]``. ``num_classes`` therefore still originates from
``len(class_mapping)`` in the caller and is never hardcoded here (SDS Section 6).

Per the SDS Section 11 import graph, ``src/federated/`` imports only from
``src/utils/``, ``src/preprocessing/``, ``src/partitioning/``, and
``src/models/``. It never imports ``src/centralized/``, ``src/evaluation/``,
``src/explainability/``, or ``src.federated.server_app`` (which does not exist
at this phase — no forward references).
"""

import logging

import numpy
import pandas
from flwr.client import NumPyClient
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight


from src.models.cnn_gru import build_cnn_gru
from src.partitioning.partition_data import partition_iid
from src.preprocessing.encode_normalize import prepare_model_ready_data
from src.utils.seed import set_global_seed

# ``partition_iid`` is re-exported here because SDS Section 14.5 lists it as a
# declared dependency of ``client_fn``'s data contract: the shards handed to
# ``client_fn`` must have been produced by that single authoritative sharding
# function and by no other. Sharding itself is performed once by the
# orchestrator, never inside this module (shards arrive pre-computed).
__all__ = ["FLClient", "client_fn", "partition_iid"]

# Module-level logger only. Modules under src/ never create log files; the five
# fixed log files are created exclusively by experiments/ scripts via
# src.utils.logger.get_logger (SDS Section 13).
_logger = logging.getLogger(__name__)


class FLClient(NumPyClient):
    """Wrap one virtual client's local model and data shard for Flower.

    Purpose:
        Provide the ``flwr.client.NumPyClient`` implementation used by every
        simulated client: hold one disjoint training shard plus the shared
        global test split, build a local CNN-GRU model, and expose
        ``get_parameters``, ``fit``, and ``evaluate`` to the Flower simulation
        engine (SDS Section 14.5).

    Attributes:
        client_id: This virtual client's integer identifier.
        train_data: This client's disjoint training shard from ``partition_iid``.
        test_data: The shared global test split — identical for every client.
        config: The orchestrator-supplied config dict.
        feature_columns: Ordered feature column names (from
            ``feature_names.pkl``, loaded by the orchestrator).
        label_encoder: The fitted ``sklearn.preprocessing.LabelEncoder``
            (from ``label_encoder.pkl``, loaded by the orchestrator).
        num_classes: ``len(class_mapping)``, supplied by the orchestrator —
            never hardcoded here.
        model: The compiled local CNN-GRU model from ``build_cnn_gru``.
    """

    def __init__(
        self,
        client_id: int,
        train_data: pandas.DataFrame,
        test_data: pandas.DataFrame,
        config: dict,
    ) -> None:
        """Create one simulated federated client.

        Purpose:
            Store this client's shard and the shared global test split, seed
            every random source via ``set_global_seed`` **before** any model is
            instantiated, then build the local CNN-GRU model.

        Args:
            client_id: This virtual client's integer identifier (0-based).
            train_data: This client's disjoint training shard, produced by
                ``partition_iid`` on the training split only.
            test_data: The shared global test split (``test_indices.pkl`` rows),
                identical for every client — never a per-client partition.
            config: The orchestrator-supplied config dict. Keys read here:
                ``seed`` (for ``set_global_seed``), ``model`` and ``training``
                (forwarded verbatim to ``build_cnn_gru``), and the
                ``runtime`` sub-dict carrying ``feature_columns``,
                ``label_encoder``, and ``num_classes``.

        Returns:
            None

        Raises:
            KeyError: If ``config`` is missing ``seed``, ``model``,
                ``training``, or any required ``runtime`` entry.
            ValueError: Propagated from ``build_cnn_gru`` if
                ``num_classes < 2``.
        """
        self.client_id = client_id
        self.train_data = train_data
        self.test_data = test_data
        self.config = config

        runtime_config = config["runtime"]
        self.feature_columns = runtime_config["feature_columns"]
        self.label_encoder = runtime_config["label_encoder"]
        # num_classes always originates from len(class_mapping) in the caller
        # (SDS Section 6) — never a literal in this module.
        self.num_classes = runtime_config["num_classes"]

        # MANDATORY ORDER (SDS Section 12 / 14.5): seed everything BEFORE the
        # model is instantiated, since Flower may run clients in separate
        # processes with independent random state.
        set_global_seed(config["seed"])

        self.model = build_cnn_gru(
            input_shape=(len(self.feature_columns), 1),
            num_classes=self.num_classes,
            model_config=config["model"],
            training_config=config["training"],
        )

        # ---- Local train/validation split of this client's OWN shard -------
        # Previously every client evaluated on the identical shared global test
        # split, which made client-side distributed evaluation degenerate: four
        # clients returned four copies of the same number, and
        # weighted_average_eval averaged a constant. Worse, all four then
        # reported a score on data the server also uses as the final held-out
        # test set, so the "federated" evaluation signal was neither federated
        # nor held out.
        #
        # Each client now holds out a fraction of its own shard as a local
        # validation split. Distributed evaluation becomes four genuinely
        # different local scores (which is what FedAvg's example-weighted
        # aggregation is designed for), and the shared global test split is
        # touched exactly once per round by the server itself.
        validation_fraction = config["federated"].get("client_val_split", 0.0)
        local_train = train_data
        local_validation = train_data
        if 0.0 < validation_fraction < 1.0 and len(train_data) > 1:
            stratify_labels = train_data["label"]
            # Stratification needs >= 2 rows per class; a small shard may not
            # satisfy that, in which case fall back to an unstratified split
            # rather than failing the round.
            if stratify_labels.value_counts().min() < 2:
                stratify_labels = None
            local_train, local_validation = train_test_split(
                train_data,
                test_size=validation_fraction,
                random_state=config["seed"],
                stratify=stratify_labels,
            )

        # ---- Tensor caching -----------------------------------------------
        # prepare_model_ready_data was previously called inside fit() AND
        # inside evaluate(), i.e. 2x per client per round -> 120 conversions of
        # the same rows over a 4-client, 15-round run. The conversion is a
        # to_numpy + reshape over ~390k x 95 values, so this was pure repeated
        # work. Built once here instead.
        self.X_train, self.y_train = prepare_model_ready_data(
            local_train,
            local_train.index.values,
            self.feature_columns,
            self.label_encoder,
            self.num_classes,
        )
        self.X_validation, self.y_validation = prepare_model_ready_data(
            local_validation,
            local_validation.index.values,
            self.feature_columns,
            self.label_encoder,
            self.num_classes,
        )
        self.num_train_examples = len(self.X_train)
        self.num_validation_examples = len(self.X_validation)

        # ---- Local class weights ------------------------------------------
        # Same rationale as the centralized path: without rebalancing, a client
        # whose shard is ~85% Normal converges to predicting Normal. Weights are
        # derived from this client's own local training rows only — a client
        # cannot see any other client's label distribution, which is exactly
        # the federated privacy constraint.
        self.class_weight = None
        if config["training"].get("use_class_weights", False):
            y_train_int = numpy.argmax(self.y_train, axis=1)
            present_classes = numpy.unique(y_train_int)
            if len(present_classes) > 1:
                weights = compute_class_weight(
                    class_weight="balanced",
                    classes=present_classes,
                    y=y_train_int,
                )
                self.class_weight = {
                    int(cls): float(weight)
                    for cls, weight in zip(present_classes, weights)
                }

        _logger.debug(
            "Client %s initialised — local_train=%d local_val=%d "
            "class_weighted=%s",
            self.client_id,
            self.num_train_examples,
            self.num_validation_examples,
            self.class_weight is not None,
        )


    def get_parameters(self, config: dict) -> list[numpy.ndarray]:
        """Return the current local model weights.

        Purpose:
            Hand the local model's weights to the Flower server, which uses
            them to initialise the global model in round 1.

        Args:
            config: Flower's per-call instruction dict. Present to satisfy the
                ``NumPyClient`` interface; no key is read from it.

        Returns:
            list[numpy.ndarray]: The local model's current weights.

        Raises:
            Nothing under normal operation.
        """
        return self.model.get_weights()

    def fit(
        self,
        parameters: list[numpy.ndarray],
        config: dict,
    ) -> tuple[list[numpy.ndarray], int, dict]:
        """Train the local model on this client's shard for one FL round.

        Purpose:
            Load the server's global weights, train locally for
            ``federated.local_epochs`` epochs at ``training.batch_size``, and
            return the updated weights, this client's example count, and the
            final local epoch's metrics.

        Args:
            parameters: The global model weights sent by the server.
            config: Flower's per-call instruction dict. Present to satisfy the
                ``NumPyClient`` interface; no key is read from it — epochs and
                batch size come from the project config exclusively.

        Returns:
            tuple[list[numpy.ndarray], int, dict]:
                - Updated local weights after local training.
                - ``self.num_train_examples`` — this client's LOCAL TRAINING
                  example count (shard minus local validation split), used by
                  the server as the FedAvg aggregation weight.

                - Metrics dict ``{"loss": ..., "accuracy": ...}`` taken from
                  the final local epoch of ``history.history``. **Both keys
                  are always present**, because this dict is consumed
                  exclusively by ``weighted_average_fit``, which requires both
                  (SDS Section 14.5 / 14.6).

        Raises:
            Propagates any Keras training exception after logging it at ERROR
            level.
        """
        try:
            self.model.set_weights(parameters)

            # Tensors were built once in __init__ via the sole
            # tensor-preparation entry point (SDS Section 14.2 / 22); rebuilding
            # them every round would repeat the same conversion 15 times.
            history = self.model.fit(
                self.X_train,
                self.y_train,
                epochs=self.config["federated"]["local_epochs"],
                batch_size=self.config["training"]["batch_size"],
                class_weight=self.class_weight,
                verbose=0,
            )


            # Final local epoch's values — both keys are mandatory.
            metrics = {
                "loss": float(history.history["loss"][-1]),
                "accuracy": float(history.history["accuracy"][-1]),
            }

            _logger.debug(
                "Client %s fit complete — examples=%d loss=%.6f accuracy=%.6f",
                self.client_id,
                self.num_train_examples,
                metrics["loss"],
                metrics["accuracy"],
            )

            # The example count is the client's LOCAL TRAINING count, which is
            # FedAvg's aggregation weight. It must exclude the local validation
            # rows, since those did not contribute a single gradient.
            return (
                self.model.get_weights(),
                self.num_train_examples,
                metrics,
            )


        except Exception:
            _logger.error(
                "Client %s failed during local training.",
                self.client_id,
                exc_info=True,
            )
            raise

    def evaluate(
        self,
        parameters: list[numpy.ndarray],
        config: dict,
    ) -> tuple[float, int, dict]:
        """Evaluate the global weights on this client's local validation split.

        Purpose:
            Load the server's global weights and evaluate them against the
            fraction of this client's own shard held out in ``__init__``
            (``federated.client_val_split``). Distributed evaluation therefore
            reports four genuinely different local scores, which is what
            FedAvg's example-weighted aggregation is designed to combine.

            The shared global test split is evaluated once per round by the
            server itself (``make_server_side_evaluate_fn`` in
            ``server_app.py``), so it remains a true held-out set instead of
            being scored by every client every round.


        Args:
            parameters: The global model weights sent by the server.
            config: Flower's per-call instruction dict. Present to satisfy the
                ``NumPyClient`` interface; no key is read from it.

        Returns:
            tuple[float, int, dict]:
                - ``loss`` from ``model.evaluate`` — returned as the first
                  tuple element so Flower aggregates it natively into
                  ``history.losses_distributed``.
                - ``self.num_validation_examples`` — this client's local
                  validation row count, used as the aggregation weight.

                - Metrics dict ``{"accuracy": ...}``. **Only ``accuracy`` is
                  present**, because this dict is consumed exclusively by
                  ``weighted_average_eval``, which reads only that key
                  (SDS Section 14.5 / 14.6).

        Raises:
            Propagates any Keras evaluation exception after logging it at ERROR
            level.
        """
        try:
            self.model.set_weights(parameters)

            # This client's OWN local validation split, cached in __init__.
            # The shared global test split is deliberately not touched here:
            # it is evaluated once per round by the server itself, so it stays
            # a genuine held-out set rather than a per-round training signal.
            loss, accuracy = self.model.evaluate(
                self.X_validation,
                self.y_validation,
                batch_size=self.config["training"]["batch_size"],
                verbose=0,
            )

            _logger.debug(
                "Client %s evaluate complete — local_val_examples=%d "
                "loss=%.6f accuracy=%.6f",
                self.client_id,
                self.num_validation_examples,
                float(loss),
                float(accuracy),
            )

            return (
                float(loss),
                self.num_validation_examples,
                {"accuracy": float(accuracy)},
            )


        except Exception:
            _logger.error(
                "Client %s failed during local evaluation.",
                self.client_id,
                exc_info=True,
            )
            raise


def client_fn(
    cid: str,
    *,
    client_shards: list[pandas.DataFrame],
    test_data: pandas.DataFrame,
    config: dict,
):
    """Instantiate the client Flower's simulation engine asks for.

    Purpose:
        The factory required by ``flwr.simulation.start_simulation``. The
        orchestrator binds the pre-computed shards, the single shared test
        split, and the config once — via ``functools.partial`` — and passes the
        resulting single-argument callable to ``start_simulation``, so Flower
        only ever supplies ``cid``. Nothing is re-read from disk on any call.

    Args:
        cid: The client ID, which Flower always passes as a string.
        client_shards: The pre-computed disjoint training shards produced once
            by ``partition_iid`` on the training split. Bound by the
            orchestrator (keyword-only, closure-style) — never computed here.
        test_data: The single shared global test split, identical for every
            client. Bound by the orchestrator.
        config: The project config dict, bound by the orchestrator.

    Returns:
        flwr.client.Client: An ``FLClient`` instance converted via
            ``.to_client()``, as required by the ``flwr==1.8.0`` API.

    Raises:
        IndexError: If ``int(cid)`` exceeds the number of pre-loaded partitions.
        ValueError: If ``cid`` is not an integer string.
    """
    client_id = int(cid)

    if client_id >= len(client_shards) or client_id < 0:
        raise IndexError(
            f"cid {cid!r} is out of range: only {len(client_shards)} client "
            f"partitions were pre-loaded."
        )

    client = FLClient(
        client_id=client_id,
        train_data=client_shards[client_id],
        test_data=test_data,
        config=config,
    )

    return client.to_client()
