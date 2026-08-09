"""Federated client implementation for the IIoT Federated IDS project.

Owns the single, authoritative Flower client (SDS Section 14.5). ``FLClient``
wraps one virtual client's local model and data shard; ``client_fn`` is the
factory ``flwr==1.8.0``'s ``start_simulation`` calls to build a client from a
client-ID string. No alternate client class or second factory may exist.

Ownership and reuse rules:
  - Each client gets its own disjoint training shard from ``partition_iid``
    plus the **same shared global test split** — never a per-client test
    partition (SDS Section 14.3, authoritative note).
  - The local model comes exclusively from ``build_cnn_gru``, and tensors
    exclusively from ``prepare_model_ready_data`` (SDS Section 22).
  - ``set_global_seed(config["seed"])`` runs before model instantiation,
    because Flower may run clients in separate processes (SDS Section 12).
  - Clients never persist anything; only the server saves the global model.
  - This module performs **no file I/O at all** — every value arrives via
    constructor arguments or ``config``, so the same client works unchanged
    against synthetic in-memory data and the real processed dataset.

Because SDS Section 14.5 fixes the constructor signature and forbids file I/O
here, the fitted ``LabelEncoder``, ordered feature columns, and class count
arrive in ``config["runtime"]``. ``num_classes`` therefore still originates
from ``len(class_mapping)`` in the caller and is never hardcoded (SDS §6).
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

    The ``flwr.client.NumPyClient`` implementation used by every simulated
    client: holds one disjoint training shard plus the shared global test
    split, builds a local CNN-GRU, and exposes ``get_parameters``, ``fit``,
    and ``evaluate`` to the simulation engine (SDS Section 14.5).

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

        Seeds every random source via ``set_global_seed`` **before** the model
        is instantiated, then builds the local CNN-GRU.

        Args:
            client_id: This virtual client's integer identifier (0-based).
            train_data: This client's disjoint training shard from
                ``partition_iid``, on the training split only.
            test_data: The shared global test split, identical for every
                client — never a per-client partition.
            config: Orchestrator-supplied config. Keys read here: ``seed``,
                ``model`` and ``training`` (forwarded to ``build_cnn_gru``),
                and ``runtime`` (``feature_columns``, ``label_encoder``,
                ``num_classes``).

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

        The server uses these to initialise the global model in round 1.

        Args:
            config: Flower's per-call instruction dict. Present to satisfy the
                ``NumPyClient`` interface; no key is read from it.

        Returns:
            list[numpy.ndarray]: The local model's current weights.
        """
        return self.model.get_weights()

    def fit(
        self,
        parameters: list[numpy.ndarray],
        config: dict,
    ) -> tuple[list[numpy.ndarray], int, dict]:
        """Train the local model on this client's shard for one FL round.

        Loads the server's global weights and trains locally for
        ``federated.local_epochs`` epochs at ``training.batch_size``.

        Args:
            parameters: The global model weights sent by the server.
            config: Flower's per-call instruction dict. Present to satisfy the
                ``NumPyClient`` interface; no key is read from it — epochs and
                batch size come from the project config exclusively.

        Returns:
            tuple[list[numpy.ndarray], int, dict]: updated local weights,
                ``self.num_train_examples`` (the LOCAL TRAINING count, i.e.
                shard minus local validation split, used as FedAvg's
                aggregation weight), and the final local epoch's
                ``{"loss", "accuracy"}``. **Both metric keys are always
                present**, since ``weighted_average_fit`` requires both
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

        Scores the fraction of this client's own shard held out in
        ``__init__`` (``federated.client_val_split``), so distributed
        evaluation reports genuinely different local scores — what FedAvg's
        example-weighted aggregation is designed to combine. The shared global
        test split is scored once per round by the server itself, so it stays
        a true held-out set.

        Args:
            parameters: The global model weights sent by the server.
            config: Flower's per-call instruction dict. Present to satisfy the
                ``NumPyClient`` interface; no key is read from it.

        Returns:
            tuple[float, int, dict]: ``loss`` first, so Flower aggregates it
                natively into ``history.losses_distributed``;
                ``self.num_validation_examples`` as the aggregation weight;
                and ``{"accuracy": ...}``. **Only ``accuracy`` is present**,
                since ``weighted_average_eval`` reads only that key
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

    The factory required by ``flwr.simulation.start_simulation``. The
    orchestrator binds the shards, shared test split, and config once via
    ``functools.partial``, so Flower only ever supplies ``cid`` and nothing is
    re-read from disk on any call.

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
