"""IID client partitioning for the federated simulation — Phase 7.

This module owns the single, authoritative sharding function for the entire
project (SDS Section 14.3). ``partition_iid`` is applied exclusively to the
training-split portion of the processed data (the rows identified by
``train_indices.pkl``); the shared global test split is never partitioned per
client (SDS Section 14.3, Contract A.2.7).

No shard is ever persisted to disk — shards exist in memory only and are
handed to the federated client factory by the caller.
"""

import numpy
import pandas


def partition_iid(
    df: pandas.DataFrame,
    num_clients: int,
    seed: int,
) -> list[pandas.DataFrame]:
    """Split a DataFrame into equal-size, non-overlapping IID client shards.

    Purpose:
        Produce ``num_clients`` disjoint, randomly shuffled, equal-size shards
        of the supplied DataFrame (IID partitioning). This function is applied
        only to the training-split portion of the processed data; it must never
        be applied to the test split.

    Args:
        df: The DataFrame to partition. In production this is always
            ``df.loc[train_indices]`` — the training rows of the processed
            dataset.
        num_clients: Number of virtual clients (shards) to produce. Sourced
            from ``config["federated"]["num_clients"]`` by the caller.
        seed: Random seed used to shuffle rows before splitting. Sourced from
            ``config["seed"]`` by the caller, guaranteeing reproducible shards.

    Returns:
        list[pandas.DataFrame]: A list of length ``num_clients``. Each element
            is a disjoint subset of ``df`` with its index reset via
            ``reset_index(drop=True)``. The shard sizes sum exactly to
            ``len(df)`` and no original row appears in more than one shard.

    Raises:
        ValueError: If ``num_clients < 1`` or ``num_clients > len(df)``.
    """
    if num_clients < 1:
        raise ValueError(
            f"num_clients must be >= 1, received {num_clients}."
        )
    if num_clients > len(df):
        raise ValueError(
            f"num_clients ({num_clients}) cannot exceed the number of rows "
            f"in df ({len(df)})."
        )

    # Seeded shuffle of positional row indices — reproducible across runs.
    rng = numpy.random.RandomState(seed)
    shuffled_positions = rng.permutation(len(df))

    # Equal-size, non-overlapping, exhaustive split of the shuffled positions.
    # numpy.array_split covers every position exactly once even when len(df)
    # is not divisible by num_clients (sizes differ by at most one row).
    position_shards = numpy.array_split(shuffled_positions, num_clients)

    client_shards: list[pandas.DataFrame] = [
        df.iloc[positions].reset_index(drop=True) for positions in position_shards
    ]

    return client_shards
