"""Unit tests for src/partitioning/partition_data.py — Phase 7.

Covers every case mandated by SDS Section 19's test-case table for this file
and the Phase 7 validation checklist:
  1. partition_iid produces exactly num_clients shards.
  2. Shard sizes sum to the original dataset size.
  3. No row appears in more than one shard.
  4. ValueError is raised on num_clients < 1.

All fixtures are synthetic and in-memory — this phase reads no artifact from
disk and persists nothing (Contract Phase 7: "Files allowed to READ: None at
build time").
"""

import numpy
import pandas
import pytest

from src.partitioning.partition_data import partition_iid


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SEED = 42


@pytest.fixture
def dummy_df() -> pandas.DataFrame:
    """A small synthetic DataFrame with a unique, traceable row marker."""
    num_rows = 100
    return pandas.DataFrame(
        {
            "row_marker": numpy.arange(num_rows),
            "feature_a": numpy.linspace(0.0, 1.0, num_rows),
            "feature_b": numpy.arange(num_rows) % 7,
            "label": ["Normal" if i % 3 == 0 else "DDoS" for i in range(num_rows)],
        }
    )


@pytest.fixture
def uneven_df() -> pandas.DataFrame:
    """A synthetic DataFrame whose row count is not divisible by num_clients."""
    num_rows = 103
    return pandas.DataFrame(
        {
            "row_marker": numpy.arange(num_rows),
            "feature_a": numpy.arange(num_rows, dtype=float),
        }
    )


# ---------------------------------------------------------------------------
# Test case 1 — exact shard count
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("num_clients", [1, 2, 4, 10])
def test_partition_iid_produces_exact_shard_count(dummy_df, num_clients):
    """partition_iid returns exactly num_clients shards."""
    shards = partition_iid(dummy_df, num_clients, SEED)

    assert isinstance(shards, list)
    assert len(shards) == num_clients
    for shard in shards:
        assert isinstance(shard, pandas.DataFrame)


# ---------------------------------------------------------------------------
# Test case 2 — shard sizes sum to the original dataset size
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("num_clients", [1, 2, 4, 10])
def test_partition_iid_shard_sizes_sum_to_dataset_size(dummy_df, num_clients):
    """The union of all shards covers every row of the source DataFrame."""
    shards = partition_iid(dummy_df, num_clients, SEED)

    assert sum(len(shard) for shard in shards) == len(dummy_df)


def test_partition_iid_covers_all_rows_when_not_evenly_divisible(uneven_df):
    """Full coverage holds even when len(df) is not divisible by num_clients."""
    num_clients = 4
    shards = partition_iid(uneven_df, num_clients, SEED)

    assert sum(len(shard) for shard in shards) == len(uneven_df)

    recovered = sorted(
        marker
        for shard in shards
        for marker in shard["row_marker"].tolist()
    )
    assert recovered == list(range(len(uneven_df)))


def test_partition_iid_shards_are_equal_size_within_one_row(dummy_df):
    """IID partitioning produces equal-size shards (±1 row when uneven)."""
    num_clients = 4
    shards = partition_iid(dummy_df, num_clients, SEED)

    sizes = [len(shard) for shard in shards]
    assert max(sizes) - min(sizes) <= 1


# ---------------------------------------------------------------------------
# Test case 3 — no row appears in more than one shard
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("num_clients", [2, 4, 10])
def test_partition_iid_shards_are_disjoint(dummy_df, num_clients):
    """No source row is assigned to more than one shard."""
    shards = partition_iid(dummy_df, num_clients, SEED)

    all_markers = [
        marker
        for shard in shards
        for marker in shard["row_marker"].tolist()
    ]

    # No duplicates across shards, and every original row present exactly once.
    assert len(all_markers) == len(set(all_markers))
    assert sorted(all_markers) == sorted(dummy_df["row_marker"].tolist())


def test_partition_iid_resets_index_on_each_shard(dummy_df):
    """Each returned shard has a fresh 0..n-1 index (reset_index(drop=True))."""
    shards = partition_iid(dummy_df, 4, SEED)

    for shard in shards:
        assert list(shard.index) == list(range(len(shard)))


def test_partition_iid_preserves_source_columns(dummy_df):
    """Partitioning never adds, drops, or reorders columns."""
    shards = partition_iid(dummy_df, 4, SEED)

    for shard in shards:
        assert list(shard.columns) == list(dummy_df.columns)


def test_partition_iid_does_not_mutate_source_dataframe(dummy_df):
    """The source DataFrame is left untouched by partitioning."""
    before = dummy_df.copy(deep=True)

    partition_iid(dummy_df, 4, SEED)

    pandas.testing.assert_frame_equal(dummy_df, before)


# ---------------------------------------------------------------------------
# Determinism (seed policy — Contract A.10)
# ---------------------------------------------------------------------------

def test_partition_iid_is_deterministic_for_a_fixed_seed(dummy_df):
    """The same seed always produces byte-identical shard contents."""
    shards_first = partition_iid(dummy_df, 4, SEED)
    shards_second = partition_iid(dummy_df, 4, SEED)

    for first, second in zip(shards_first, shards_second):
        pandas.testing.assert_frame_equal(first, second)


def test_partition_iid_shuffles_rows_before_splitting(dummy_df):
    """Shards are randomly drawn, not contiguous slices of the source order."""
    shards = partition_iid(dummy_df, 4, SEED)

    first_shard_markers = shards[0]["row_marker"].tolist()
    contiguous_head = list(range(len(first_shard_markers)))

    assert first_shard_markers != contiguous_head


# ---------------------------------------------------------------------------
# Test case 4 — ValueError on invalid num_clients
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("num_clients", [0, -1, -5])
def test_partition_iid_raises_value_error_on_num_clients_below_one(
    dummy_df, num_clients
):
    """num_clients < 1 is rejected with ValueError."""
    with pytest.raises(ValueError):
        partition_iid(dummy_df, num_clients, SEED)


def test_partition_iid_raises_value_error_when_num_clients_exceeds_rows(dummy_df):
    """num_clients > len(df) is rejected with ValueError."""
    with pytest.raises(ValueError):
        partition_iid(dummy_df, len(dummy_df) + 1, SEED)
