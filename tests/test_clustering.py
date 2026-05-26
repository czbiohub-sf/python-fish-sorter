"""Tests for the cluster strategy seam."""

import numpy as np
import pytest

from fish_sorter.helpers.embedding.clustering import (
    ClusterStrategy,
    HDBSCANStrategy,
    build_cluster_strategy,
)


def _two_blob_embeddings(seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    a = rng.normal(loc=0.0, scale=0.05, size=(40, 8))
    b = rng.normal(loc=5.0, scale=0.05, size=(40, 8))
    return np.concatenate([a, b], axis=0).astype(np.float32)


def test_hdbscan_strategy_separates_two_blobs():
    embeddings = _two_blob_embeddings()
    strategy = HDBSCANStrategy(min_cluster_size=5)
    labels = strategy.cluster(embeddings)
    assert labels.shape == (80,)
    unique = set(labels.tolist())
    non_noise = {u for u in unique if u != -1}
    assert len(non_noise) >= 2, f"expected at least 2 clusters, got {unique}"


def test_build_cluster_strategy_returns_hdbscan_for_hdbscan_method():
    cfg = {"clustering": {"method": "hdbscan", "params": {"min_cluster_size": 5}}}
    strategy = build_cluster_strategy(cfg)
    assert isinstance(strategy, HDBSCANStrategy)
    assert isinstance(strategy, ClusterStrategy)


def test_build_cluster_strategy_is_case_insensitive():
    cfg = {"clustering": {"method": "HDBSCAN", "params": {"min_cluster_size": 5}}}
    strategy = build_cluster_strategy(cfg)
    assert isinstance(strategy, HDBSCANStrategy)


def test_build_cluster_strategy_raises_on_unknown_method():
    cfg = {"clustering": {"method": "magic", "params": {}}}
    with pytest.raises(ValueError, match="Unknown clustering method"):
        build_cluster_strategy(cfg)


def test_build_cluster_strategy_raises_on_missing_block():
    with pytest.raises(ValueError, match="missing a `clustering` block"):
        build_cluster_strategy({})
