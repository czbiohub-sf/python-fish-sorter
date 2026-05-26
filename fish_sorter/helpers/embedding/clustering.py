"""Clustering strategy seam for the embedding pipeline.

HDBSCAN is the day-one implementation, but `FindingDory` and the vendored
`LabelTool` consume clusters through `ClusterStrategy` so the algorithm can be
swapped via the config without touching UI code.
"""

from typing import Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class ClusterStrategy(Protocol):
    """Cluster a (N, D) embedding array into (N,) integer labels.

    Convention: `-1` denotes noise / unclustered points.
    """

    def cluster(self, embeddings: np.ndarray) -> np.ndarray: ...


class HDBSCANStrategy:
    """Density clustering via hdbscan."""

    def __init__(self, min_cluster_size: int = 10, min_samples=None):
        import hdbscan  # lazy import — keeps `import clustering` cheap
        self._impl = hdbscan.HDBSCAN(
            min_cluster_size=int(min_cluster_size),
            min_samples=None if min_samples is None else int(min_samples),
        )

    def cluster(self, embeddings: np.ndarray) -> np.ndarray:
        return self._impl.fit_predict(embeddings)


def build_cluster_strategy(cfg: dict) -> ClusterStrategy:
    """Construct the configured strategy.

    Reads `cfg["clustering"]["method"]` (case-insensitive) and forwards
    `cfg["clustering"]["params"]` as constructor kwargs.
    """
    block = cfg.get("clustering")
    if not isinstance(block, dict) or "method" not in block:
        raise ValueError(
            "Config is missing a `clustering` block with `method` and `params`."
        )
    method = str(block["method"]).lower()
    params = block.get("params", {}) or {}
    if method == "hdbscan":
        return HDBSCANStrategy(**params)
    raise ValueError(
        f"Unknown clustering method: {method!r}. "
        f"Add a `ClusterStrategy` implementation in clustering.py and a branch here."
    )
