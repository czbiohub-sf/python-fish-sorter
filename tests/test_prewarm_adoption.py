"""Tests for the Finding Dory pre-warm adoption helper.

`_subset_embeddings` is the pure piece of the dock's prewarm-adoption path:
the background pre-warm embeds every well, and when the dock adopts that
result with `filter_to_singlets` enabled it must restrict the embeddings to
the singlet keep-set. (The Qt wiring around it isn't unit-tested here.)
"""

import numpy as np

from fish_sorter.GUI.finding_dory import _subset_embeddings


def test_subset_embeddings_keeps_only_requested_wells():
    # Two channels, 4 wells each (indices 0..3), 3-dim embeddings.
    embeds = {
        "BF": np.arange(12, dtype=np.float32).reshape(4, 3),
        "GFP": np.arange(100, 112, dtype=np.float32).reshape(4, 3),
    }
    idx = {"BF": np.array([0, 1, 2, 3]), "GFP": np.array([0, 1, 2, 3])}

    out_e, out_i = _subset_embeddings(embeds, idx, np.array([1, 3]))

    assert np.array_equal(out_i["BF"], np.array([1, 3]))
    assert np.array_equal(out_i["GFP"], np.array([1, 3]))
    assert np.array_equal(out_e["BF"], embeds["BF"][[1, 3]])
    assert np.array_equal(out_e["GFP"], embeds["GFP"][[1, 3]])


def test_subset_embeddings_respects_per_channel_index_maps():
    # Channels can carry different well-index maps; the keep-set is by well id.
    embeds = {
        "BF": np.array([[0.0], [1.0], [2.0]], dtype=np.float32),   # wells 0,2,4
        "GFP": np.array([[9.0], [8.0]], dtype=np.float32),         # wells 2,5
    }
    idx = {"BF": np.array([0, 2, 4]), "GFP": np.array([2, 5])}

    out_e, out_i = _subset_embeddings(embeds, idx, np.array([2, 4]))

    assert np.array_equal(out_i["BF"], np.array([2, 4]))
    assert np.array_equal(out_e["BF"], np.array([[1.0], [2.0]], dtype=np.float32))
    # GFP only has well 2 in the keep-set.
    assert np.array_equal(out_i["GFP"], np.array([2]))
    assert np.array_equal(out_e["GFP"], np.array([[9.0]], dtype=np.float32))


def test_subset_embeddings_empty_keep_set_yields_empty_arrays():
    embeds = {"BF": np.arange(6, dtype=np.float32).reshape(3, 2)}
    idx = {"BF": np.array([0, 1, 2])}

    out_e, out_i = _subset_embeddings(embeds, idx, np.array([], dtype=np.int64))

    assert out_i["BF"].shape == (0,)
    assert out_e["BF"].shape == (0, 2)
