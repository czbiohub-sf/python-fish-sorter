"""Tests for LabelStore source-of-truth guarantees.

Covers the two correctness fixes:
- Track D: the channel list is deduplicated at the single accessor
  chokepoint (``_channels_for_line``), so cross-channel tuples have exactly
  one entry per distinct channel.
- Track E: ``delete_group`` purges the well-assignment vector (and returns
  the true count) even when the group name has drifted out of ``groups``,
  so deleted wells never linger as orphaned/over-counted assignments.
"""

import pandas as pd

from fish_sorter.helpers.labelling.store import _scope_key, LabelStore


def _make_store(well_ids, experiment="exp_2dpf_myo6b"):
    rows = []
    for wid in well_ids:
        well_name = wid.split("_", 1)[1] if "_" in wid else wid
        rows.append({"well_id": wid, "experiment": experiment, "well_name": well_name})
    return LabelStore(pd.DataFrame(rows))


# ---------------------------------------------------------------------------
# Track D — channel list dedup
# ---------------------------------------------------------------------------


def test_channels_for_line_dedupes_explicit_list():
    store = _make_store(["exp_A1"])
    # Duplicate channels seeded into _line_channels (the drift this fixes).
    store._line_channels["myo6b"] = ["GFP", "GFP", "TXR", "TXR"]
    assert store._channels_for_line("myo6b") == ["GFP", "TXR"]


def test_channels_for_line_is_order_stable():
    store = _make_store(["exp_A1"])
    store._line_channels["myo6b"] = ["TXR", "GFP", "TXR"]
    assert store._channels_for_line("myo6b") == ["TXR", "GFP"]


def test_channels_for_line_falls_back_to_scopes():
    store = _make_store(["exp_A1"])
    # No _line_channels entry — derive from registered scopes, deduped.
    store._get_scope(_scope_key("myo6b", "GFP"))
    store._get_scope(_scope_key("myo6b", "TXR"))
    assert sorted(store._channels_for_line("myo6b")) == ["GFP", "TXR"]


# ---------------------------------------------------------------------------
# Track E — delete_group robustness
# ---------------------------------------------------------------------------


def test_delete_group_returns_true_count():
    store = _make_store(["exp_A1", "exp_A2", "exp_A3"])
    store.assign("myo6b", "GFP", ["exp_A1", "exp_A2", "exp_A3"], "cluster_0")
    assert store.delete_group("myo6b", "GFP", "cluster_0") == 3
    assert store.counts("myo6b", "GFP") == {}


def test_delete_group_purges_orphaned_assignments():
    """Group name dropped from ``groups`` but wells still assigned.

    Before the fix this hit the early ``return 0`` and left the wells
    orphaned (still counted/colored as assigned). Now it purges them and
    reports the real count.
    """
    store = _make_store(["exp_A1", "exp_A2"])
    store.assign("myo6b", "GFP", ["exp_A1", "exp_A2"], "cluster_0")
    scope = store._get_scope(_scope_key("myo6b", "GFP"))
    scope["groups"].remove("cluster_0")  # force the drift

    removed = store.delete_group("myo6b", "GFP", "cluster_0")
    assert removed == 2
    assert store.assignments("myo6b", "GFP") == {}


def test_delete_group_absent_name_no_assignments_returns_zero():
    store = _make_store(["exp_A1"])
    assert store.delete_group("myo6b", "GFP", "never_existed") == 0
