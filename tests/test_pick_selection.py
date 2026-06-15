"""Tests for Track F — Finding Dory classes in Pick Selection.

Covers the pure helpers (`discover_pick_features_and_combos`, `map_combos_to_wells`,
`latest_classifications_csv`), the graceful `match_pick` guard, and the end-to-end
contract: a Finding Dory wide CSV → discovered features/combos → pickable → the real
`match_pick` join selects exactly each combo's wells and exposes `slotName`.
"""

import os

import pandas as pd
import pytest

from fish_sorter.GUI.finding_dory import write_wide_csv
from fish_sorter.GUI.picking import Pick, latest_classifications_csv
from fish_sorter.GUI.selection_gui import (
    discover_pick_features_and_combos,
    group_features_for_display,
    map_combos_to_wells,
)
from fish_sorter.helpers.labelling.store import LabelStore

WELL_CLASS = ["empty", "deformed", "wrong_o", "unknown", "multiple", "singlet"]


def _make_store(well_ids, experiment="exp_2dpf_myo6b"):
    """Build a LabelStore with metadata for the given well_ids (mirrors test_wide_csv)."""
    rows = []
    for wid in well_ids:
        well_name = wid.split("_", 1)[1] if "_" in wid else wid
        rows.append({"well_id": wid, "experiment": experiment, "well_name": well_name})
    return LabelStore(pd.DataFrame(rows))


# ---------------------------------------------------------------------------
# discover_pick_features_and_combos
# ---------------------------------------------------------------------------


def test_discover_per_channel_features_and_combos():
    class_df = pd.DataFrame(
        {
            "slotName": ["A01", "A02", "A03", "A04", "A05"],
            "empty": [0, 0, 0, 1, 0],
            "singlet": [1, 1, 1, 0, 1],
            "multiple": [0, 0, 0, 0, 0],
            "deformed": [0, 0, 0, 0, 0],
            "lHead": [0, 0, 0, 0, 0],
            "GFP_a": [1, 1, 0, 0, 0],
            "GFP_b": [0, 0, 1, 0, 0],
            "TXR_x": [1, 1, 1, 0, 0],
        }
    )
    features, combos = discover_pick_features_and_combos(class_df, WELL_CLASS)

    # well_class (config) first, then the dynamic per-channel columns — no gEye-style cols.
    assert features == WELL_CLASS + ["GFP_a", "GFP_b", "TXR_x"]

    # Sorted by descending well-count; most-assigned combo first.
    assert combos[0]["count"] == 2
    assert combos[0]["checks"] == {"singlet": 1, "GFP_a": 1, "TXR_x": 1}
    # singlet is always checked; the all-zero catch-all (A05) is included.
    assert all(c["checks"].get("singlet") == 1 for c in combos)
    assert any(c["checks"] == {"singlet": 1} for c in combos)
    # 3 distinct combos over the 4 singlet wells; the empty well (A04) never appears.
    assert len(combos) == 3
    assert sum(c["count"] for c in combos) == 4


def test_discover_classical_features():
    class_df = pd.DataFrame(
        {
            "slotName": ["A01", "A02"],
            "empty": [0, 0],
            "deformed": [0, 0],
            "wrong_o": [0, 0],
            "unknown": [0, 0],
            "multiple": [0, 0],
            "singlet": [1, 1],
            "gEye": [1, 0],
            "gHeart": [0, 1],
            "lHead": [0, 0],
        }
    )
    features, combos = discover_pick_features_and_combos(class_df, WELL_CLASS)
    assert features == WELL_CLASS + ["gEye", "gHeart"]
    assert len(combos) == 2  # two distinct gEye/gHeart signatures


def test_discover_no_dynamic_columns_single_catchall():
    class_df = pd.DataFrame(
        {
            "slotName": ["A01", "A02"],
            "empty": [0, 0],
            "singlet": [1, 1],
            "multiple": [0, 0],
            "deformed": [0, 0],
            "lHead": [0, 0],
        }
    )
    features, combos = discover_pick_features_and_combos(class_df, WELL_CLASS)
    assert features == WELL_CLASS  # no feature columns discovered
    assert combos == [{"checks": {"singlet": 1}, "count": 2}]


def test_discover_no_singlets_yields_no_combos():
    class_df = pd.DataFrame(
        {
            "slotName": ["A01"],
            "empty": [1],
            "singlet": [0],
            "multiple": [0],
            "deformed": [0],
            "lHead": [0],
            "GFP_a": [0],
        }
    )
    features, combos = discover_pick_features_and_combos(class_df, WELL_CLASS)
    assert "GFP_a" in features
    assert combos == []


# ---------------------------------------------------------------------------
# map_combos_to_wells (cap + order + dropped)
# ---------------------------------------------------------------------------


def test_map_combos_caps_and_keeps_most_assigned():
    combos = [
        {"checks": {"singlet": 1}, "count": 5},
        {"checks": {"singlet": 1}, "count": 3},
        {"checks": {"singlet": 1}, "count": 1},
    ]
    mapped, dropped = map_combos_to_wells(combos, ["w0", "w1"])
    assert dropped == 1
    assert [w for w, _ in mapped] == ["w0", "w1"]
    assert mapped[0][1]["count"] == 5  # most-assigned mapped first


def test_map_combos_no_overflow():
    combos = [{"checks": {"singlet": 1}, "count": 2}]
    mapped, dropped = map_combos_to_wells(combos, ["w0", "w1", "w2"])
    assert dropped == 0
    assert len(mapped) == 1


# ---------------------------------------------------------------------------
# group_features_for_display (one line per channel)
# ---------------------------------------------------------------------------


def test_group_features_one_line_per_channel():
    features = WELL_CLASS + ["GFP_a", "GFP_b", "TXR_x"]
    wc, channel_groups, ungrouped = group_features_for_display(features, WELL_CLASS)
    assert wc == WELL_CLASS
    assert channel_groups == [["GFP", ["GFP_a", "GFP_b"]], ["TXR", ["TXR_x"]]]
    assert ungrouped == []


def test_group_features_classical_features_ungrouped():
    features = WELL_CLASS + ["gEye", "gHeart"]
    wc, channel_groups, ungrouped = group_features_for_display(features, WELL_CLASS)
    assert channel_groups == []
    assert ungrouped == ["gEye", "gHeart"]


def test_group_features_well_class_with_underscore_not_a_channel():
    # wrong_o has an underscore but is well-class — must not become channel 'wrong'.
    features = ["wrong_o", "singlet", "GFP_x"]
    wc, channel_groups, ungrouped = group_features_for_display(features, WELL_CLASS)
    assert "wrong_o" in wc
    assert channel_groups == [["GFP", ["GFP_x"]]]
    assert ungrouped == []


def test_group_features_covers_every_feature():
    features = WELL_CLASS + ["GFP_a", "TXR_x", "gEye"]
    wc, channel_groups, ungrouped = group_features_for_display(features, WELL_CLASS)
    flat = list(wc) + [f for _, feats in channel_groups for f in feats] + list(ungrouped)
    assert sorted(flat) == sorted(features)  # no feature dropped or duplicated


# ---------------------------------------------------------------------------
# latest_classifications_csv
# ---------------------------------------------------------------------------


def test_latest_classifications_csv_picks_newest_by_mtime(tmp_path):
    older = tmp_path / "20260201_000000_p_classifications.csv"
    newer = tmp_path / "20260101_000000_p_classifications.csv"  # earlier name, newer mtime
    older.write_text("slotName\nA01\n")
    newer.write_text("slotName\nA01\n")
    (tmp_path / "20260301_000000_p_pickable.csv").write_text("dispenseWell\nw0\n")  # ignored
    os.utime(older, (1_000_100, 1_000_100))
    os.utime(newer, (1_000_200, 1_000_200))

    result = latest_classifications_csv(str(tmp_path))
    assert os.path.basename(result) == newer.name


def test_latest_classifications_csv_none_when_absent(tmp_path):
    assert latest_classifications_csv(str(tmp_path)) is None
    assert latest_classifications_csv(str(tmp_path / "does_not_exist")) is None


# ---------------------------------------------------------------------------
# match_pick guard (graceful refusal — no exception)
# ---------------------------------------------------------------------------


def _pick_with(class_df, pickable_df):
    p = Pick()
    p.configured = True  # bypass requires_setup; match_pick only needs the two frames
    p.class_file = class_df
    p.pick_param_file = pickable_df
    return p


def test_match_pick_guard_blocks_on_missing_active_column():
    # Stale pickable references GFP_x=1, but the loaded classification has no GFP_x.
    class_df = pd.DataFrame(
        {"slotName": ["A01"], "empty": [0], "singlet": [1], "multiple": [0],
         "deformed": [0], "lHead": [0]}
    )
    pickable_df = pd.DataFrame(
        {"dispenseWell": ["w0"], "empty": [0], "singlet": [1], "multiple": [0],
         "deformed": [0], "GFP_x": [1]}
    )
    p = _pick_with(class_df, pickable_df)
    p.match_pick()

    assert p.match_warning is not None
    assert "GFP_x" in p.match_warning
    assert len(p.matches) == 0  # pick list NOT built
    assert list(p.matches.columns) == ["slotName", "dispenseWell", "lHead"]


def test_match_pick_ignores_benign_all_zero_absent_column():
    # wrong_o is in the pickable (config well-class) but all-zero and absent from the
    # Dory classification — must NOT trip the guard.
    class_df = pd.DataFrame(
        {"slotName": ["A01"], "empty": [0], "singlet": [1], "multiple": [0],
         "deformed": [0], "lHead": [0]}
    )
    pickable_df = pd.DataFrame(
        {"dispenseWell": ["w0"], "empty": [0], "singlet": [1], "multiple": [0],
         "deformed": [0], "wrong_o": [0]}
    )
    p = _pick_with(class_df, pickable_df)
    p.match_pick()

    assert p.match_warning is None
    assert len(p.matches) == 1
    assert p.matches.iloc[0]["slotName"] == "A01"


# ---------------------------------------------------------------------------
# End-to-end: wide CSV -> discover -> pickable -> real match_pick join
# ---------------------------------------------------------------------------


def _expected_slots(class_df, feature_cols, checks):
    """Replicate match_pick's exact-match join on the columns shared with class_df."""
    shared = [c for c in feature_cols if c in class_df.columns]
    mask = pd.Series(True, index=class_df.index)
    for c in shared:
        mask &= class_df[c] == int(checks.get(c, 0))
    return set(class_df.loc[mask, "slotName"])


def test_end_to_end_wide_csv_is_pickable(tmp_path):
    wells = ["expA_A01", "expA_A02", "expA_A03", "expA_A04"]
    store = _make_store(wells)
    fish_line = "myo6b"
    store._line_channels[fish_line] = ["GFP", "TXR"]

    # Two wells share a combo; one has a different GFP group; one is an unlabeled singlet.
    store.assign(fish_line, "GFP", ["expA_A01", "expA_A02"], "clusterA")
    store.assign(fish_line, "TXR", ["expA_A01", "expA_A02"], "cluster1")
    store.assign(fish_line, "GFP", ["expA_A03"], "clusterB")
    store.assign(fish_line, "TXR", ["expA_A03"], "cluster1")
    # A04 left unassigned → singlet catch-all.

    csv_path = tmp_path / "20260615_000000_expA_classifications.csv"
    write_wide_csv(
        store=store,
        well_order=wells,
        lhead_map={},
        channels=["GFP", "TXR"],
        fish_line=fish_line,
        path=str(csv_path),
    )
    class_df = pd.read_csv(csv_path)
    assert "slotName" in class_df.columns  # the rename — no KeyError downstream

    feature_cols, combos = discover_pick_features_and_combos(class_df, WELL_CLASS)
    assert combos[0]["count"] == 2  # the shared combo is most-assigned

    # Build pickable.csv exactly as save_select would: ['dispenseWell'] + feature_cols.
    disp_wells = [f"w{i}" for i in range(len(combos))]
    rows = []
    for well, combo in zip(disp_wells, combos):
        row = {"dispenseWell": well}
        for col in feature_cols:
            row[col] = int(combo["checks"].get(col, 0))
        rows.append(row)
    pickable_df = pd.DataFrame(rows)[["dispenseWell"] + feature_cols]

    p = _pick_with(class_df, pickable_df)
    p.match_pick()

    assert p.match_warning is None
    assert list(p.matches.columns) == ["slotName", "dispenseWell", "lHead"]

    # Every combo's dispense well matches exactly the wells in that combo.
    for well, combo in zip(disp_wells, combos):
        got = set(p.matches.loc[p.matches["dispenseWell"] == well, "slotName"])
        assert got == _expected_slots(class_df, feature_cols, combo["checks"])

    # Sanity: all four singlet wells are accounted for across the pick list.
    assert set(p.matches["slotName"]) == {"A01", "A02", "A03", "A04"}
