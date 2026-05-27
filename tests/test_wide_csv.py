"""Tests for the classify-compatible wide-CSV writer."""

import re

import pandas as pd
import pytest

from fish_sorter.GUI.finding_dory import default_csv_path, write_wide_csv
from fish_sorter.helpers.labelling.store import LabelStore


def _make_store(well_ids, experiment="exp_2dpf_myo6b"):
    """Build a LabelStore with metadata for the given well_ids."""
    rows = []
    for wid in well_ids:
        # well_id convention: "<experiment>_<well_name>"
        well_name = wid.split("_", 1)[1] if "_" in wid else wid
        rows.append({"well_id": wid, "experiment": experiment, "well_name": well_name})
    metadata = pd.DataFrame(rows)
    return LabelStore(metadata)


# ---------------------------------------------------------------------------
# default_csv_path
# ---------------------------------------------------------------------------


def test_default_csv_path_uses_timestamp_pattern(tmp_path):
    path = default_csv_path(str(tmp_path), "myprefix", timestamp="20260527_120000")
    assert path.endswith("20260527_120000_myprefix_classifications.csv")
    assert str(tmp_path) in path


def test_default_csv_path_auto_timestamps(tmp_path):
    path = default_csv_path(str(tmp_path), "p")
    # Should match YYYYMMDD_HHMMSS_p_classifications.csv
    assert re.search(r"\d{8}_\d{6}_p_classifications\.csv$", path)


# ---------------------------------------------------------------------------
# write_wide_csv — schema
# ---------------------------------------------------------------------------


def test_wide_csv_default_columns_present(tmp_path):
    wells = ["expA_A01", "expA_A02"]
    store = _make_store(wells)
    fish_line = "myo6b"
    # No assignments at all → all globals zero, singlet inferred to 1.

    out = tmp_path / "out.csv"
    write_wide_csv(
        store=store,
        well_order=wells,
        lhead_map={},
        channels=["GFP"],
        fish_line=fish_line,
        path=str(out),
    )
    df = pd.read_csv(out)
    # Default columns appear in this order before per-channel ones.
    expected_prefix = ["well_name", "empty", "singlet", "multiple", "deformed", "lHead"]
    assert list(df.columns)[: len(expected_prefix)] == expected_prefix
    assert df["empty"].tolist() == [0, 0]
    assert df["singlet"].tolist() == [1, 1]
    assert df["multiple"].tolist() == [0, 0]
    assert df["deformed"].tolist() == [0, 0]
    assert df["lHead"].tolist() == [0, 0]


def test_wide_csv_global_groups_propagate(tmp_path):
    wells = ["expA_A01", "expA_A02", "expA_A03"]
    store = _make_store(wells)
    fish_line = "myo6b"

    # Mark A01 empty in GFP. The LabelStore propagates global groups across
    # channels for this fish_line — that's the upstream contract, here we just
    # confirm the wide-CSV reflects it.
    store._line_channels[fish_line] = ["GFP", "TXR"]
    store.assign(fish_line, "GFP", ["expA_A01"], "empty")
    store.assign(fish_line, "TXR", ["expA_A02"], "multiple")
    store.assign(fish_line, "GFP", ["expA_A03"], "deformed")

    out = tmp_path / "out.csv"
    write_wide_csv(
        store=store,
        well_order=wells,
        lhead_map={},
        channels=["GFP", "TXR"],
        fish_line=fish_line,
        path=str(out),
    )
    df = pd.read_csv(out)
    df = df.set_index("well_name")

    assert df.loc["A01", "empty"] == 1
    assert df.loc["A01", "singlet"] == 0
    assert df.loc["A02", "multiple"] == 1
    assert df.loc["A02", "singlet"] == 0
    assert df.loc["A03", "deformed"] == 1
    assert df.loc["A03", "singlet"] == 0


def test_wide_csv_custom_per_channel_columns(tmp_path):
    wells = ["expA_A01", "expA_A02", "expA_A03"]
    store = _make_store(wells)
    fish_line = "myo6b"
    store._line_channels[fish_line] = ["GFP", "TXR"]

    # Two custom groups in GFP, one in TXR — produces 3 dynamic columns.
    store.assign(fish_line, "GFP", ["expA_A01"], "speckled")
    store.assign(fish_line, "GFP", ["expA_A02"], "smooth")
    store.assign(fish_line, "TXR", ["expA_A02"], "bright_heart")

    out = tmp_path / "out.csv"
    write_wide_csv(
        store=store,
        well_order=wells,
        lhead_map={},
        channels=["GFP", "TXR"],
        fish_line=fish_line,
        path=str(out),
    )
    df = pd.read_csv(out).set_index("well_name")

    assert "GFP_speckled" in df.columns
    assert "GFP_smooth" in df.columns
    assert "TXR_bright_heart" in df.columns

    assert df.loc["A01", "GFP_speckled"] == 1
    assert df.loc["A01", "GFP_smooth"] == 0
    assert df.loc["A01", "TXR_bright_heart"] == 0

    assert df.loc["A02", "GFP_smooth"] == 1
    assert df.loc["A02", "TXR_bright_heart"] == 1

    assert df.loc["A03", "GFP_speckled"] == 0
    assert df.loc["A03", "GFP_smooth"] == 0
    assert df.loc["A03", "TXR_bright_heart"] == 0


def test_wide_csv_lhead_map(tmp_path):
    wells = ["expA_A01", "expA_A02"]
    store = _make_store(wells)
    fish_line = "myo6b"

    out = tmp_path / "out.csv"
    write_wide_csv(
        store=store,
        well_order=wells,
        lhead_map={"expA_A01": True, "expA_A02": False},
        channels=["GFP"],
        fish_line=fish_line,
        path=str(out),
    )
    df = pd.read_csv(out).set_index("well_name")
    assert df.loc["A01", "lHead"] == 1
    assert df.loc["A02", "lHead"] == 0


def test_wide_csv_booleans_are_int(tmp_path):
    wells = ["expA_A01"]
    store = _make_store(wells)
    fish_line = "myo6b"
    store.assign(fish_line, "GFP", ["expA_A01"], "weird")

    out = tmp_path / "out.csv"
    write_wide_csv(
        store=store,
        well_order=wells,
        lhead_map={"expA_A01": True},
        channels=["GFP"],
        fish_line=fish_line,
        path=str(out),
    )
    df = pd.read_csv(out)
    # All scoring columns must be int dtype, never bool or object —
    # `selection_gui.py` reads these by name and expects 0/1 integers.
    for col in ("empty", "singlet", "multiple", "deformed", "lHead", "GFP_weird"):
        assert df[col].dtype.kind == "i", f"{col} is {df[col].dtype}, expected int"


def test_wide_csv_singlet_inference_off(tmp_path):
    wells = ["expA_A01"]
    store = _make_store(wells)
    fish_line = "myo6b"
    store.assign(fish_line, "GFP", ["expA_A01"], "empty")

    out = tmp_path / "out.csv"
    write_wide_csv(
        store=store,
        well_order=wells,
        lhead_map={},
        channels=["GFP"],
        fish_line=fish_line,
        path=str(out),
        infer_singlet=False,
    )
    df = pd.read_csv(out)
    assert df["empty"].tolist() == [1]
    # With inference off, singlet column is always 0.
    assert df["singlet"].tolist() == [0]


def test_wide_csv_well_defaults_override_globals(tmp_path):
    """When `well_defaults` is supplied (Finding Dory's mode), it wins over LabelStore.

    Finding Nemo owns empty/singlet/multiple/deformed/lHead; Finding Dory
    passes those values through `points_layer.features` and they should appear
    verbatim in the CSV, even if LabelStore globals say something different.
    """
    wells = ["expA_A01", "expA_A02"]
    store = _make_store(wells)
    fish_line = "myo6b"

    # LabelStore says A01 is empty — but the points_layer override says otherwise.
    store.assign(fish_line, "GFP", ["expA_A01"], "empty")

    out = tmp_path / "out.csv"
    write_wide_csv(
        store=store,
        well_order=wells,
        lhead_map={},
        channels=["GFP"],
        fish_line=fish_line,
        path=str(out),
        well_defaults={
            "expA_A01": {"empty": 0, "singlet": 1, "multiple": 0, "deformed": 0, "lHead": 1},
            "expA_A02": {"empty": 1, "singlet": 0, "multiple": 0, "deformed": 0, "lHead": 0},
        },
    )
    df = pd.read_csv(out).set_index("well_name")

    assert df.loc["A01", "empty"] == 0    # override wins over LabelStore global
    assert df.loc["A01", "singlet"] == 1
    assert df.loc["A01", "lHead"] == 1
    assert df.loc["A02", "empty"] == 1


def test_wide_csv_well_defaults_partial_falls_back(tmp_path):
    """If well_defaults supplies only some columns, the rest follow normal rules."""
    wells = ["expA_A01"]
    store = _make_store(wells)
    fish_line = "myo6b"

    out = tmp_path / "out.csv"
    write_wide_csv(
        store=store,
        well_order=wells,
        lhead_map={"expA_A01": True},  # fallback for lHead
        channels=["GFP"],
        fish_line=fish_line,
        path=str(out),
        well_defaults={
            # Only lHead specified; the rest fall through to defaults.
            "expA_A01": {"lHead": 0},
        },
    )
    df = pd.read_csv(out).set_index("well_name")
    assert df.loc["A01", "lHead"] == 0  # explicit override beats lhead_map
    assert df.loc["A01", "singlet"] == 1  # inferred from no globals


def test_wide_csv_row_order_matches_well_order(tmp_path):
    wells = ["expA_B02", "expA_A01", "expA_A02"]
    store = _make_store(wells)
    fish_line = "myo6b"

    out = tmp_path / "out.csv"
    write_wide_csv(
        store=store,
        well_order=wells,
        lhead_map={},
        channels=["GFP"],
        fish_line=fish_line,
        path=str(out),
    )
    df = pd.read_csv(out)
    assert df["well_name"].tolist() == ["B02", "A01", "A02"]
