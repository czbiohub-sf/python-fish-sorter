"""Finding Dory scaffold — module-level wide-CSV writer.

The full `FindingDory(QWidget)` dock lands in Chunk 4. This file currently
holds only the classify-compatible CSV serializer that Chunk 4 will call from
the dock's "Save" button.

The wide-CSV writer lives here (the workflow layer) rather than in
`helpers/labelling/store.py` (the data model) so the vendored `LabelStore`
stays a thin mirror of upstream zebra — future re-syncs stay clean.
"""

from __future__ import annotations

import logging
import os
from datetime import datetime
from typing import Dict, Iterable, List, Optional

import pandas as pd

from fish_sorter.helpers.labelling.store import GLOBAL_GROUPS, LabelStore

log = logging.getLogger(__name__)


def default_csv_path(expt_dir: str, prefix: str, timestamp: Optional[str] = None) -> str:
    """Construct the `{TIMESTAMP}_{PREFIX}_classifications.csv` path.

    Matches `classify.py:316–317` — Finding Dory writes alongside Finding Nemo
    so downstream `SelectGUI` reads either interchangeably.
    """
    if timestamp is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    fname = f"{timestamp}_{prefix}_classifications.csv"
    return os.path.normpath(os.path.join(expt_dir, fname))


def write_wide_csv(
    store: LabelStore,
    well_order: List[str],
    lhead_map: Dict[str, bool],
    channels: Iterable[str],
    fish_line: str,
    path: str,
    infer_singlet: bool = True,
) -> str:
    """Write a classify-compatible wide CSV from a `LabelStore` snapshot.

    Schema (one row per well, columns in this order):

      well_name, empty, singlet, multiple, deformed, lHead,
      {channel0}_{label0}, {channel0}_{label1}, ...,
      {channel1}_{label0}, ...

    Where:
      - well_name comes from `store.well_metadata` (matched by well_id).
      - empty/multiple/deformed are 1 iff the well is assigned to the
        same-named global group in any scope of `fish_line`. (Global groups
        propagate across all channels per `LabelStore.assign()`.)
      - lHead = int(lhead_map.get(well_id, False)).
      - singlet, if `infer_singlet=True`, is the int complement of any global:
        `int(not (empty or multiple or deformed))`. Set False to write singlet
        as 0 everywhere (e.g. if the caller manages singlet separately).
      - `{channel}_{group}` is 1 iff the well is assigned to that non-global
        group in that channel's scope, else 0.

    Args:
        store: The `LabelStore` to serialize.
        well_order: Ordered list of `well_id`s; produces one CSV row each.
        lhead_map: `well_id -> bool`, typically from `Classify.find_orientation`.
        channels: Channels whose custom-group columns get emitted.
        fish_line: Scope key — wells are looked up in `(fish_line, channel)`.
        path: Output CSV path (use `default_csv_path` for the standard name).
        infer_singlet: When True, `singlet = not (empty or multiple or deformed)`.

    Returns:
        The absolute path of the written CSV.
    """
    channels = list(channels)

    # Collect per-channel custom (non-global) groups, preserving creation order.
    custom_by_channel: Dict[str, List[str]] = {}
    for ch in channels:
        all_groups = store.groups(fish_line, ch)
        custom_by_channel[ch] = [g for g in all_groups if g not in GLOBAL_GROUPS]

    # Build well_id -> well_name lookup from metadata.
    well_name_by_id: Dict[str, str] = {}
    if "well_id" in store.well_metadata.columns and "well_name" in store.well_metadata.columns:
        for _, row in store.well_metadata[["well_id", "well_name"]].iterrows():
            well_name_by_id[str(row["well_id"])] = str(row["well_name"])

    rows = []
    for wid in well_order:
        # Globals: True if assigned anywhere in this fish_line's scopes.
        flags = {g: 0 for g in ("empty", "multiple", "deformed")}
        for ch in channels:
            assigned = store.assignments(fish_line, ch).get(wid)
            if assigned in flags:
                flags[assigned] = 1

        singlet = (
            int(not (flags["empty"] or flags["multiple"] or flags["deformed"]))
            if infer_singlet
            else 0
        )

        row = {
            "well_name": well_name_by_id.get(wid, wid),
            "empty": flags["empty"],
            "singlet": singlet,
            "multiple": flags["multiple"],
            "deformed": flags["deformed"],
            "lHead": int(bool(lhead_map.get(wid, False))),
        }

        # Per-channel custom columns: {channel}_{group}, in (channel, group) order.
        for ch in channels:
            assigned = store.assignments(fish_line, ch).get(wid)
            for g in custom_by_channel[ch]:
                row[f"{ch}_{g}"] = int(assigned == g)

        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(path, index=False)
    log.info(f"wrote wide CSV {df.shape[0]} rows x {df.shape[1]} cols → {path}")
    return path
