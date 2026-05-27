"""Scoped label store for the Finding Dory workflow.

Vendored from `zebrafish-unsupervised-classification/fish_classify/labelling/label_tool.py`
lines 1–281. Kept structurally identical to make future syncs with zebra
straightforward — do not refactor or rename methods without coordinating.

Each *scope* is a ``(fish_line, channel)`` pair; every scope has its own
ordered group list and ``well_id -> group_name`` assignment dict. ``well_id``
is ``<experiment_folder>_<well_name>``.

The wide-CSV serializer (Finding Dory's classify-compatible output) lives in
`fish_sorter/GUI/finding_dory.py`, NOT here — that keeps this file a thin
mirror of the upstream class.
"""

import json
import logging
from collections import defaultdict
from pathlib import Path  # noqa: F401  (kept for parity with upstream)
from typing import Dict, List, Optional, Tuple  # noqa: F401

import numpy as np  # noqa: F401  (kept for parity with upstream)
import pandas as pd

log = logging.getLogger(__name__)

# Default groups always present in every scope
DEFAULT_GROUPS = ["empty", "multiple", "deformed"]

# Global groups propagate across all channels for a fish line and are
# finalized — wells assigned to these cannot be reassigned without
# explicit unassign.
GLOBAL_GROUPS = set(DEFAULT_GROUPS)

# Distinct colours for groups (tab20 palette, RGBA float)
_TAB20 = [
    [0.12, 0.47, 0.71, 1.0],
    [1.00, 0.50, 0.05, 1.0],
    [0.17, 0.63, 0.17, 1.0],
    [0.84, 0.15, 0.16, 1.0],
    [0.58, 0.40, 0.74, 1.0],
    [0.55, 0.34, 0.29, 1.0],
    [0.89, 0.47, 0.76, 1.0],
    [0.50, 0.50, 0.50, 1.0],
    [0.74, 0.74, 0.13, 1.0],
    [0.09, 0.75, 0.81, 1.0],
    [0.68, 0.78, 0.91, 1.0],
    [1.00, 0.73, 0.47, 1.0],
    [0.60, 0.87, 0.54, 1.0],
    [1.00, 0.60, 0.59, 1.0],
    [0.77, 0.69, 0.84, 1.0],
    [0.77, 0.61, 0.58, 1.0],
    [0.97, 0.71, 0.85, 1.0],
    [0.78, 0.78, 0.78, 1.0],
    [0.86, 0.86, 0.55, 1.0],
    [0.62, 0.85, 0.90, 1.0],
]

_NOISE_COLOR = [0.35, 0.35, 0.35, 0.5]
_UNASSIGNED_COLOR = [0.60, 0.60, 0.60, 0.6]


def _scope_key(fish_line: str, channel: str) -> str:
    return f"{fish_line}|{channel}"


class LabelStore:
    """Per-scope label model.

    Each *scope* is a ``(fish_line, channel)`` pair.  Every scope has its
    own ordered group list and ``well_id -> group_name`` assignment dict.
    ``well_id`` is ``<experiment_folder>_<well_name>``.
    """

    def __init__(self, well_metadata: pd.DataFrame):
        self.well_metadata = well_metadata
        self._scopes: Dict[str, dict] = {}
        self._line_channels: Dict[str, List[str]] = {}

    def _get_scope(self, key: str) -> dict:
        if key not in self._scopes:
            self._scopes[key] = {
                "groups": list(DEFAULT_GROUPS),
                "assignments": {},
            }
        return self._scopes[key]

    # -- mutation (all take explicit scope) --------------------------------

    def create_group(self, fish_line: str, channel: str, name: str) -> bool:
        scope = self._get_scope(_scope_key(fish_line, channel))
        if name in scope["groups"]:
            return False
        scope["groups"].append(name)
        return True

    def rename_group(self, fish_line: str, channel: str, old: str, new: str) -> bool:
        scope = self._get_scope(_scope_key(fish_line, channel))
        if old not in scope["groups"] or new in scope["groups"]:
            return False
        idx = scope["groups"].index(old)
        scope["groups"][idx] = new
        for wid in list(scope["assignments"]):
            if scope["assignments"][wid] == old:
                scope["assignments"][wid] = new
        return True

    def delete_group(self, fish_line: str, channel: str, name: str) -> int:
        scope = self._get_scope(_scope_key(fish_line, channel))
        if name not in scope["groups"]:
            return 0
        scope["groups"].remove(name)
        removed = 0
        for wid in list(scope["assignments"]):
            if scope["assignments"][wid] == name:
                del scope["assignments"][wid]
                removed += 1
        return removed

    def is_finalized(self, fish_line: str, well_id: str) -> bool:
        """True if the well is assigned to a global group in any channel."""
        for sk, scope in self._scopes.items():
            if not sk.startswith(fish_line + "|"):
                continue
            g = scope["assignments"].get(well_id)
            if g in GLOBAL_GROUPS:
                return True
        return False

    def assign(self, fish_line: str, channel: str, well_ids: List[str], group: str):
        self.create_group(fish_line, channel, group)
        scope = self._get_scope(_scope_key(fish_line, channel))
        for wid in well_ids:
            if group not in GLOBAL_GROUPS and self.is_finalized(fish_line, wid):
                continue
            scope["assignments"][wid] = group

        if group in GLOBAL_GROUPS:
            all_channels = self._channels_for_line(fish_line)
            for other_ch in all_channels:
                if other_ch == channel:
                    continue
                self.create_group(fish_line, other_ch, group)
                other_scope = self._get_scope(_scope_key(fish_line, other_ch))
                for wid in well_ids:
                    other_scope["assignments"][wid] = group

    def unassign(self, fish_line: str, channel: str, well_ids: List[str]):
        """Unassign wells. For global groups, unassigns across all channels."""
        scope = self._get_scope(_scope_key(fish_line, channel))
        for wid in well_ids:
            removed_group = scope["assignments"].pop(wid, None)
            if removed_group in GLOBAL_GROUPS:
                for sk, other_scope in self._scopes.items():
                    if sk.startswith(fish_line + "|") and sk != _scope_key(fish_line, channel):
                        other_scope["assignments"].pop(wid, None)

    def _channels_for_line(self, fish_line: str) -> List[str]:
        if fish_line in self._line_channels:
            return list(self._line_channels[fish_line])
        channels = []
        prefix = fish_line + "|"
        for sk in self._scopes:
            if sk.startswith(prefix):
                channels.append(sk[len(prefix):])
        return channels

    def _propagate_global_groups(self):
        """Re-propagate all global group assignments to all channels per line."""
        for sk, scope in list(self._scopes.items()):
            fish_line, channel = sk.split("|", 1)
            all_channels = self._channels_for_line(fish_line)
            for wid, group in list(scope["assignments"].items()):
                if group not in GLOBAL_GROUPS:
                    continue
                for other_ch in all_channels:
                    if other_ch == channel:
                        continue
                    self.create_group(fish_line, other_ch, group)
                    other_scope = self._get_scope(_scope_key(fish_line, other_ch))
                    other_scope["assignments"][wid] = group

    # -- queries -----------------------------------------------------------

    def groups(self, fish_line: str, channel: str) -> List[str]:
        return list(self._get_scope(_scope_key(fish_line, channel))["groups"])

    def assignments(self, fish_line: str, channel: str) -> Dict[str, str]:
        return self._get_scope(_scope_key(fish_line, channel))["assignments"]

    def get_group_members(self, fish_line: str, channel: str, group: str) -> List[str]:
        asgn = self.assignments(fish_line, channel)
        return [wid for wid, g in asgn.items() if g == group]

    def group_color(self, fish_line: str, channel: str, group: str) -> List[float]:
        grps = self.groups(fish_line, channel)
        if group not in grps:
            return list(_UNASSIGNED_COLOR)
        idx = grps.index(group) % len(_TAB20)
        return list(_TAB20[idx])

    def counts(self, fish_line: str, channel: str) -> Dict[str, int]:
        c: Dict[str, int] = defaultdict(int)
        for g in self.assignments(fish_line, channel).values():
            c[g] += 1
        return dict(c)

    # -- persistence -------------------------------------------------------

    def save_csv(self, path: str):
        """Save all scoped assignments as a flat (long-format) CSV.

        This is the upstream zebra serializer; Finding Dory uses
        `write_wide_csv` in `fish_sorter/GUI/finding_dory.py` for the
        classify-compatible wide format. This method is kept for parity
        with the upstream class.
        """
        rows = []
        for scope_key, scope in self._scopes.items():
            fish_line, channel = scope_key.split("|", 1)
            for wid, group in scope["assignments"].items():
                match = self.well_metadata[self.well_metadata["well_id"] == wid]
                if len(match) > 0:
                    row = match.iloc[0]
                    rows.append({
                        "well_id": wid,
                        "group": group,
                        "fish_line": fish_line,
                        "channel": channel,
                        "experiment": row["experiment"],
                        "well_name": row["well_name"],
                    })
                else:
                    rows.append({
                        "well_id": wid,
                        "group": group,
                        "fish_line": fish_line,
                        "channel": channel,
                        "experiment": "",
                        "well_name": "",
                    })
        df = pd.DataFrame(rows)
        df.to_csv(path, index=False)
        total = sum(len(s["assignments"]) for s in self._scopes.values())
        log.info(f"Saved {total} assignments across {len(self._scopes)} scopes to {path}")

    def load_csv(self, path: str):
        df = pd.read_csv(path)
        loaded = 0
        for _, row in df.iterrows():
            wid = str(row["well_id"])
            group = str(row.get("group", ""))
            fish_line = str(row.get("fish_line", ""))
            channel = str(row.get("channel", ""))
            if not group or not fish_line or not channel:
                continue
            self.assign(fish_line, channel, [wid], group)
            loaded += 1
        log.info(f"Loaded {loaded} assignments from {path}")

    def to_json(self) -> dict:
        return {"scopes": self._scopes}

    @classmethod
    def from_json(cls, data: dict, well_metadata: pd.DataFrame) -> "LabelStore":
        store = cls(well_metadata)
        store._scopes = data.get("scopes", {})
        return store
