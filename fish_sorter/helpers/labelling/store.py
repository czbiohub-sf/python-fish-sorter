"""Scoped label store for the Finding Dory workflow.

Each experiment is a ``(fish_line, channel)`` pair; every picking experiment has its own
ordered group list and ``well_id -> group_name`` assignment dict. ``well_id``
is ``<experiment_folder>_<well_name>``.
"""

from collections import defaultdict
from typing import Dict, List

import pandas as pd

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

_UNASSIGNED_COLOR = [0.60, 0.60, 0.60, 0.6]


def _scope_key(fish_line: str, channel: str) -> str:
    return f"{fish_line}|{channel}"


class LabelStore:
    """Per-experiment label model.

    Each experiment is a ``(fish_line, channel)`` pair.  Every scope has its
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
        # Purge assignments referencing ``name`` even if it has drifted out
        # of ``groups`` — the assignment vector is the source of truth, so
        # we never leave orphaned wells (which would still render/count as
        # assigned). Return the true number of wells unassigned.
        if name in scope["groups"]:
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
        # Single deduping chokepoint for the channel list: every consumer
        # gets an order-stable, duplicate-free list so cross-channel tuples
        # have exactly one entry per distinct channel.
        if fish_line in self._line_channels:
            return list(dict.fromkeys(self._line_channels[fish_line]))
        channels = []
        prefix = fish_line + "|"
        for sk in self._scopes:
            if sk.startswith(prefix):
                channels.append(sk[len(prefix):])
        return list(dict.fromkeys(channels))

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
