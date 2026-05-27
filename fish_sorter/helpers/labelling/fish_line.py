"""Fish line parsing from experiment folder names.

Vendored from `zebrafish-unsupervised-classification/fish_classify/labelling/fish_line.py`.

Experiment folders follow the convention::

    YYYYMMDD_<age>_<fish_line>[_suffix]

This module strips the date, age markers, and trailing suffixes to extract
the canonical fish line name, then groups experiments by line.
"""

import re
from collections import defaultdict
from typing import Dict, List


_DATE_PREFIX = re.compile(r"^\d{6,8}_")

_AGE_MARKERS = [
    re.compile(r"_?\d+[dh]pf_?"),
    re.compile(r"-\d+[dh]pf"),
]

_TRAILING_SUFFIXES = [
    re.compile(r"_\d+$"),
    re.compile(r"[-_]v\d+$"),
    re.compile(r"_retry$"),
    re.compile(r"_round\d+$"),
    re.compile(r"_pick\d+$"),
    re.compile(r"_plus$"),
    re.compile(r"_\d+x$"),
]


def parse_fish_line(experiment_name: str) -> str:
    """Extract the canonical fish line name from an experiment folder name."""
    name = experiment_name
    name = _DATE_PREFIX.sub("", name)
    for pattern in _AGE_MARKERS:
        name = pattern.sub("", name)
    changed = True
    while changed:
        changed = False
        for pattern in _TRAILING_SUFFIXES:
            new_name = pattern.sub("", name)
            if new_name != name:
                name = new_name
                changed = True
    name = name.strip("_-")
    name = re.sub(r"__+", "_", name)
    return name


def group_experiments_by_line(names: List[str]) -> Dict[str, List[str]]:
    """Group experiment folder names by their parsed fish line."""
    groups: Dict[str, List[str]] = defaultdict(list)
    for name in names:
        line = parse_fish_line(name)
        groups[line].append(name)
    return dict(groups)
