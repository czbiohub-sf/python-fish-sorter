"""Per-channel display configuration for napari.

Vendored from `zebrafish-unsupervised-classification/fish_classify/data/channel_mapping.py`.
Provides the napari colormap and RGB compositing colors used when rendering
individual fluorescence channels. The data path is otherwise channel-agnostic.
"""

import logging
from dataclasses import dataclass
from typing import Dict, Tuple

log = logging.getLogger(__name__)


@dataclass
class ChannelDisplayConfig:
    """Display settings for a single fluorescence channel."""

    napari_colormap: str
    rgb_color: Tuple[float, float, float]
    umap_rgba: Tuple[float, float, float, float]


CHANNEL_DISPLAY_MAP: Dict[str, ChannelDisplayConfig] = {
    "DAPI": ChannelDisplayConfig("blue", (0.0, 0.0, 1.0), (0.2, 0.2, 1.0, 1.0)),
    "BF": ChannelDisplayConfig("gray", (0.5, 0.5, 0.5), (0.7, 0.7, 0.7, 1.0)),
    "GFP": ChannelDisplayConfig("green", (0.0, 1.0, 0.0), (0.0, 0.8, 0.0, 1.0)),
    "CIT": ChannelDisplayConfig("green", (0.4, 0.9, 0.0), (0.4, 0.9, 0.0, 1.0)),
    "TXR": ChannelDisplayConfig("red", (1.0, 0.0, 0.0), (0.8, 0.0, 0.0, 1.0)),
    "CY5": ChannelDisplayConfig("magenta", (0.8, 0.0, 0.8), (0.8, 0.0, 0.8, 1.0)),
    "MCHERRY": ChannelDisplayConfig("red", (1.0, 0.2, 0.0), (1.0, 0.2, 0.0, 1.0)),
}

_FALLBACK_COLORMAPS = ["cyan", "yellow", "magenta"]
_fallback_counter = 0


def get_channel_display(channel_name: str) -> ChannelDisplayConfig:
    """Look up display config for a channel, with fallback for unknown names."""
    global _fallback_counter
    upper = channel_name.upper()
    if upper in CHANNEL_DISPLAY_MAP:
        return CHANNEL_DISPLAY_MAP[upper]

    cmap = _FALLBACK_COLORMAPS[_fallback_counter % len(_FALLBACK_COLORMAPS)]
    _fallback_counter += 1
    fallback_colors = {
        "cyan": (0.0, 1.0, 1.0),
        "yellow": (1.0, 1.0, 0.0),
        "magenta": (0.8, 0.0, 0.8),
    }
    rgb = fallback_colors[cmap]
    cfg = ChannelDisplayConfig(cmap, rgb, (*rgb, 1.0))
    CHANNEL_DISPLAY_MAP[upper] = cfg
    log.warning(
        f"Unknown channel '{channel_name}' assigned fallback colormap '{cmap}'. "
        f"Add it to CHANNEL_DISPLAY_MAP for explicit mapping."
    )
    return cfg
