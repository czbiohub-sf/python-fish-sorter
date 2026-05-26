"""Per-channel uint16 mosaic normalization, matching the training-time
`FishWellLoader.get_well_crop(normalize=True)` semantics.

No module-level defaults — every call takes an explicit `ChannelContrastConfig`,
because the contrast parameters travel with the checkpoint bundle and evolve
between model generations.

Pipeline (one pass per channel):
  1. Compute a 65536-bin histogram via np.bincount and derive low/high
     percentiles (plus an adaptive trimmed-mean upper bound for fluorescent
     channels) from the cumulative.
  2. Linear stretch (x - low) / (high - low), clip to [0, 1], cast to float32.
  3. Optional asinh tonemap (asinh_knee > 0) to expand the dim end of
     fluorescent channels.
"""

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np


@dataclass(frozen=True)
class ChannelContrastConfig:
    """Per-channel contrast curve.

    Two upper-bound modes:
    - ``adaptive_high=False``: linear stretch with ``high_percentile`` as the
      fixed upper bound (typical for BF — predictable, exposure-stable).
    - ``adaptive_high=True``: upper bound is the data-weighted mean of pixels
      in (high_gate_percentile, high_trim_percentile], floored by
      `high_percentile`. Tracks where the bright signal lives without
      per-plate hand-tuning (typical for fluorescent).

    `asinh_knee=0.0` means no post-stretch tonemap. `>0` applies
    `asinh(knee*x) / asinh(knee)` on the [0, 1] stretched image, lifting the
    dim end so faint fluorescence stays visible above noise.
    """

    low_percentile: float
    high_percentile: float
    asinh_knee: float
    adaptive_high: bool = False
    high_gate_percentile: float = 99.5
    high_trim_percentile: float = 99.99

    @classmethod
    def from_dict(cls, d: dict) -> "ChannelContrastConfig":
        return cls(
            low_percentile=float(d["low_percentile"]),
            high_percentile=float(d["high_percentile"]),
            asinh_knee=float(d.get("asinh_knee", 0.0)),
            adaptive_high=bool(d.get("adaptive_high", False)),
            high_gate_percentile=float(d.get("high_gate_percentile", 99.5)),
            high_trim_percentile=float(d.get("high_trim_percentile", 99.99)),
        )


def _uint16_histogram(arr: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
    """Return (counts, cdf, total) for a uint16 array — one O(N) scan."""
    if arr.dtype != np.uint16:
        raise TypeError(f"_uint16_histogram requires uint16, got {arr.dtype}")
    counts = np.bincount(arr.ravel(), minlength=65536)
    cdf = np.cumsum(counts).astype(np.float64)
    return counts, cdf, float(cdf[-1])


def _percentiles_from_cdf(cdf: np.ndarray, total: float, percentiles: List[float]) -> List[float]:
    """Look up percentiles via inverse-cdf search (matches np.percentile for uint16)."""
    out = []
    for p in percentiles:
        target = p / 100.0 * total
        idx = int(np.searchsorted(cdf, target))
        out.append(float(idx))
    return out


def _trimmed_mean_above_from_counts(
    counts: np.ndarray,
    cdf: np.ndarray,
    total: float,
    gate_percentile: float,
    trim_percentile: float,
) -> float:
    """Mean pixel value in (gate_percentile, trim_percentile], from the histogram.

    Falls back to the gate value if the band is empty.
    """
    if not 0.0 <= gate_percentile < trim_percentile <= 100.0:
        raise ValueError(
            f"Need 0 <= gate ({gate_percentile}) < trim ({trim_percentile}) <= 100"
        )
    gate_idx = int(np.searchsorted(cdf, gate_percentile / 100.0 * total))
    trim_idx = int(np.searchsorted(cdf, trim_percentile / 100.0 * total))
    lo = min(gate_idx + 1, 65535)
    hi = min(trim_idx + 1, 65536)
    if hi <= lo:
        return float(gate_idx)
    band_counts = counts[lo:hi]
    band_total = band_counts.sum()
    if band_total == 0:
        return float(gate_idx)
    band_values = np.arange(lo, hi, dtype=np.float64)
    return float((band_counts * band_values).sum() / band_total)


def compute_channel_stats(mosaic: np.ndarray, cfg: ChannelContrastConfig) -> Tuple[float, float]:
    """Return `(low, high)` for the linear stretch step.

    For `adaptive_high=True`, the high bound is `max(p_high_percentile,
    trimmed_mean_above(gate, trim))` — the fixed percentile acts as a floor
    if the adaptive mean is too low.
    """
    counts, cdf, total = _uint16_histogram(mosaic)
    if cfg.adaptive_high:
        low_v, high_floor = _percentiles_from_cdf(
            cdf, total, [cfg.low_percentile, cfg.high_percentile]
        )
        adaptive = _trimmed_mean_above_from_counts(
            counts, cdf, total, cfg.high_gate_percentile, cfg.high_trim_percentile
        )
        return low_v, max(high_floor, adaptive)
    low_v, high_v = _percentiles_from_cdf(
        cdf, total, [cfg.low_percentile, cfg.high_percentile]
    )
    return low_v, high_v


def apply_normalization(
    arr: np.ndarray,
    low: float,
    high: float,
    asinh_knee: float,
) -> np.ndarray:
    """Apply linear stretch + optional asinh tonemap, returning float32 in [0, 1].

    `arr` may be uint16 (raw mosaic / crop) or float; the stretch is pointwise.
    """
    arr = arr.astype(np.float32, copy=False)
    if high > low:
        out = (arr - low) / (high - low)
    else:
        out = arr - low
    np.clip(out, 0.0, 1.0, out=out)
    if asinh_knee > 0.0:
        k = np.float32(asinh_knee)
        out = np.arcsinh(out * k) / np.arcsinh(k)
    return out


def normalize_mosaic(
    mosaic: np.ndarray, channel_name: str, contrast_cfg: ChannelContrastConfig
) -> np.ndarray:
    """One-shot helper: compute (low, high) plate-wide and apply the curve.

    Prefer `compute_channel_stats(mosaic, cfg)` + `apply_normalization(crop, ...)`
    when you want to crop *before* materializing the full float32 mosaic —
    that path is memory-cheaper and produces bitwise-identical wells (the
    normalization is pointwise after the percentiles are known).

    `channel_name` is unused here but kept in the signature so callers
    document intent at the call site.
    """
    del channel_name  # explicit: only used for documentation
    low, high = compute_channel_stats(mosaic, contrast_cfg)
    return apply_normalization(mosaic, low, high, contrast_cfg.asinh_knee)
