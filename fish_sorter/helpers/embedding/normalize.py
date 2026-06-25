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
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.ndimage import gaussian_filter


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
    invert: bool = False  # BF: flip polarity so the (dark) embryo becomes bright

    @classmethod
    def from_dict(cls, d: dict) -> "ChannelContrastConfig":
        return cls(
            low_percentile=float(d["low_percentile"]),
            high_percentile=float(d["high_percentile"]),
            asinh_knee=float(d.get("asinh_knee", 0.0)),
            adaptive_high=bool(d.get("adaptive_high", False)),
            high_gate_percentile=float(d.get("high_gate_percentile", 99.5)),
            high_trim_percentile=float(d.get("high_trim_percentile", 99.99)),
            invert=bool(d.get("invert", False)),
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
    # Always materialize a fresh float32 buffer we own, then transform it
    # in place — avoids allocating several full-size temporaries per call,
    # which matters for the multi-GB crop stacks in the embedding pipeline.
    out = arr.astype(np.float32)
    out -= low
    if high > low:
        out /= (high - low)
    np.clip(out, 0.0, 1.0, out=out)
    if asinh_knee > 0.0:
        out *= np.float32(asinh_knee)
        np.arcsinh(out, out=out)
        out /= np.arcsinh(asinh_knee)
    return out


# ---------------------------------------------------------------------------
# Multi-contrast (3-view) rendering.
#
# Mirrors zebra's `well_loader.render_multicontrast` / `MC_DEFAULTS` after the
# image-pipeline update that moved multi-contrast from an in-model channel
# adapter into the *data path*: a single raw channel is rendered into 3
# complementary views ([linear, high-pass, bright]) and fed to a 3-channel
# backbone. Kept byte-identical to training so `parity_check.py` holds.
#
# Percentile lookup here reuses `_percentiles_from_cdf` (bincount + searchsorted),
# which is the same algorithm as zebra's `_uint16_percentiles`.
# ---------------------------------------------------------------------------

# Default multi-contrast parameters (locked via zebra's Stage-1 preview).
# Percentiles are computed per mosaic; `bright_k` is the asinh compression for
# the bright view. Overridable per checkpoint via the ckpt's `mc_params` hparam.
MC_DEFAULTS = dict(
    low_pct=0.1, mid_pct=99.96, knee_pct=99.99, ref_pct=99.999,
    bright_k=10.0, blur_sigma=10.0,
)


def resolve_mc_params(mc_params: Optional[dict]) -> dict:
    """Merge checkpoint-provided `mc_params` over `MC_DEFAULTS`."""
    return {**MC_DEFAULTS, **(mc_params or {})}


def compute_mc_bins(
    mosaic: np.ndarray, mc: dict
) -> Tuple[float, float, float, float]:
    """Return the per-mosaic ``(low, mid, knee, ref)`` intensity bins.

    `mc` is a resolved params dict (see :func:`resolve_mc_params`). `ref` is the
    raw max when ``ref_pct >= 100`` (so a single hot pixel just clips), else the
    ``ref_pct`` percentile — matching zebra's bin computation in
    ``FishWellLoader.__init__``.
    """
    _, cdf, total = _uint16_histogram(mosaic)
    low, mid, knee = _percentiles_from_cdf(
        cdf, total, [mc["low_pct"], mc["mid_pct"], mc["knee_pct"]]
    )
    if mc["ref_pct"] >= 100.0:
        ref = float(mosaic.max())
    else:
        ref = _percentiles_from_cdf(cdf, total, [mc["ref_pct"]])[0]
    return low, mid, knee, ref


def render_multicontrast(
    raw2d: np.ndarray, low: float, mid: float, knee: float, ref: float,
    bright_k: float = 10.0, blur_sigma: float = 10.0, invert: bool = False,
) -> np.ndarray:
    """Render a raw single-channel crop into 3 complementary views, (3,H,W) [0,1].

    Byte-identical port of zebra's ``render_multicontrast``. Views:
      [0] linear   = stretch (low, mid), clip [0,1]
      [1] highpass = linear - gaussianblur(linear), per-image min/max -> [0,1]
      [2] bright   = asinh threshold: dark below ``mid``; above,
                     arcsinh(((raw-mid)/(knee-mid))*k) normalized by the same at
                     the robust reference ``ref`` (NOT the max), clipped [0,1].

    ``invert`` (used for brightfield) flips polarity so the dark embryo becomes
    bright, matching fluorescence: the linear view is inverted, the bright view
    is recomputed as an asinh expansion of that inverted-bright signal, and the
    highpass (structure) is derived from the inverted linear.
    """
    raw2d = raw2d.astype(np.float32)
    if invert:
        # Absorption BF: don't clip the dark (embryo) end, or it saturates to
        # pure white after inversion. Uncap the low bound to the raw floor.
        low = 0.0
    linear = np.clip((raw2d - low) / max(mid - low, 1e-6), 0.0, 1.0).astype(np.float32)

    if invert:
        linear = (1.0 - linear).astype(np.float32)

    blurred = gaussian_filter(linear, sigma=blur_sigma, mode="reflect")
    hp = linear - blurred
    rng = float(hp.max() - hp.min())
    hp = ((hp - hp.min()) / (rng if rng > 1e-8 else 1.0)).astype(np.float32)

    if invert:
        # No meaningful "bright foci" in absorption BF; expand the inverted
        # (now-bright) embryo signal with the same asinh curve for view 3.
        bright = np.clip(
            np.arcsinh(linear * bright_k) / np.arcsinh(bright_k), 0.0, 1.0
        ).astype(np.float32)
    else:
        x = np.clip(raw2d - mid, 0.0, None) / max(knee - mid, 1e-6)
        b = np.arcsinh(x * bright_k)
        bref = float(np.arcsinh((ref - mid) / max(knee - mid, 1e-6) * bright_k))
        bright = np.clip(b / (bref if bref > 1e-8 else 1.0), 0.0, 1.0).astype(np.float32)

    return np.stack([linear, hp, bright], axis=0)
