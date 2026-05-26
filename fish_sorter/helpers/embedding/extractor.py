"""Per-channel embedding extraction from raw napari mosaics.

`EmbeddingExtractor` reproduces the training-time normalization pipeline
(`FishWellLoader.get_well_crop(normalize=True)`) and runs the trimmed
`FishDINOv3` backbone over per-well crops, returning per-channel embedding
arrays.

The extractor consumes raw uint16 mosaics directly (`napari.layers.Image.data`)
because percentile normalization is plate-wide — cropping before computing
percentiles would shift the statistics and produce out-of-distribution input.

No caching of any kind: each call recomputes from scratch. Backbone construction
is paid once per `EmbeddingExtractor` instance lifetime (typically once per app
session) and is independent of the per-plate `extract_from_mosaic` calls.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch

from .backbones import FishDINOv3, resolve_weights_path
from .normalize import (
    ChannelContrastConfig,
    apply_normalization,
    compute_channel_stats,
)

log = logging.getLogger(__name__)


def _resolve_device(device_arg: str) -> torch.device:
    """Map `device_arg` ('auto' | 'cuda' | 'mps' | 'cpu') to a torch.device."""
    if device_arg != "auto":
        return torch.device(device_arg)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# Mini-batch defaults per device family. Override at construction if needed.
_BATCH_DEFAULTS = {"cuda": 32, "mps": 16, "cpu": 8}


class EmbeddingExtractor:
    """Loads a checkpoint once; embeds many plates over its lifetime."""

    def __init__(self, cfg: dict, mode: str, batch_size: Optional[int] = None):
        if mode not in cfg.get("models", {}):
            raise ValueError(
                f"mode {mode!r} not found in cfg['models']. "
                f"Available: {sorted(cfg.get('models', {}))}"
            )
        model_cfg = cfg["models"][mode]
        self.mode = mode
        self.cfg = cfg
        self.crop_size: Tuple[int, int] = tuple(model_cfg["crop_size"])  # (h, w)
        resize_to = model_cfg.get("resize_to")
        self.resize_to: Optional[Tuple[int, int]] = (
            tuple(resize_to) if resize_to is not None else None
        )
        self.device = _resolve_device(cfg.get("device", "auto"))
        self.batch_size = batch_size or _BATCH_DEFAULTS.get(self.device.type, 8)

        # Per-channel contrast bundles.
        contrast_block = model_cfg.get("contrast", {})
        if "_FLUOR" not in contrast_block:
            raise ValueError(
                f"models.{mode}.contrast must define a '_FLUOR' fallback bundle."
            )
        self._fluor_contrast = ChannelContrastConfig.from_dict(contrast_block["_FLUOR"])
        self._contrast_by_channel: Dict[str, ChannelContrastConfig] = {
            name: ChannelContrastConfig.from_dict(spec)
            for name, spec in contrast_block.items()
            if name != "_FLUOR"
        }

        # Build the backbone.
        variant = model_cfg.get("model_arch", "vits16")
        repo_path = cfg.get("dinov3_repo_path")
        weights_dir = cfg.get("dinov3_weights_dir")
        weights_path = (
            resolve_weights_path(weights_dir, variant) if weights_dir else None
        )
        log.info(
            f"Constructing FishDINOv3 variant={variant} device={self.device} "
            f"crop_size={self.crop_size} repo={repo_path}"
        )
        self.backbone = FishDINOv3(
            variant=variant, in_channels=1, repo_path=repo_path, weights_path=weights_path,
        )

        # Apply the BYOL-trained checkpoint over the pretrained DINOv3 weights.
        ckpt_path = model_cfg.get("checkpoint_path")
        if not ckpt_path:
            raise ValueError(f"models.{mode}.checkpoint_path is required")
        self._apply_checkpoint(ckpt_path)

        self.backbone.to(self.device).eval()

    # -- checkpoint loading --------------------------------------------------

    def _apply_checkpoint(self, ckpt_path: str) -> None:
        if not Path(ckpt_path).exists():
            raise FileNotFoundError(f"Model checkpoint not found at {ckpt_path}")
        log.info(f"Loading checkpoint: {ckpt_path}")
        # weights_only=True restricts the unpickler to tensors + plain Python
        # types — avoids importing training-time classes (torchmetrics,
        # pytorch-lightning hyperparameter objects, etc.) that aren't installed
        # in this venv. Fall back to a full load if the ckpt has objects the
        # safe loader rejects; in that case the user will need the relevant
        # training deps installed.
        try:
            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        except Exception as e:
            log.warning(
                f"weights_only=True load failed ({type(e).__name__}: {e}); "
                f"falling back to full pickle load. If this raises "
                f"ModuleNotFoundError, install the named training dep in this venv."
            )
            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        sd = ckpt.get("state_dict", ckpt) if isinstance(ckpt, dict) else ckpt

        # Lightning saves the FishDINOv3 under "online_network.backbone." for BYOL.
        # If a different training entry point is used, callers can adjust by
        # passing a pre-stripped state_dict via the config in the future.
        prefix = "online_network.backbone."
        stripped = {k[len(prefix):]: v for k, v in sd.items() if k.startswith(prefix)}
        if not stripped:
            # Maybe the ckpt is already at the FishDINOv3 root — try as-is.
            log.warning(
                f"No keys with prefix {prefix!r} in checkpoint. "
                f"Loading raw state_dict (strict=False)."
            )
            stripped = dict(sd)

        result = self.backbone.load_state_dict(stripped, strict=False)
        if result.missing_keys:
            log.info(f"Checkpoint missing {len(result.missing_keys)} keys (expected for training-only heads).")
        if result.unexpected_keys:
            log.info(f"Checkpoint had {len(result.unexpected_keys)} unexpected keys (likely BYOL projector/predictor).")

    # -- public API ----------------------------------------------------------

    def contrast_for(self, channel_name: str) -> ChannelContrastConfig:
        """Return the contrast bundle for `channel_name`, falling back to fluorescent."""
        return self._contrast_by_channel.get(channel_name.upper(), self._fluor_contrast)

    def extract_from_mosaic(
        self,
        mosaics: Dict[str, np.ndarray],
        well_centers_px: np.ndarray,
        well_crop_px: Tuple[int, int],
        well_indices_to_embed: Optional[np.ndarray] = None,
        progress_cb: Optional[Callable[[int, int], None]] = None,
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
        """Compute per-channel embeddings for the requested wells.

        Args:
            mosaics: Channel-name → full uint16 mosaic from the napari viewer.
            well_centers_px: (N, 2) array of well centers in mosaic pixels,
                ordered (y, x).
            well_crop_px: (h, w) of the per-well crop drawn from the mosaic,
                from `array_json.slot_length / pixel_size_um`. This matches
                the training-time bounding box; we then center-crop to the
                model's `crop_size`.
            well_indices_to_embed: Optional indices into `well_centers_px`;
                normalization still uses the full mosaic for each channel.
            progress_cb: Optional callback(current, total) emitted in roughly
                one increment per channel × batch.

        Returns:
            (per_channel_embeddings, per_channel_indices). Both keyed by
            channel name. `embeddings[c]` has shape (M, output_dim) where M is
            the number of wells embedded for that channel; `indices[c]` gives
            the original index in `well_centers_px` for each row.
        """
        if well_centers_px.ndim != 2 or well_centers_px.shape[1] != 2:
            raise ValueError(
                f"well_centers_px must have shape (N, 2), got {well_centers_px.shape}"
            )
        n_total = well_centers_px.shape[0]
        if well_indices_to_embed is None:
            keep = np.arange(n_total, dtype=np.int64)
        else:
            keep = np.asarray(well_indices_to_embed, dtype=np.int64)

        slot_h, slot_w = well_crop_px
        target_h, target_w = self.crop_size

        per_channel_embeddings: Dict[str, np.ndarray] = {}
        per_channel_indices: Dict[str, np.ndarray] = {}
        steps_total = max(1, len(mosaics))
        step = 0

        for channel_name, mosaic in mosaics.items():
            if mosaic.dtype != np.uint16:
                raise TypeError(
                    f"Channel {channel_name!r}: expected uint16 mosaic, got {mosaic.dtype}"
                )
            cfg = self.contrast_for(channel_name)
            low, high = compute_channel_stats(mosaic, cfg)
            crops_u16 = _crop_wells_uint16(mosaic, well_centers_px[keep], slot_h, slot_w)
            crops_f32 = apply_normalization(crops_u16, low, high, cfg.asinh_knee)
            crops_f32 = _center_crop(crops_f32, target_h, target_w)

            embeddings = self._forward(crops_f32)
            per_channel_embeddings[channel_name] = embeddings
            per_channel_indices[channel_name] = keep.copy()

            step += 1
            if progress_cb is not None:
                progress_cb(step, steps_total)

        return per_channel_embeddings, per_channel_indices

    # -- forward pass --------------------------------------------------------

    def _forward(self, crops: np.ndarray) -> np.ndarray:
        """Run the backbone over `crops` of shape (N, H, W) float32 in [0, 1]."""
        n = crops.shape[0]
        out: List[np.ndarray] = []
        bs = self.batch_size

        autocast_dtype = None
        if self.device.type == "cuda":
            autocast_dtype = torch.float16
        elif self.device.type == "cpu":
            autocast_dtype = torch.bfloat16
        # MPS stays fp32 (autocast for ViT is still flaky).

        for start in range(0, n, bs):
            batch = crops[start : start + bs]
            x = torch.from_numpy(batch).unsqueeze(1)  # (B, 1, H, W)
            x = x.to(self.device, non_blocking=True)
            with torch.inference_mode():
                if autocast_dtype is not None:
                    with torch.autocast(device_type=self.device.type, dtype=autocast_dtype):
                        emb = self.backbone(x)
                else:
                    emb = self.backbone(x)
            out.append(emb.float().cpu().numpy())
        if not out:
            return np.zeros((0, self.backbone.get_embedding_dim()), dtype=np.float32)
        return np.concatenate(out, axis=0)


# ---------------------------------------------------------------------------
# Cropping helpers
# ---------------------------------------------------------------------------


def _crop_wells_uint16(
    mosaic: np.ndarray, centers_yx: np.ndarray, h: int, w: int
) -> np.ndarray:
    """Crop a (N, h, w) uint16 buffer from `mosaic` centered on each `centers_yx`.

    Out-of-bounds pixels are zero-padded.
    """
    n = centers_yx.shape[0]
    mh, mw = mosaic.shape[:2]
    out = np.zeros((n, h, w), dtype=np.uint16)
    half_h, half_w = h // 2, w // 2
    for i in range(n):
        cy, cx = int(centers_yx[i, 0]), int(centers_yx[i, 1])
        y0, y1 = cy - half_h, cy - half_h + h
        x0, x1 = cx - half_w, cx - half_w + w
        # Clip into mosaic frame
        sy0, sy1 = max(0, y0), min(mh, y1)
        sx0, sx1 = max(0, x0), min(mw, x1)
        if sy1 <= sy0 or sx1 <= sx0:
            continue
        oy0, ox0 = sy0 - y0, sx0 - x0
        out[i, oy0:oy0 + (sy1 - sy0), ox0:ox0 + (sx1 - sx0)] = mosaic[sy0:sy1, sx0:sx1]
    return out


def _center_crop(buf: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
    """Center-crop or pad each (h, w) slice to (target_h, target_w)."""
    n, h, w = buf.shape
    if h == target_h and w == target_w:
        return buf
    # Crop dims first, then pad whatever is still short.
    src_y0 = max(0, (h - target_h) // 2)
    src_x0 = max(0, (w - target_w) // 2)
    cropped = buf[:, src_y0:src_y0 + min(h, target_h), src_x0:src_x0 + min(w, target_w)]
    if cropped.shape[1] == target_h and cropped.shape[2] == target_w:
        return cropped
    padded = np.zeros((n, target_h, target_w), dtype=cropped.dtype)
    pad_y0 = (target_h - cropped.shape[1]) // 2
    pad_x0 = (target_w - cropped.shape[2]) // 2
    padded[:, pad_y0:pad_y0 + cropped.shape[1], pad_x0:pad_x0 + cropped.shape[2]] = cropped
    return padded


# ---------------------------------------------------------------------------
# Config loading helper
# ---------------------------------------------------------------------------


def load_config(cfg_path: Path) -> dict:
    """Load and lightly validate the labeller config."""
    with open(cfg_path, "r") as f:
        cfg = json.load(f)
    if "models" not in cfg or not isinstance(cfg["models"], dict):
        raise ValueError(f"{cfg_path}: missing top-level 'models' dict")
    return cfg
