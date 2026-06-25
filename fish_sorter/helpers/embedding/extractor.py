"""Per-channel embedding extraction from raw napari mosaics.

`EmbeddingExtractor` reproduces the training-time normalization pipeline
(`FishWellLoader.get_well_crop(normalize=True)`) and runs the trimmed
`FishDINOv3` backbone over per-well crops, returning per-channel embedding
arrays.

The extractor consumes raw uint16 mosaics directly (`napari.layers.Image.data`)
because percentile normalization is plate-wide — cropping before computing
percentiles would shift the statistics and produce out-of-distribution input.
"""

from __future__ import annotations

import json
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch

from .backbones import FishDINOv3, resolve_weights_path
from .normalize import (
    ChannelContrastConfig,
    apply_normalization,
    compute_channel_stats,
    compute_mc_bins,
    render_multicontrast,
    resolve_mc_params,
)

log = logging.getLogger(__name__)


def _resolve_device(device_arg: str) -> torch.device:
    """Map `device_arg` ('auto' | 'cuda' | 'mps' | 'cpu') to a torch.device.

    Explicit choices (`cuda` / `mps`) are validated up-front; if the requested
    accelerator isn't available, we raise instead of silently falling back —
    you almost certainly don't want a CPU inference run by accident.
    """
    if device_arg == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "config device='cuda' but CUDA is not available. "
                "Check NVIDIA driver + torch CUDA wheel install, or set device='auto' "
                "(falls back to MPS or CPU) or device='cpu'."
            )
        return torch.device("cuda")
    if device_arg == "mps":
        if not (getattr(torch.backends, "mps", None) is not None
                and torch.backends.mps.is_available()):
            raise RuntimeError(
                "config device='mps' but MPS is not available. "
                "Set device='auto' or device='cpu' instead."
            )
        return torch.device("mps")
    if device_arg == "cpu":
        return torch.device("cpu")
    if device_arg != "auto":
        return torch.device(device_arg)
    # 'auto': prefer cuda > mps > cpu
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# Mini-batch defaults per device family. Override at construction if needed.
_BATCH_DEFAULTS = {"cuda": 32, "mps": 16, "cpu": 8}

# Physical-scale normalization target. Mirrors zebra's
# `well_loader.STANDARD_PIXEL_UM` (the 2.5x objective scale): each crop is
# resampled to this um/px so a fixed center crop covers the same physical area
# across magnifications (e.g. a 5x plate -> 2.5x standard). MUST equal zebra's
# value exactly for embedding parity. A 2.5x plate (~2.6 um/px) is within the
# skip tolerance and passes through untouched.
STANDARD_PIXEL_UM = 5324.8 / 2048.0  # ~2.600 um/px
_RESCALE_SKIP_TOL = 0.05  # skip resampling when within 5% of target um/px


class EmbeddingExtractor:
    """Loads the model, runs the forward pass, and returns per-channel embeddings."""

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
        # Physical-scale resampling target (um/px). Defaults to the 2.5x
        # standard; set per-model or top-level `target_pixel_um: null` to
        # disable. The actual resample factor needs the plate's pixel size,
        # passed per call to `extract_from_mosaic`.
        self.target_pixel_um: Optional[float] = model_cfg.get(
            "target_pixel_um", cfg.get("target_pixel_um", STANDARD_PIXEL_UM)
        )

        device_arg = cfg.get("device", "auto")
        self.device = _resolve_device(device_arg)
        log.info(f"EmbeddingExtractor device: {self.device} (config: {device_arg!r})")
        # Priority: explicit constructor arg > top-level config > per-device default.
        self.batch_size = (
            batch_size
            or cfg.get("batch_size")
            or _BATCH_DEFAULTS.get(self.device.type, 8)
        )
        # Default True for speed; set false at top-level config to force fp32 forward.
        self.use_autocast = bool(cfg.get("autocast", True))

        # Per-channel contrast bundles.
        # these allow custom normalization parameters per channel (ie DAPI can normalize differently from GFP if needed), but require a "_FLUOR" fallback for default params.
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

        # Pre-load ckpt so we can detect training-time options that change the
        # MODEL graph (e.g. multi_contrast channel adapter, which adds a blur
        # kernel buffer). The detected flags then drive FishDINOv3 construction.
        ckpt_path = model_cfg.get("checkpoint_path")
        if not ckpt_path:
            raise ValueError(f"models.{mode}.checkpoint_path is required")
        sd, hparams = self._load_checkpoint(ckpt_path)

        # Resolve which multi-contrast scheme the checkpoint uses. Two exist:
        #   - data-path render (current): the 3 views ([linear, highpass,
        #     bright]) are synthesized in normalize.render_multicontrast and the
        #     backbone takes in_channels=3 (its channel adapter is pass-through).
        #     Signalled by in_channels==3 in the ckpt's hyper_parameters.
        #   - legacy in-model adapter: the old UnifiedChannelAdapter builds the
        #     3 views *inside* the graph from a 1-channel input, detected via its
        #     `channel_adapter._blur_kernel` buffer.
        # Config may force a boolean via `multi_contrast`; data-path is preferred
        # whenever the ckpt says in_channels==3.
        ckpt_in_channels = int(hparams.get("in_channels", 0) or 0)
        has_blur_kernel = any("channel_adapter._blur_kernel" in k for k in sd)
        forced = model_cfg.get("multi_contrast")  # None | bool
        if forced is not None:
            log.info(f"multi_contrast override from config: {forced}")

        if ckpt_in_channels == 3 or (forced and not has_blur_kernel):
            # New data-path multi-contrast.
            self.render_mc = True
            model_in_channels = 3
            model_multi_contrast = False
            self.mc = resolve_mc_params(
                model_cfg.get("mc_params") or hparams.get("mc_params")
            )
            log.info(
                f"multi-contrast: data-path render (in_channels=3), mc={self.mc}"
            )
        elif has_blur_kernel or forced:
            # Legacy in-model multi-contrast (1-channel input, adapter expands).
            self.render_mc = False
            model_in_channels = 1
            model_multi_contrast = True
            self.mc = None
            log.info("multi-contrast: legacy in-model adapter (in_channels=1)")
        else:
            self.render_mc = False
            model_in_channels = 1
            model_multi_contrast = False
            self.mc = None
            log.info("single-channel mode (no multi-contrast)")

        # Build the backbone.
        variant = model_cfg.get("model_arch", "vitb16")
        repo_path = cfg.get("dinov3_repo_path")
        weights_dir = cfg.get("dinov3_weights_dir")
        weights_path = (
            resolve_weights_path(weights_dir, variant) if weights_dir else None
        )
        log.info(
            f"Constructing FishDINOv3 variant={variant} device={self.device} "
            f"crop_size={self.crop_size} repo={repo_path} "
            f"in_channels={model_in_channels} multi_contrast={model_multi_contrast}"
        )
        self.backbone = FishDINOv3(
            variant=variant,
            in_channels=model_in_channels,
            repo_path=repo_path,
            weights_path=weights_path,
            multi_contrast=model_multi_contrast,
        )

        # Apply the (already-loaded) state dict.
        self._apply_state_dict(sd)

        self.backbone.to(self.device).eval()

    # -- checkpoint loading --------------------------------------------------

    def _load_checkpoint(self, ckpt_path: str) -> Tuple[Dict, Dict]:
        """Open the ckpt and return ``(state_dict, hyper_parameters)``.

        Tries `weights_only=True` first to avoid importing training-time
        classes (torchmetrics, pytorch-lightning, etc.). Falls back to the full
        pickle loader if the ckpt has non-allowlisted objects.

        `hyper_parameters` (Lightning's saved hparams) carries the data-pipeline
        flags we need to reproduce at inference — `in_channels` and `mc_params`.
        Returns an empty dict for it on bare state-dict checkpoints.
        """
        if not Path(ckpt_path).exists():
            raise FileNotFoundError(f"Model checkpoint not found at {ckpt_path}")
        log.info(f"Loading checkpoint: {ckpt_path}")
        try:
            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        except Exception as e:
            log.warning(
                f"weights_only=True load failed ({type(e).__name__}: {e}); "
                f"falling back to full pickle load. If this raises "
                f"ModuleNotFoundError, install the named training dep in this venv."
            )
            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        if isinstance(ckpt, dict):
            sd = ckpt.get("state_dict", ckpt)
            hparams = ckpt.get("hyper_parameters", {}) or {}
        else:
            sd, hparams = ckpt, {}
        return sd, hparams

    def _apply_state_dict(self, sd: Dict) -> None:
        """Strip the right prefix from `sd` and load it into self.backbone."""
        # Try common prefixes; pick whichever gives a non-empty stripped dict.
        # Lightning typically saves the FishDINOv3 under "online_network.backbone."
        # for BYOL, but other entry points can wrap differently.
        # this is a bit hacky but it's robust to whatever wrapping the training code did, and ammendable if the training repo is updated.
        candidate_prefixes = (
            "online_backbone.",
            "online_network.backbone.",
            "model.online_backbone.",
            "model.online_network.backbone.",
            "module.online_backbone.",
            "module.online_network.backbone.",
            "backbone.",
            "",  # already at FishDINOv3 root
        )
        stripped: Dict = {}
        chosen = ""
        if sd:
            for prefix in candidate_prefixes:
                sub = {k[len(prefix):]: v for k, v in sd.items() if k.startswith(prefix)}
                if sub:
                    stripped = sub
                    chosen = prefix
                    break
            if not stripped:
                log.error(
                    f"No candidate prefix matched any keys. First 10 ckpt keys: "
                    f"{list(sd.keys())[:10]}"
                )
                raise RuntimeError("Could not find a usable key prefix in checkpoint.")

        result = self.backbone.load_state_dict(stripped, strict=False)
        # note if missing keys are found, the model may or may not produce meaningful embeddings.
        # this is a sanity check log.
        log.info(
            f"checkpoint loaded with prefix={chosen!r}: "
            f"{len(stripped) - len(result.unexpected_keys)} keys applied, "
            f"{len(result.missing_keys)} missing, "
            f"{len(result.unexpected_keys)} unexpected"
        )
        # Loud diagnostic: if nearly nothing loaded, the prefix is wrong and
        # the model is still at random init — embeddings will be meaningless.
        backbone_param_count = sum(1 for _ in self.backbone.state_dict())
        applied = len(stripped) - len(result.unexpected_keys)
        if applied < backbone_param_count * 0.5:
            log.warning(
                f"Only {applied}/{backbone_param_count} backbone params got "
                f"weights from the ckpt — the model is mostly at random init. "
                f"Embeddings will not match a properly-trained reference. "
                f"First 10 ckpt keys: {list(sd.keys())[:10]}"
            )

    # -- public API ----------------------------------------------------------

    def contrast_for(self, channel_name: str) -> ChannelContrastConfig:
        """Return the contrast bundle for `channel_name`, falling back to fluorescent."""
        return self._contrast_by_channel.get(channel_name.upper(), self._fluor_contrast)

    def _resolve_rescale_factor(self, pixel_size_um: Optional[float]) -> Optional[float]:
        """Return the resample factor for this plate, or None to skip.

        Factor = ``pixel_size_um / target_pixel_um`` (mirrors zebra's
        `FishWellLoader.__init__`); skipped when the plate is within
        `_RESCALE_SKIP_TOL` of the target or when either scale is missing.
        """
        if not pixel_size_um or not self.target_pixel_um or self.target_pixel_um <= 0:
            if pixel_size_um is None:
                log.info(
                    "pixel_size_um not provided; skipping physical-scale resampling "
                    "(correct only if the plate is already at the target scale)"
                )
            return None
        ratio = pixel_size_um / self.target_pixel_um
        if abs(ratio - 1.0) <= _RESCALE_SKIP_TOL:
            log.info(
                f"plate {pixel_size_um:.3f} um/px within {_RESCALE_SKIP_TOL:.0%} of "
                f"target {self.target_pixel_um:.3f}; no resampling"
            )
            return None
        log.info(
            f"resampling crops {pixel_size_um:.3f} -> {self.target_pixel_um:.3f} um/px "
            f"(factor {ratio:.3f})"
        )
        return ratio

    def extract_from_mosaic(
        self,
        mosaics: Dict[str, np.ndarray],
        well_centers_px: np.ndarray,
        well_crop_px: Tuple[int, int],
        well_indices_to_embed: Optional[np.ndarray] = None,
        progress_cb: Optional[Callable[[int, int], None]] = None,
        pixel_size_um: Optional[float] = None,
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
        """Compute per-channel embeddings for the requested wells.

        Args:
            mosaics: Channel-name -> full uint16 mosaic from the napari viewer.
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
            pixel_size_um: This plate's physical scale (um/px), e.g. CAM_PX_UM /
                magnification. When given (and `self.target_pixel_um` is set),
                each raw crop is resampled to the target scale so a fixed center
                crop covers the same physical area training saw — required for
                parity at non-2.5x magnifications. `None` skips resampling
                (correct only when the plate is already at the target scale).

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

        # Physical-scale resample factor: < 1 downsamples a higher-mag plate to
        # the target um/px. Skipped when within tolerance (e.g. a 2.5x plate) or
        # when the plate scale wasn't supplied.
        rescale_factor = self._resolve_rescale_factor(pixel_size_um)

        # Worker count for the per-well multi-contrast render (GIL-releasing).
        mc_workers = int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count() or 4))

        per_channel_embeddings: Dict[str, np.ndarray] = {}
        per_channel_indices: Dict[str, np.ndarray] = {}
        steps_total = max(1, len(mosaics))
        step = 0

        for channel_name, mosaic in mosaics.items():
            if mosaic.dtype != np.uint16:
                raise TypeError(
                    f"Channel {channel_name!r}: expected uint16 mosaic, got {mosaic.dtype}"
                )
            ch_t0 = time.perf_counter()
            log.info(f"[{channel_name}] mosaic shape={mosaic.shape} dtype={mosaic.dtype}")

            cfg = self.contrast_for(channel_name)

            t0 = time.perf_counter()
            crops_u16 = _crop_wells_uint16(mosaic, well_centers_px[keep], slot_h, slot_w)
            log.info(
                f"[{channel_name}] cropped {len(keep)} wells "
                f"(slot {slot_h}x{slot_w}, buf {crops_u16.nbytes/1e6:.1f} MB, "
                f"{time.perf_counter()-t0:.2f}s)"
            )

            # Physical-scale resample on the raw float32 crop, *before* render /
            # normalize and *before* center-cropping — matching training, where
            # `get_well_crop` rescales the float32 crop and the val transform
            # center-crops afterward. Both downstream branches consume `crops`.
            if rescale_factor is not None:
                t0 = time.perf_counter()
                crops = _rescale_batch(crops_u16, rescale_factor)  # float32
                log.info(
                    f"[{channel_name}] resampled {slot_h}x{slot_w} -> "
                    f"{crops.shape[1]}x{crops.shape[2]} (factor {rescale_factor:.3f}, "
                    f"{time.perf_counter()-t0:.2f}s)"
                )
            else:
                crops = crops_u16

            if self.render_mc:
                t0 = time.perf_counter()
                low, mid, knee, ref = compute_mc_bins(mosaic, self.mc)
                log.info(
                    f"[{channel_name}] mc bins low={low:.1f} mid={mid:.1f} "
                    f"knee={knee:.1f} ref={ref:.1f} invert={cfg.invert} "
                    f"({time.perf_counter()-t0:.2f}s)"
                )
                t0 = time.perf_counter()
                # Render the 3 views at the *slot* extent, then center-crop. The
                # high-pass blur and its per-image min/max are spatial, so this
                # render-then-crop order must match training (zebra renders on the
                # full well crop, then the val transform center-crops) — cropping
                # first would change the high-pass normalization. Rendered in
                # parallel across wells (the blur/arcsinh release the GIL).
                views = _render_multicontrast_batch(
                    crops, low, mid, knee, ref,
                    bright_k=self.mc["bright_k"],
                    blur_sigma=self.mc["blur_sigma"],
                    invert=cfg.invert,
                    max_workers=mc_workers,
                )  # (N, 3, h, w)
                crops_f32 = _center_crop(views, target_h, target_w)
                log.info(
                    f"[{channel_name}] rendered 3-view multi-contrast "
                    f"({mc_workers} workers) + center-cropped to "
                    f"{target_h}x{target_w} ({time.perf_counter()-t0:.2f}s)"
                )
            else:
                t0 = time.perf_counter()
                low, high = compute_channel_stats(mosaic, cfg)
                log.info(
                    f"[{channel_name}] percentiles low={low:.1f} high={high:.1f} "
                    f"asinh_knee={cfg.asinh_knee} ({time.perf_counter()-t0:.2f}s)"
                )
                t0 = time.perf_counter()
                # Center-crop *first*, then normalize. Normalization is pointwise
                # (its low/high/asinh come from the full mosaic, not the crop), so
                # cropping first is identical in result but skips normalizing the
                # ~40% of pixels discarded. `crops` is uint16 here (or float32 if
                # resampled above); apply_normalization handles both.
                cc = _center_crop(crops, target_h, target_w)
                crops_f32 = apply_normalization(cc, low, high, cfg.asinh_knee)
                log.info(
                    f"[{channel_name}] center-cropped to {target_h}x{target_w} + "
                    f"normalized ({time.perf_counter()-t0:.2f}s)"
                )

            t0 = time.perf_counter()
            embeddings = self._forward(crops_f32, log_prefix=f"[{channel_name}]")
            log.info(
                f"[{channel_name}] forward pass {embeddings.shape} "
                f"({time.perf_counter()-t0:.2f}s total)"
            )

            per_channel_embeddings[channel_name] = embeddings
            per_channel_indices[channel_name] = keep.copy()
            log.info(f"[{channel_name}] DONE ({time.perf_counter()-ch_t0:.2f}s)")

            step += 1
            if progress_cb is not None:
                progress_cb(step, steps_total)

        return per_channel_embeddings, per_channel_indices

    # -- forward pass --------------------------------------------------------

    def _forward(self, crops: np.ndarray, log_prefix: str = "") -> np.ndarray:
        """Run the backbone over `crops` float32 in [0, 1].

        Accepts `(N, H, W)` (single-channel; a channel axis is added) or
        `(N, C, H, W)` (multi-contrast views; passed through as-is).
        """
        n = crops.shape[0]
        out: List[np.ndarray] = []
        bs = self.batch_size

        autocast_dtype = None
        if self.use_autocast:
            if self.device.type == "cuda":
                autocast_dtype = torch.float16
            elif self.device.type == "cpu":
                autocast_dtype = torch.bfloat16
            # MPS stays fp32

        n_batches = (n + bs - 1) // bs
        log.info(
            f"{log_prefix} forward: {n} wells, batch_size={bs}, "
            f"{n_batches} batches on {self.device} "
            f"(autocast={autocast_dtype})"
        )

        for batch_idx, start in enumerate(range(0, n, bs), start=1):
            t0 = time.perf_counter()
            batch = crops[start : start + bs]
            x = torch.from_numpy(batch)
            if x.ndim == 3:
                x = x.unsqueeze(1)  # (B, H, W) -> (B, 1, H, W)
            # else already (B, C, H, W) for multi-contrast views
            x = x.to(self.device, non_blocking=True)
            with torch.inference_mode():
                if autocast_dtype is not None:
                    with torch.autocast(device_type=self.device.type, dtype=autocast_dtype):
                        emb = self.backbone(x)
                else:
                    emb = self.backbone(x)
            if self.device.type == "cuda":
                torch.cuda.synchronize()
            out.append(emb.float().cpu().numpy())
            dt = time.perf_counter() - t0
            log.info(
                f"{log_prefix}  batch {batch_idx}/{n_batches} "
                f"({batch.shape[0]} wells) in {dt:.2f}s "
                f"-> {batch.shape[0]/dt:.1f} wells/s"
            )
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


def _render_multicontrast_batch(
    crops: np.ndarray,
    low: float,
    mid: float,
    knee: float,
    ref: float,
    *,
    bright_k: float,
    blur_sigma: float,
    invert: bool,
    max_workers: int,
) -> np.ndarray:
    """Render each crop in `crops` (N, H, W) to 3 views, returning (N, 3, H, W).

    The per-well `render_multicontrast` is dominated by `scipy.gaussian_filter`
    and `np.arcsinh`, both of which release the GIL, so a thread pool gives real
    parallelism. `ThreadPoolExecutor.map` preserves input order, so the output
    stays aligned to the well order (required to keep embeddings ↔ indices
    aligned).
    """
    n = crops.shape[0]
    h, w = crops.shape[-2], crops.shape[-1]
    if n == 0:
        return np.zeros((0, 3, h, w), dtype=np.float32)

    def _one(i: int) -> np.ndarray:
        return render_multicontrast(
            crops[i], low, mid, knee, ref,
            bright_k=bright_k, blur_sigma=blur_sigma, invert=invert,
        )

    workers = max(1, min(max_workers, n))
    if workers == 1:
        views = [_one(i) for i in range(n)]
    else:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            views = list(executor.map(_one, range(n)))
    return np.stack(views, axis=0)


def _rescale_batch(buf: np.ndarray, factor: float) -> np.ndarray:
    """Resample a batch of crops by `factor`, returning float32.

    Mirrors zebra's `FishWellLoader._maybe_rescale`: antialiased bilinear
    interpolation (correct for both up- and down-sampling) on the float32 crop.
    `buf` is `(N, H, W)`; the new size is `round(H*factor) x round(W*factor)`.
    Each crop is resampled independently, so batching matches per-crop results.
    """
    import torch
    import torch.nn.functional as F

    new_h = max(1, int(round(buf.shape[-2] * factor)))
    new_w = max(1, int(round(buf.shape[-1] * factor)))
    t = torch.from_numpy(np.ascontiguousarray(buf)).float().unsqueeze(1)  # (N,1,H,W)
    t = F.interpolate(
        t, size=(new_h, new_w), mode="bilinear", align_corners=False, antialias=True
    )
    return t.squeeze(1).numpy()  # (N, new_h, new_w) float32


def _center_crop(buf: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
    """Center-crop or pad the trailing (h, w) of `buf` to (target_h, target_w).

    Works for both `(N, h, w)` (single-channel) and `(N, C, h, w)`
    (multi-contrast views) — only the last two axes are touched.
    """
    h, w = buf.shape[-2], buf.shape[-1]
    if h == target_h and w == target_w:
        return buf
    lead = buf.shape[:-2]
    # Crop dims first, then pad whatever is still short.
    src_y0 = max(0, (h - target_h) // 2)
    src_x0 = max(0, (w - target_w) // 2)
    cropped = buf[..., src_y0:src_y0 + min(h, target_h), src_x0:src_x0 + min(w, target_w)]
    if cropped.shape[-2] == target_h and cropped.shape[-1] == target_w:
        return cropped
    padded = np.zeros((*lead, target_h, target_w), dtype=cropped.dtype)
    pad_y0 = (target_h - cropped.shape[-2]) // 2
    pad_x0 = (target_w - cropped.shape[-1]) // 2
    padded[..., pad_y0:pad_y0 + cropped.shape[-2], pad_x0:pad_x0 + cropped.shape[-1]] = cropped
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


# ---------------------------------------------------------------------------
# Mode resolution + one-shot embedding pass (shared by the Finding Dory dock
# and the post-stitch background pre-warm)
# ---------------------------------------------------------------------------


def resolve_mode(cfg: dict, pick_type: str, default: str = "fish") -> str:
    """Map a fish-sorter ``pick_type`` to a labeller model bundle name.

    Reads ``cfg['pick_type_to_mode']``. Unknown pick types fall back to
    ``default`` with a warning — the caller still gets a usable mode, but the
    log flags the mismatch so a misconfigured ``pick_type_to_mode`` is visible.
    """
    mode_map = cfg.get("pick_type_to_mode", {})
    if pick_type in mode_map:
        return mode_map[pick_type]
    log.warning(
        f"pick_type {pick_type!r} not in pick_type_to_mode; "
        f"falling back to mode={default!r}"
    )
    return default


def compute_embeddings(
    cfg: dict,
    mode: str,
    *,
    channels: List[str],
    mosaics: Dict[str, np.ndarray],
    well_centers: np.ndarray,
    well_crop_px: Tuple[int, int],
    n_total: int,
    keep_indices: Optional[np.ndarray] = None,
    progress_cb: Optional[Callable[[int, int], None]] = None,
    status_cb: Optional[Callable[[str], None]] = None,
    pixel_size_um: Optional[float] = None,
) -> Tuple[Optional["EmbeddingExtractor"], Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """Build the extractor (or mock) and embed every channel.

    This is the single embedding pass shared by ``FindingDory`` (on the Finding
    Dory button) and the background pre-warm kicked off after stitching. It is
    Qt-free so it can run on a worker thread owned by either caller.

    Honors ``cfg['dev_mock_embeddings']``: generates well-separated synthetic
    clusters per channel and skips the model entirely, useful for testing the pipeline
    without the overhead of loading the model and running the forward pass.

    Args:
        cfg: Parsed labeller config.
        mode: Model bundle key into ``cfg['models']`` (see ``resolve_mode``).
        channels: Channel names to embed, in order.
        mosaics: Channel-name -> full uint16 mosaic. Ignored in mock mode.
        well_centers: (N, 2) well centers in mosaic pixels, ordered (y, x).
        well_crop_px: (h, w) per-well crop size drawn from the mosaic.
        n_total: Total well count — used to default ``keep_indices`` in mock
            mode (the real path defaults it from ``well_centers``).
        keep_indices: Optional indices into ``well_centers`` to embed; ``None``
            embeds all wells.
        progress_cb: Optional callback(step, total), one step per channel.
        status_cb: Optional callback(message) for coarse phase updates.
        pixel_size_um: This plate's physical scale (um/px); forwarded to
            ``extract_from_mosaic`` for physical-scale resampling. Ignored in
            mock mode.

    Returns:
        (extractor_or_None, per_channel_embeddings, per_channel_indices).
        The extractor is ``None`` in mock mode.
    """
    mock = bool(cfg.get("dev_mock_embeddings", False))

    if mock:
        if status_cb is not None:
            status_cb("Mock embeddings (dev mode)…")
        keep = (
            keep_indices if keep_indices is not None
            else np.arange(n_total, dtype=np.int64)
        )
        model_cfg = cfg["models"][mode]
        emb_dim = 2 * int(model_cfg.get("embedding_dim", 384))
        n_clusters = 6
        n_wells = len(keep)
        embeds: Dict[str, np.ndarray] = {}
        idx: Dict[str, np.ndarray] = {}
        for ch_idx, ch in enumerate(channels):
            rng = np.random.default_rng(ch_idx + 1)
            centers = rng.standard_normal((n_clusters, emb_dim)).astype(np.float32) * 15.0
            assignments = rng.integers(0, n_clusters, size=n_wells)
            noise = rng.standard_normal((n_wells, emb_dim)).astype(np.float32) * 0.3
            embeds[ch] = centers[assignments] + noise
            idx[ch] = np.asarray(keep, dtype=np.int64)
        return None, embeds, idx

    if status_cb is not None:
        status_cb("Loading checkpoint…")
    extractor = EmbeddingExtractor(cfg, mode=mode)
    if status_cb is not None:
        status_cb("Computing embeddings…")

    embeds, idx = extractor.extract_from_mosaic(
        mosaics=mosaics,
        well_centers_px=well_centers,
        well_crop_px=well_crop_px,
        well_indices_to_embed=keep_indices,
        progress_cb=progress_cb,
        pixel_size_um=pixel_size_um,
    )
    return extractor, embeds, idx
