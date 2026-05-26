"""Compare fish-sorter's EmbeddingExtractor against zebra's `fish-classify analyze`.

Run from the fish-sorter venv:

    uv run python scripts/parity_check.py --config /path/to/parity.json

`parity.json` schema:

    {
      "fish_sorter_config": "/path/to/fish_sorter/configs/labeller/config.json",
      "mode": "fish",
      "mosaics": {
        "BF":  "/path/to/plate/BF.tif",
        "GFP": "/path/to/plate/GFP.tif"
      },
      "well_centers_npy": "/tmp/parity_centers.npy",
      "well_crop_px": [416, 1808],
      "zebra_embeddings": {
        "BF":  "/path/to/zebra_out/<expt>/embeddings_BF.npz",
        "GFP": "/path/to/zebra_out/<expt>/embeddings_GFP.npz"
      },
      "threshold": 0.999
    }

To produce `well_centers_npy` (one-off, in the **zebra** venv):

    import numpy as np
    from fish_classify.data.well_loader import WellPlateMapping
    m = WellPlateMapping(array_file=..., pixel_size_um=...)
    m.set_calibration(um_TL=..., um_BR=...)
    wells = m.load_wells(grid_top_left=...)
    np.save("/tmp/parity_centers.npy", wells["actual_px"][:, ::-1])  # (y, x)

Exit code: 0 on pass, 1 on failure (worst-case cosine sim below threshold).
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import torch
from tifffile import imread

from fish_sorter.helpers.embedding.extractor import EmbeddingExtractor, load_config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stdout,
)
log = logging.getLogger("parity")


def _cosine(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    na = np.linalg.norm(a, axis=1)
    nb = np.linalg.norm(b, axis=1)
    return (a * b).sum(axis=1) / np.maximum(na * nb, 1e-12)


def _load_reference_embedding(path: str, ch: str):
    """Load a (N, D) reference embedding for `ch` from `.npy` or `.npz`.

    `.npy`: assumed to be the raw (N, D) array.
    `.npz`: looked up by key `emb_<ch>` first, falling back to the only key
            present if there's just one.
    """
    p = Path(path)
    if not p.exists():
        print(f"[{ch}] FAIL: reference file not found: {path}", file=sys.stderr)
        return None
    if p.suffix == ".npy":
        return np.load(path).astype(np.float32)
    with np.load(path) as ref:
        key = f"emb_{ch}"
        if key in ref.files:
            return ref[key].astype(np.float32)
        if len(ref.files) == 1:
            return ref[ref.files[0]].astype(np.float32)
        print(
            f"[{ch}] FAIL: {path} has no key {key!r}; available: {ref.files}",
            file=sys.stderr,
        )
        return None


def main() -> int:
    ap = argparse.ArgumentParser(description="fish-sorter vs zebra embedding parity")
    ap.add_argument("--config", required=True, help="Path to parity.json")
    args = ap.parse_args()

    with open(args.config) as f:
        parity = json.load(f)
    threshold = float(parity.get("threshold", 0.999))
    log.info(f"parity config: {args.config}")
    log.info(f"threshold: {threshold}")

    log.info(f"torch={torch.__version__} cuda_available={torch.cuda.is_available()}")
    if torch.cuda.is_available():
        log.info(f"cuda device: {torch.cuda.get_device_name(0)}")

    t0 = time.perf_counter()
    fs_cfg = load_config(Path(parity["fish_sorter_config"]))
    log.info(f"labeller config loaded from {parity['fish_sorter_config']}")
    extractor = EmbeddingExtractor(fs_cfg, mode=parity["mode"])
    log.info(
        f"EmbeddingExtractor constructed ({time.perf_counter()-t0:.2f}s); "
        f"device={extractor.device}, batch_size={extractor.batch_size}, "
        f"crop_size={extractor.crop_size}"
    )

    mosaics = {}
    for ch, path in parity["mosaics"].items():
        t0 = time.perf_counter()
        m = imread(path)
        log.info(
            f"loaded mosaic {ch} from {path} "
            f"shape={m.shape} dtype={m.dtype} ({time.perf_counter()-t0:.2f}s)"
        )
        if m.dtype != np.uint16:
            log.error(f"channel {ch} mosaic is {m.dtype}, expected uint16")
            return 1
        mosaics[ch] = m

    well_centers = np.load(parity["well_centers_npy"])
    well_crop_px = tuple(parity["well_crop_px"])

    # If zebra filtered out empty wells before embedding, its embedding array is
    # shorter than well_centers. Point this at the (M,) int array of indices
    # into well_centers that zebra kept (usually saved as idx_<ch> in zebra's
    # cache .npz, or `np.where(classification_csv['empty'] == 0)[0]`).
    indices_path = parity.get("well_indices_npy")
    well_indices_to_embed = None
    if indices_path:
        well_indices_to_embed = np.load(indices_path).astype(np.int64)
        print(
            f"Filtering to {len(well_indices_to_embed)} wells "
            f"(of {len(well_centers)} total) per {indices_path}"
        )

    ours, idx_by_ch = extractor.extract_from_mosaic(
        mosaics=mosaics,
        well_centers_px=well_centers,
        well_crop_px=well_crop_px,
        well_indices_to_embed=well_indices_to_embed,
    )

    any_fail = False
    for ch, ours_emb in ours.items():
        ref_path = parity.get("zebra_embeddings", {}).get(ch)
        if ref_path is None:
            print(f"[{ch}] skipped — no zebra reference configured")
            continue
        ref_emb = _load_reference_embedding(ref_path, ch)
        if ref_emb is None:
            any_fail = True
            continue
        if ref_emb.shape != ours_emb.shape:
            print(
                f"[{ch}] FAIL: shape mismatch ours={ours_emb.shape} vs zebra={ref_emb.shape}.\n"
                f"  If zebra filtered empty wells before embedding "
                f"(typical for `fish-classify analyze`), populate the "
                f"`well_indices_npy` field in parity.json with the indices zebra kept "
                f"so fish-sorter embeds the same subset in the same order.",
                file=sys.stderr,
            )
            any_fail = True
            continue
        sims = _cosine(ours_emb, ref_emb)
        worst_idx = int(sims.argmin())
        worst = float(sims[worst_idx])
        mean = float(sims.mean())
        status = "PASS" if worst >= threshold else "FAIL"
        print(f"[{ch}] {status}: mean={mean:.6f} worst={worst:.6f} (well idx {worst_idx})")
        if worst < threshold:
            any_fail = True

    if any_fail:
        print(
            f"\nAt least one channel fell below threshold {threshold}. "
            "Likely causes: ckpt prefix isn't 'online_network.backbone.'; "
            "contrast block doesn't match training values; "
            "well_crop_px doesn't match the array JSON slot dimensions.",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
