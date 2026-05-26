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
import sys
from pathlib import Path

import numpy as np
from tifffile import imread

from fish_sorter.helpers.embedding.extractor import EmbeddingExtractor, load_config


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

    fs_cfg = load_config(Path(parity["fish_sorter_config"]))
    extractor = EmbeddingExtractor(fs_cfg, mode=parity["mode"])

    mosaics = {ch: imread(path) for ch, path in parity["mosaics"].items()}
    for ch, m in mosaics.items():
        if m.dtype != np.uint16:
            print(f"FAIL: channel {ch} mosaic is {m.dtype}, expected uint16", file=sys.stderr)
            return 1

    well_centers = np.load(parity["well_centers_npy"])
    well_crop_px = tuple(parity["well_crop_px"])

    ours, _ = extractor.extract_from_mosaic(
        mosaics=mosaics,
        well_centers_px=well_centers,
        well_crop_px=well_crop_px,
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
                f"[{ch}] FAIL: shape mismatch ours={ours_emb.shape} vs zebra={ref_emb.shape}. "
                "Same array JSON / well ordering?",
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
