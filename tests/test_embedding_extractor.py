"""Tests for the vendored inference module.

Covers:
- Histogram-driven percentile computation matches `np.percentile`
- `apply_normalization` produces float32 in [0, 1] with expected curve
- `_crop_wells_uint16` and `_center_crop` handle in-bounds and out-of-bounds
- `EmbeddingExtractor.extract_from_mosaic` end-to-end with a stubbed backbone
  (so we don't depend on a real DINOv3 repo or `.ckpt` file)
"""

import json
import numpy as np
import pytest
import torch
import torch.nn as nn

from fish_sorter.helpers.embedding import extractor as ex_mod
from fish_sorter.helpers.embedding.extractor import (
    EmbeddingExtractor,
    _center_crop,
    _crop_wells_uint16,
    _resolve_device,
    compute_embeddings,
    load_config,
    resolve_mode,
)
from fish_sorter.helpers.embedding.normalize import (
    ChannelContrastConfig,
    apply_normalization,
    compute_channel_stats,
)


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------


def test_compute_channel_stats_matches_numpy_percentile_for_linear():
    rng = np.random.default_rng(0)
    mosaic = rng.integers(0, 65535, size=(512, 512), dtype=np.uint16)
    cfg = ChannelContrastConfig(
        low_percentile=0.1, high_percentile=99.9, asinh_knee=0.0, adaptive_high=False,
    )
    low, high = compute_channel_stats(mosaic, cfg)

    # np.percentile interpolates; for uint16 inputs the histogram lookup gives
    # the value at the next integer above the target rank, so tolerance is ±1.
    np_low = np.percentile(mosaic, 0.1)
    np_high = np.percentile(mosaic, 99.9)
    assert abs(low - np_low) <= 1.0
    assert abs(high - np_high) <= 1.0


def test_apply_normalization_produces_float32_in_unit_range():
    arr = np.arange(0, 1000, dtype=np.uint16).reshape(50, 20)
    out = apply_normalization(arr, low=100.0, high=900.0, asinh_knee=0.0)
    assert out.dtype == np.float32
    assert out.min() == 0.0
    assert out.max() == 1.0
    # midpoint check: pixel = 500 → (500-100)/800 = 0.5
    assert abs(out.flat[500] - 0.5) < 1e-6


def test_apply_normalization_with_asinh_lifts_dim_end():
    arr = np.array([[0, 25, 100]], dtype=np.uint16)
    linear = apply_normalization(arr, low=0.0, high=100.0, asinh_knee=0.0)
    asinh = apply_normalization(arr, low=0.0, high=100.0, asinh_knee=5.0)
    assert linear[0, 1] == pytest.approx(0.25)
    # asinh(knee*x)/asinh(knee) with knee=5 and x=0.25 ≈ 0.4530.
    # Endpoint values stay anchored at 0 and 1; only the dim-end is lifted.
    assert asinh[0, 1] > linear[0, 1], "asinh should lift the 0.25 input above its linear value"
    assert asinh[0, 0] == 0.0
    assert asinh[0, 2] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Crop helpers
# ---------------------------------------------------------------------------


def test_crop_wells_uint16_in_bounds():
    mosaic = np.arange(100 * 100, dtype=np.uint16).reshape(100, 100)
    centers = np.array([[50, 50], [25, 25]])
    crops = _crop_wells_uint16(mosaic, centers, h=10, w=10)
    assert crops.shape == (2, 10, 10)
    # Centered crop at (50, 50) of size 10×10 means y in [45, 55), x in [45, 55).
    assert crops[0, 0, 0] == mosaic[45, 45]
    assert crops[0, 9, 9] == mosaic[54, 54]


def test_crop_wells_uint16_zero_pads_out_of_bounds():
    mosaic = np.ones((50, 50), dtype=np.uint16) * 1000
    # Center near top-left corner — crop extends off the mosaic.
    centers = np.array([[2, 2]])
    crops = _crop_wells_uint16(mosaic, centers, h=10, w=10)
    assert crops.shape == (1, 10, 10)
    # Top-left pixels should be zero (off-mosaic); bottom-right inside mosaic.
    assert crops[0, 0, 0] == 0
    assert crops[0, 9, 9] == 1000


def test_center_crop_exact_match_is_passthrough():
    buf = np.arange(2 * 8 * 8, dtype=np.float32).reshape(2, 8, 8)
    out = _center_crop(buf, target_h=8, target_w=8)
    assert out is buf or np.array_equal(out, buf)


def test_center_crop_smaller_target_crops_center():
    buf = np.arange(1 * 8 * 8, dtype=np.float32).reshape(1, 8, 8)
    out = _center_crop(buf, target_h=4, target_w=4)
    assert out.shape == (1, 4, 4)
    # Center of an 8×8 is the (2..6, 2..6) slice; pixel (2, 2) → buf[0, 2, 2]
    assert out[0, 0, 0] == buf[0, 2, 2]


def test_center_crop_larger_target_pads():
    buf = np.full((1, 4, 4), 7.0, dtype=np.float32)
    out = _center_crop(buf, target_h=8, target_w=8)
    assert out.shape == (1, 8, 8)
    # Center should still be 7; edges should be 0.
    assert out[0, 4, 4] == 7.0
    assert out[0, 0, 0] == 0.0


# ---------------------------------------------------------------------------
# Device resolution
# ---------------------------------------------------------------------------


def test_resolve_device_cpu_is_always_available():
    assert _resolve_device("cpu").type == "cpu"


def test_resolve_device_cuda_raises_when_unavailable(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="CUDA is not available"):
        _resolve_device("cuda")


def test_resolve_device_mps_raises_when_unavailable(monkeypatch):
    # Force mps unavailable regardless of host
    fake_mps = type("F", (), {"is_available": staticmethod(lambda: False)})()
    monkeypatch.setattr(torch.backends, "mps", fake_mps, raising=False)
    with pytest.raises(RuntimeError, match="MPS is not available"):
        _resolve_device("mps")


def test_resolve_device_auto_falls_back_cleanly(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    fake_mps = type("F", (), {"is_available": staticmethod(lambda: False)})()
    monkeypatch.setattr(torch.backends, "mps", fake_mps, raising=False)
    assert _resolve_device("auto").type == "cpu"


# ---------------------------------------------------------------------------
# End-to-end EmbeddingExtractor with a stubbed backbone
# ---------------------------------------------------------------------------


class _StubBackbone(nn.Module):
    """Tiny stand-in for FishDINOv3 — returns mean of input as the embedding."""

    output_dim = 4

    def __init__(self, *args, **kwargs):
        super().__init__()
        self._dummy = nn.Linear(1, 1)  # gives load_state_dict something to bind

    def forward(self, x):  # (B, 1, H, W) → (B, 4)
        b = x.shape[0]
        flat = x.reshape(b, -1)
        return torch.stack(
            [flat.mean(dim=1), flat.std(dim=1), flat.min(dim=1).values, flat.max(dim=1).values],
            dim=1,
        )

    def get_embedding_dim(self):
        return self.output_dim


@pytest.fixture
def stub_extractor(monkeypatch, tmp_path):
    """Build an EmbeddingExtractor with stubbed backbone + fake ckpt."""
    monkeypatch.setattr(ex_mod, "FishDINOv3", lambda **kwargs: _StubBackbone())
    monkeypatch.setattr(ex_mod, "resolve_weights_path", lambda *args, **kwargs: None)

    # Touch a fake ckpt file so the path-exists check passes; the stub backbone
    # ignores its contents anyway via the no-op state_dict.
    fake_ckpt = tmp_path / "best.ckpt"
    torch.save({"state_dict": {}}, fake_ckpt)

    cfg = {
        "device": "cpu",
        "dinov3_repo_path": str(tmp_path),
        "dinov3_weights_dir": str(tmp_path),
        "clustering": {"method": "hdbscan", "params": {"min_cluster_size": 5}},
        "models": {
            "fish": {
                "checkpoint_path": str(fake_ckpt),
                "model_arch": "vits16",
                "crop_size": [32, 32],
                "resize_to": None,
                "contrast": {
                    "BF":     {"low_percentile": 0.1, "high_percentile": 99.9, "asinh_knee": 0.0, "adaptive_high": False},
                    "_FLUOR": {"low_percentile": 0.5, "high_percentile": 99.97, "asinh_knee": 5.0, "adaptive_high": True, "high_gate_percentile": 99.5, "high_trim_percentile": 99.99},
                },
            }
        },
    }
    return EmbeddingExtractor(cfg, mode="fish", batch_size=2)


def test_extract_from_mosaic_shapes(stub_extractor):
    rng = np.random.default_rng(42)
    n_wells = 5
    mosaics = {
        "BF": rng.integers(0, 65535, size=(200, 200), dtype=np.uint16),
        "GFP": rng.integers(0, 65535, size=(200, 200), dtype=np.uint16),
    }
    centers = np.array([[50, 50], [100, 100], [150, 150], [60, 140], [120, 70]])
    embeds, indices = stub_extractor.extract_from_mosaic(
        mosaics=mosaics,
        well_centers_px=centers,
        well_crop_px=(40, 40),
    )
    assert set(embeds.keys()) == {"BF", "GFP"}
    assert embeds["BF"].shape == (n_wells, 4)
    assert embeds["GFP"].shape == (n_wells, 4)
    assert np.array_equal(indices["BF"], np.arange(n_wells))


def test_extract_from_mosaic_subsets_via_well_indices(stub_extractor):
    rng = np.random.default_rng(7)
    mosaics = {"BF": rng.integers(0, 65535, size=(200, 200), dtype=np.uint16)}
    centers = np.array([[50, 50], [100, 100], [150, 150]])
    embeds, indices = stub_extractor.extract_from_mosaic(
        mosaics=mosaics,
        well_centers_px=centers,
        well_crop_px=(40, 40),
        well_indices_to_embed=np.array([0, 2]),
    )
    assert embeds["BF"].shape == (2, 4)
    assert np.array_equal(indices["BF"], np.array([0, 2]))


def test_extract_from_mosaic_emits_progress(stub_extractor):
    rng = np.random.default_rng(13)
    mosaics = {
        "BF": rng.integers(0, 65535, size=(100, 100), dtype=np.uint16),
        "GFP": rng.integers(0, 65535, size=(100, 100), dtype=np.uint16),
        "TXR": rng.integers(0, 65535, size=(100, 100), dtype=np.uint16),
    }
    centers = np.array([[50, 50]])
    seen = []
    stub_extractor.extract_from_mosaic(
        mosaics=mosaics,
        well_centers_px=centers,
        well_crop_px=(40, 40),
        progress_cb=lambda i, n: seen.append((i, n)),
    )
    assert seen == [(1, 3), (2, 3), (3, 3)]


def test_extract_from_mosaic_rejects_non_uint16(stub_extractor):
    mosaics = {"BF": np.zeros((100, 100), dtype=np.float32)}
    centers = np.array([[50, 50]])
    with pytest.raises(TypeError, match="uint16"):
        stub_extractor.extract_from_mosaic(
            mosaics=mosaics,
            well_centers_px=centers,
            well_crop_px=(40, 40),
        )


# ---------------------------------------------------------------------------
# Config loader
# ---------------------------------------------------------------------------


def test_load_config_round_trip(tmp_path):
    cfg = {
        "device": "cpu",
        "models": {"fish": {"crop_size": [32, 32], "contrast": {"_FLUOR": {"low_percentile": 0.5, "high_percentile": 99.0, "asinh_knee": 0.0}}}},
    }
    path = tmp_path / "config.json"
    path.write_text(json.dumps(cfg))
    loaded = load_config(path)
    assert loaded == cfg


def test_load_config_missing_models_key_raises(tmp_path):
    path = tmp_path / "broken.json"
    path.write_text(json.dumps({"device": "cpu"}))
    with pytest.raises(ValueError, match="'models'"):
        load_config(path)


# ---------------------------------------------------------------------------
# resolve_mode
# ---------------------------------------------------------------------------


def test_resolve_mode_maps_known_pick_type():
    cfg = {"pick_type_to_mode": {"larvae": "fish", "embryo": "egg"}}
    assert resolve_mode(cfg, "larvae") == "fish"
    assert resolve_mode(cfg, "embryo") == "egg"


def test_resolve_mode_unknown_falls_back_to_default():
    cfg = {"pick_type_to_mode": {"larvae": "fish"}}
    assert resolve_mode(cfg, "embryo") == "fish"
    assert resolve_mode(cfg, "embryo", default="egg") == "egg"


def test_resolve_mode_missing_block_falls_back():
    assert resolve_mode({}, "larvae") == "fish"


# ---------------------------------------------------------------------------
# compute_embeddings — shared one-shot pass (mock + real via stub backbone)
# ---------------------------------------------------------------------------


def test_compute_embeddings_mock_skips_model_and_shapes():
    """Mock mode returns synthetic per-channel embeddings without a model."""
    cfg = {
        "dev_mock_embeddings": True,
        "models": {"fish": {"embedding_dim": 8}},
    }
    extractor, embeds, idx = compute_embeddings(
        cfg,
        "fish",
        channels=["BF", "GFP"],
        mosaics={},  # ignored in mock mode
        well_centers=np.zeros((5, 2)),
        well_crop_px=(40, 40),
        n_total=5,
    )
    assert extractor is None
    assert set(embeds.keys()) == {"BF", "GFP"}
    assert embeds["BF"].shape == (5, 16)  # 2 * embedding_dim
    assert np.array_equal(idx["GFP"], np.arange(5))


def test_compute_embeddings_mock_honors_keep_indices():
    cfg = {"dev_mock_embeddings": True, "models": {"fish": {"embedding_dim": 4}}}
    _, embeds, idx = compute_embeddings(
        cfg,
        "fish",
        channels=["BF"],
        mosaics={},
        well_centers=np.zeros((6, 2)),
        well_crop_px=(40, 40),
        n_total=6,
        keep_indices=np.array([1, 3, 5]),
    )
    assert embeds["BF"].shape == (3, 8)
    assert np.array_equal(idx["BF"], np.array([1, 3, 5]))


def test_compute_embeddings_real_path_runs_extractor(monkeypatch, tmp_path):
    """Non-mock path builds an EmbeddingExtractor and embeds the mosaics."""
    monkeypatch.setattr(ex_mod, "FishDINOv3", lambda **kwargs: _StubBackbone())
    monkeypatch.setattr(ex_mod, "resolve_weights_path", lambda *a, **k: None)
    fake_ckpt = tmp_path / "best.ckpt"
    torch.save({"state_dict": {}}, fake_ckpt)
    cfg = {
        "device": "cpu",
        "dinov3_repo_path": str(tmp_path),
        "models": {
            "fish": {
                "checkpoint_path": str(fake_ckpt),
                "model_arch": "vits16",
                "crop_size": [32, 32],
                "contrast": {
                    "_FLUOR": {"low_percentile": 0.5, "high_percentile": 99.97, "asinh_knee": 0.0},
                },
            }
        },
    }
    mosaics = {"BF": np.random.default_rng(0).integers(0, 65535, (100, 100), dtype=np.uint16)}
    centers = np.array([[50, 50], [60, 60]])
    statuses = []
    extractor, embeds, idx = compute_embeddings(
        cfg,
        "fish",
        channels=["BF"],
        mosaics=mosaics,
        well_centers=centers,
        well_crop_px=(40, 40),
        n_total=2,
        status_cb=statuses.append,
    )
    assert isinstance(extractor, EmbeddingExtractor)
    assert embeds["BF"].shape == (2, _StubBackbone.output_dim)
    assert np.array_equal(idx["BF"], np.arange(2))
    assert "Loading checkpoint…" in statuses and "Computing embeddings…" in statuses
