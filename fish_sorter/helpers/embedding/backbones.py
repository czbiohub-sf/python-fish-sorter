"""DINOv3 inference backbone (trimmed vendor of zebra-repo `models/backbones.py`).

Only the forward path needed to produce embeddings is kept: `GeMPooling`,
`ImageNetNormalize`, and `FishDINOv3` (gem-pooling variant). Training-only
machinery (BYOL projector heads, freeze helpers, alternative pooling modes,
`forward_dense`, env-var path defaults) is intentionally removed — paths come
from the labeller config, never from the environment.
"""

import logging
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn

from .channel_adapter import UnifiedChannelAdapter

log = logging.getLogger(__name__)


# Default weights filenames per variant. Looked up under `weights_dir`.
DINOV3_LOCAL_WEIGHTS = {
    "vits16": "dinov3_vits16.pth",
    "vits16plus": "dinov3_vits16plus.pth",
    "vitb16": "dinov3_vitb16.pth",
    "vitl16": "dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth",
    "convnext_tiny": "dinov3_convnext_tiny.pth",
    "convnext_small": "dinov3_convnext_small.pth",
}

# Output dim per DINOv3 variant.
EMBEDDING_DIMS = {
    "vits16": 384,
    "vits16plus": 384,
    "vitb16": 768,
    "vitl16": 1024,
    "vitl16plus": 1024,
    "vithplus": 1280,
    "vit7b": 4096,
    "convnext_tiny": 768,
    "convnext_small": 768,
    "convnext_base": 1024,
    "convnext_large": 1536,
}


class GeMPooling(nn.Module):
    """Generalized Mean (GeM) pooling over patch tokens.

    Applied separately to positive and negative parts then recombined, since
    DINOv3 patch tokens come from layer normalization (zero-centered).
    """

    def __init__(self, p: float = 3.0, eps: float = 1e-6):
        super().__init__()
        self.p = nn.Parameter(torch.tensor(p))
        self.eps = eps

    def forward(self, patch_tokens: torch.Tensor) -> torch.Tensor:
        p = self.p
        pos = patch_tokens.clamp(min=0)
        neg = (-patch_tokens).clamp(min=0)
        pos_pool = pos.clamp(min=self.eps).pow(p).mean(dim=1).pow(1.0 / p)
        neg_pool = neg.clamp(min=self.eps).pow(p).mean(dim=1).pow(1.0 / p)
        return pos_pool - neg_pool


class ImageNetNormalize(nn.Module):
    """ImageNet normalization for pretrained models."""

    def __init__(self):
        super().__init__()
        self.register_buffer(
            "mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean) / self.std


def _normalize_variant(model_arch: str) -> str:
    """Accept either bare variant ('vits16') or hub name ('dinov3_vits16')."""
    return model_arch[len("dinov3_"):] if model_arch.startswith("dinov3_") else model_arch


class FishDINOv3(nn.Module):
    """DINOv3 backbone for inference on zebrafish well crops.

    Loads a DINOv3 ViT/ConvNeXt via local `torch.hub` and routes single- or
    multi-channel crops through `UnifiedChannelAdapter` → ImageNet normalize →
    backbone. Pooling concatenates the CLS token with GeM-pooled patch tokens.
    """

    def __init__(
        self,
        variant: str = "vits16",
        in_channels: int = 1,
        repo_path: Optional[str] = None,
        weights_path: Optional[str] = None,
        multi_contrast: bool = False,
    ):
        super().__init__()
        variant = _normalize_variant(variant)
        if variant not in EMBEDDING_DIMS:
            raise ValueError(
                f"Unknown variant: {variant}. Choose from {sorted(EMBEDDING_DIMS)}"
            )
        self.variant = variant
        self.in_channels = in_channels
        self.embed_dim = EMBEDDING_DIMS[variant]

        self.channel_adapter = UnifiedChannelAdapter(in_channels, multi_contrast=multi_contrast)
        self.normalize = ImageNetNormalize()
        self.backbone = self._load_backbone(variant, repo_path, weights_path)
        self.patch_pooler = GeMPooling(p=3.0)
        self.output_dim = 2 * self.embed_dim  # CLS + GeM-pooled patches

    @staticmethod
    def _load_backbone(
        variant: str,
        repo_path: Optional[str],
        weights_path: Optional[str],
    ) -> nn.Module:
        """Load a DINOv3 backbone from a local hub clone, optionally applying weights."""
        if repo_path is None:
            raise ValueError(
                "repo_path is required (set dinov3_repo_path in the labeller config)"
            )
        if not Path(repo_path).exists():
            raise FileNotFoundError(f"DINOv3 repo not found at {repo_path}")

        model_name = f"dinov3_{variant}"
        log.info(f"Loading DINOv3 {variant} from local repo: {repo_path}")

        backbone = torch.hub.load(
            repo_path,
            model_name,
            source="local",
            pretrained=False,
        )
        if weights_path is not None:
            wp = Path(weights_path)
            if not wp.exists():
                raise FileNotFoundError(f"DINOv3 weights not found at {weights_path}")
            log.info(f"Applying local DINOv3 weights: {weights_path}")
            state_dict = torch.load(str(weights_path), map_location="cpu")
            backbone.load_state_dict(state_dict)
        return backbone

    def _get_cls_and_patch_tokens(self, x: torch.Tensor):
        feat = self.backbone.forward_features(x)
        if isinstance(feat, dict):
            return feat["x_norm_clstoken"], feat["x_norm_patchtokens"]
        # Fallback for non-dict returns
        return feat[:, 0], feat[:, 1:]

    def patch_grid_for(self, crop_h: int, crop_w: int) -> tuple:
        ps = 16  # DINOv3 ViT patch size
        if crop_h % ps or crop_w % ps:
            raise ValueError(
                f"Crop size ({crop_h}, {crop_w}) is not divisible by patch_size={ps}"
            )
        return (crop_h // ps, crop_w // ps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.channel_adapter(x)
        x = self.normalize(x)
        cls, patches = self._get_cls_and_patch_tokens(x)
        return torch.cat([cls, self.patch_pooler(patches)], dim=1)

    def get_embedding_dim(self) -> int:
        return self.output_dim

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device


def resolve_weights_path(weights_dir: str, variant: str) -> Optional[str]:
    """Return the local `.pth` for `variant` under `weights_dir`, or None if missing."""
    variant = _normalize_variant(variant)
    fname = DINOV3_LOCAL_WEIGHTS.get(variant)
    if fname is None:
        return None
    path = Path(weights_dir) / fname
    return str(path) if path.exists() else None
