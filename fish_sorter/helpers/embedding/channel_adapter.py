"""Unified channel adapter for DINOv3-style backbones.

Vendored from `zebrafish-unsupervised-classification/fish_classify/models/channel_adapter.py`.
Always outputs 3 channels to the backbone by zero-padding (or multi-contrast
mapping when explicitly enabled for single-channel inputs).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class UnifiedChannelAdapter(nn.Module):
    """Always outputs 3 channels to DINOv3.

    Two modes for single-channel input:
    - zero-pad (default): [x, 0, 0]
    - multi-contrast: [linear, high-pass, bright-only]

    For 2 or 3 channel input, always zero-pads.
    """

    def __init__(
        self,
        in_channels: int,
        multi_contrast: bool = False,
        highpass_sigma: float = 10.0,
        bright_gamma: float = 5.0,
    ):
        super().__init__()
        if in_channels not in (1, 2, 3):
            raise ValueError(
                f"UnifiedChannelAdapter only supports 1-3 input channels, got {in_channels}"
            )
        self.in_channels = in_channels
        self.out_channels = 3
        self.multi_contrast = multi_contrast and in_channels == 1
        self.bright_gamma = bright_gamma

        if self.multi_contrast:
            self.highpass_sigma = highpass_sigma
            kernel_size = int(6 * highpass_sigma) | 1
            self.register_buffer(
                "_blur_kernel", self._make_gaussian_kernel(kernel_size, highpass_sigma)
            )

    @staticmethod
    def _make_gaussian_kernel(size: int, sigma: float) -> torch.Tensor:
        coords = torch.arange(size, dtype=torch.float32) - size // 2
        g = torch.exp(-0.5 * (coords / sigma) ** 2)
        kernel = g.outer(g)
        kernel = kernel / kernel.sum()
        return kernel.view(1, 1, size, size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[1] != self.in_channels:
            raise ValueError(
                f"Expected {self.in_channels} input channels, got {x.shape[1]}"
            )

        if self.in_channels == 3:
            return x

        if self.multi_contrast:
            ch_r = x

            pad = self._blur_kernel.shape[-1] // 2
            blurred = F.conv2d(x, self._blur_kernel, padding=pad)
            highpass = x - blurred
            hp_flat = highpass.view(highpass.shape[0], -1)
            hp_min = hp_flat.min(dim=1).values.view(-1, 1, 1, 1)
            hp_max = hp_flat.max(dim=1).values.view(-1, 1, 1, 1)
            ch_g = (highpass - hp_min) / (hp_max - hp_min + 1e-8)

            ch_b = x.clamp(min=1e-6).pow(self.bright_gamma)

            return torch.cat([ch_r, ch_g, ch_b], dim=1)

        batch_size, _, height, width = x.shape
        padding = torch.zeros(
            batch_size,
            3 - self.in_channels,
            height,
            width,
            dtype=x.dtype,
            device=x.device,
        )
        return torch.cat([x, padding], dim=1)

    def extra_repr(self) -> str:
        s = f"in_channels={self.in_channels}, out_channels={self.out_channels}"
        if self.multi_contrast:
            s += f", multi_contrast(highpass_sigma={self.highpass_sigma}, bright_gamma={self.bright_gamma})"
        return s
