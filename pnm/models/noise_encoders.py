"""
Noise encoder variants for PixArt Score Predictor.

Six variants:
1. SpatialShrinkNoiseEncoder - Idea 1: Conv downsample 64→32→16, output [B, 1024]
2. ChannelSquashNoiseEncoder - Idea 2: 1x1 conv 4→1, flatten, output [B, 4096]
3. BasicCNNAvgPoolNoiseEncoder - Basic CNN + GlobalAvgPool, output [B, 256]
4. BasicCNNMaxPoolNoiseEncoder - Basic CNN + GlobalMaxPool, output [B, 256]
5. ResNetAvgPoolNoiseEncoder - ResNet with AvgPool skip connections, output [B, 256]
6. ResNetMaxPoolNoiseEncoder - ResNet with MaxPool skip connections, output [B, 256]
"""

from typing import Literal

import torch
import torch.nn as nn


class SpatialShrinkNoiseEncoder(nn.Module):
    """
    Idea 1: "The Aggressive Whittling" - Noise Branch (Spatial Shrink)

    Apply Convolution to reduce the spatial dimension (64x64), then flatten.

    [B, 4, 64, 64] -> Conv downsample (64→32→16) -> [B, 4, 16, 16] -> flatten -> [B, 1024]
    """

    def __init__(self, in_channels: int = 4):
        super().__init__()

        # Conv layers to downsample spatial: 64 -> 32 -> 16
        self.conv = nn.Sequential(
            # 64 -> 32
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.SiLU(),

            # 32 -> 16
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.SiLU(),
        )

        # Output: 4 * 16 * 16 = 1024
        self._output_dim = in_channels * 16 * 16

    @property
    def output_dim(self) -> int:
        return self._output_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, 4, 64, 64]
        Returns:
            [B, 1024]
        """
        x = self.conv(x)  # [B, 4, 16, 16]
        x = x.flatten(start_dim=1)  # [B, 1024]
        return x


class ChannelSquashNoiseEncoder(nn.Module):
    """
    Idea 2: "The 4096 Match" - Noise Branch (Channel Squash)

    Remove the channel dimension (4) via 1x1 convolution, keep the spatial area (flattened to 4096).

    [B, 4, 64, 64] -> 1x1 conv (4→1) -> [B, 1, 64, 64] -> flatten -> [B, 4096]
    """

    def __init__(self, in_channels: int = 4):
        super().__init__()

        # 1x1 convolution to squeeze channels: 4 -> 1
        self.channel_squeeze = nn.Sequential(
            nn.Conv2d(in_channels, 1, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(1),
            nn.SiLU(),
        )

        # Output: 1 * 64 * 64 = 4096
        self._output_dim = 64 * 64

    @property
    def output_dim(self) -> int:
        return self._output_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, 4, 64, 64]
        Returns:
            [B, 4096]
        """
        x = self.channel_squeeze(x)  # [B, 1, 64, 64]
        x = x.flatten(start_dim=1)  # [B, 4096]
        return x


class BasicCNNGlobalPoolNoiseEncoder(nn.Module):
    """
    Latest Idea: Basic CNN with Global Pooling

    Use a Basic CNN with Global Pooling (Avg or Max) to understand the texture of the noise.

    Steps:
        1. Conv blocks (no skip) downsample spatial, increase channels -> [B, 256, 8, 8]
        2. Global Pooling: AvgPool or MaxPool (8x8→1x1) -> [B, 256, 1, 1]
        3. Flatten -> [B, 256]
    """

    def __init__(self, in_channels: int = 4, out_channels: int = 256, pool_type: str = 'avg'):
        super().__init__()

        # Conv blocks: 64x64 -> 32x32 -> 16x16 -> 8x8
        self.conv = nn.Sequential(
            # 64 -> 32
            nn.Conv2d(in_channels, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.SiLU(),

            # 32 -> 16
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.SiLU(),

            # 16 -> 8
            nn.Conv2d(128, out_channels, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(),
        )

        # Global pooling
        if pool_type == 'avg':
            self.pool = nn.AdaptiveAvgPool2d((1, 1))
        elif pool_type == 'max':
            self.pool = nn.AdaptiveMaxPool2d((1, 1))
        else:
            raise ValueError(f"Unknown pool_type: {pool_type}. Use 'avg' or 'max'")

        self._output_dim = out_channels

    @property
    def output_dim(self) -> int:
        return self._output_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, 4, 64, 64]
        Returns:
            [B, 256]
        """
        x = self.conv(x)  # [B, 256, 8, 8]
        x = self.pool(x)  # [B, 256, 1, 1]
        x = x.flatten(start_dim=1)  # [B, 256]
        return x


class BasicCNNAvgPoolNoiseEncoder(BasicCNNGlobalPoolNoiseEncoder):
    """BasicCNN with Average Pooling."""

    def __init__(self, in_channels: int = 4, out_channels: int = 256):
        super().__init__(in_channels, out_channels, pool_type='avg')


class BasicCNNMaxPoolNoiseEncoder(BasicCNNGlobalPoolNoiseEncoder):
    """BasicCNN with Max Pooling."""

    def __init__(self, in_channels: int = 4, out_channels: int = 256):
        super().__init__(in_channels, out_channels, pool_type='max')

class ResidualDownsampleBlock(nn.Module):
    """
    Residual block with downsampling via strided convolution.

    Main path: Conv(s=2) → BN → SiLU → Conv(s=1) → BN
    Skip path: Pool2d(2) → 1x1 Conv (for channel matching)

    The skip uses pooling (Avg or Max) instead of strided conv for smoother
    gradient flow and different inductive bias.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        pool_type: Literal['avg', 'max'] = 'avg',
    ):
        super().__init__()

        # Main path: two convs with downsampling on first
        self.main = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
        )

        # Skip path: pooling for spatial downsample + 1x1 conv for channel match
        if pool_type == 'avg':
            self.skip_pool = nn.AvgPool2d(kernel_size=2, stride=2)
        else:
            self.skip_pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # 1x1 conv to match channels (always needed since we change channels)
        self.skip_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)

        self.activation = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        main_out = self.main(x)
        skip_out = self.skip_conv(self.skip_pool(x))
        return self.activation(main_out + skip_out)


class ResNetNoiseEncoder(nn.Module):
    """
    ResNet-style noise encoder with pooling-based skip connections.

    Architecture (from image):
        [B, 4, 64, 64]
            │
        Stem: Conv2d(4→32, k=3, p=1) + BN + SiLU
            │
        [B, 32, 64, 64]
            │
        ResidualDownsampleBlock(32→64, pool skip)   → [B, 64, 32, 32]
            │
        ResidualDownsampleBlock(64→128, pool skip)  → [B, 128, 16, 16]
            │
        ResidualDownsampleBlock(128→256, pool skip) → [B, 256, 8, 8]
            │
        ResidualDownsampleBlock(256→256, pool skip) → [B, 256, 4, 4]
            │
        Flatten → [B, 4096]
            │
        Linear(4096→256) + SiLU → [B, 256]

    Output: [B, 256]
    """

    def __init__(
        self,
        in_channels: int = 4,
        out_channels: int = 256,
        pool_type: Literal['avg', 'max'] = 'avg',
    ):
        super().__init__()

        # Stem: initial conv without downsampling
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.SiLU(),
        )

        # Residual blocks with downsampling
        # 64x64 → 32x32 → 16x16 → 8x8 → 4x4
        self.blocks = nn.Sequential(
            ResidualDownsampleBlock(32, 64, pool_type=pool_type),    # 64→32
            ResidualDownsampleBlock(64, 128, pool_type=pool_type),   # 32→16
            ResidualDownsampleBlock(128, 256, pool_type=pool_type),  # 16→8
            ResidualDownsampleBlock(256, 256, pool_type=pool_type),  # 8→4
        )

        # Final projection: flatten + linear
        # After blocks: [B, 256, 4, 4] = [B, 4096] flattened
        self.head = nn.Sequential(
            nn.Flatten(start_dim=1),
            nn.Linear(256 * 4 * 4, out_channels),
            nn.SiLU(),
        )

        self._output_dim = out_channels

    @property
    def output_dim(self) -> int:
        return self._output_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, 4, 64, 64]
        Returns:
            [B, 256]
        """
        x = self.stem(x)      # [B, 32, 64, 64]
        x = self.blocks(x)    # [B, 256, 4, 4]
        x = self.head(x)      # [B, 256]
        return x


class ResNetAvgPoolNoiseEncoder(ResNetNoiseEncoder):
    """ResNet noise encoder with AvgPool skip connections."""

    def __init__(self, in_channels: int = 4, out_channels: int = 256):
        super().__init__(in_channels, out_channels, pool_type='avg')


class ResNetMaxPoolNoiseEncoder(ResNetNoiseEncoder):
    """ResNet noise encoder with MaxPool skip connections."""

    def __init__(self, in_channels: int = 4, out_channels: int = 256):
        super().__init__(in_channels, out_channels, pool_type='max')


def get_noise_encoder(name: str, **kwargs) -> nn.Module:
    """Factory function to create noise encoder by name."""
    encoders = {
        'spatial_shrink': SpatialShrinkNoiseEncoder,
        'channel_squash': ChannelSquashNoiseEncoder,
        'basic_cnn_avg': BasicCNNAvgPoolNoiseEncoder,
        'basic_cnn_max': BasicCNNMaxPoolNoiseEncoder,
        'resnet_avgpool': ResNetAvgPoolNoiseEncoder,
        'resnet_maxpool': ResNetMaxPoolNoiseEncoder,
    }

    return encoders[name](**kwargs)
