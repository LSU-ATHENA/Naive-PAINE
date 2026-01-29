"""
Text encoder variants for PixArt Score Predictor.

Four variants:
1. PerTokenScalarTextEncoder - Linear 4096→1 per token, output [B, 120]
2. PoolingTextEncoder - Global pool over tokens, output [B, 4096]
3. CompressionFirstTextEncoder - Compress→Attn→FFN→Pool, output [B, 512]
4. CompressionLaterTextEncoder - Attn→FFN→Compress→Pool, output [B, 512]
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class PerTokenScalarTextEncoder(nn.Module):
    """
    Per-Token Scalar Encoder.

    Projects each token's 4096-dim embedding down to a single scalar,
    resulting in one score per token position.

    [B, 120, 4096] -> Linear layers (4096→2048→1024→512→256→128→1) -> [B, 120, 1] -> squeeze -> [B, 120]
    """

    def __init__(self, embed_dim: int = 4096):
        super().__init__()

        # Progressive projection: 4096 -> 2048 -> 1024 -> 512 -> 256 -> 128 -> 1
        self.project = nn.Sequential(
            nn.Linear(embed_dim, 2048),
            nn.SiLU(),
            nn.Linear(2048, 1024),
            nn.SiLU(),
            nn.Linear(1024, 512),
            nn.SiLU(),
            nn.Linear(512, 256),
            nn.SiLU(),
            nn.Linear(256, 128),
            nn.SiLU(),
            nn.Linear(128, 1),
        )

        self._output_dim = 120  # Token count

    @property
    def output_dim(self) -> int:
        return self._output_dim

    def forward(
        self,
        prompt_embeds: torch.Tensor,
        prompt_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            prompt_embeds: [B, 120, 4096]
            prompt_mask: [B, 120] - 1 for valid tokens, 0 for padding
        Returns:
            [B, 120]
        """
        # Project embedding dimension: [B, 120, 4096] -> [B, 120, 1]
        x = self.project(prompt_embeds)

        # Squeeze last dimension: [B, 120, 1] -> [B, 120]
        x = x.squeeze(-1)

        # Apply mask (zero out padding positions)
        x = x * prompt_mask

        return x


class PoolingTextEncoder(nn.Module):
    """
    Idea 2: "The 4096 Match"

    Remove the sequence dimension (120) via global pooling, keep the full embedding (4096).

    [B, 120, 4096] -> masked mean pool -> [B, 4096]
    """

    def __init__(self, embed_dim: int = 4096):
        super().__init__()
        self._output_dim = embed_dim

    @property
    def output_dim(self) -> int:
        return self._output_dim

    def forward(
        self,
        prompt_embeds: torch.Tensor,
        prompt_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            prompt_embeds: [B, 120, 4096]
            prompt_mask: [B, 120] - 1 for valid tokens, 0 for padding
        Returns:
            [B, 4096]
        """
        # Expand mask: [B, 120] -> [B, 120, 1]
        mask = prompt_mask.unsqueeze(-1)

        # Masked sum: [B, 120, 4096] * [B, 120, 1] -> sum -> [B, 4096]
        x_sum = (prompt_embeds * mask).sum(dim=1)

        # Count valid tokens: [B, 1]
        mask_sum = mask.sum(dim=1).clamp(min=1e-8)

        # Mean: [B, 4096]
        pooled = x_sum / mask_sum

        return pooled


class CompressionFirstTextEncoder(nn.Module):
    """
    Latest Idea A: "Compression First" (Efficient)

    4096 is too big for a transformer. Shrink it to 512 before running attention.

    Steps:
        1. Linear 4096→512 -> [B, 120, 512]
        2. Self-Attention on 512 dim -> [B, 120, 512]
        3. FFN (512→2048→512 w/ SiLU) -> [B, 120, 512]
        4. Global pooling (collapse 120→1) -> [B, 512]
    """

    def __init__(
        self,
        embed_dim: int = 4096,
        hidden_dim: int = 512,
        num_heads: int = 8,
        ffn_mult: int = 4,
    ):
        super().__init__()

        # Step 1: Linear compression
        self.compress = nn.Linear(embed_dim, hidden_dim)

        # Step 2: Self-Attention
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            batch_first=True,
        )

        # Step 3: FFN
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * ffn_mult),
            nn.SiLU(),
            nn.Linear(hidden_dim * ffn_mult, hidden_dim),
        )

        self._output_dim = hidden_dim

    @property
    def output_dim(self) -> int:
        return self._output_dim

    def forward(
        self,
        prompt_embeds: torch.Tensor,
        prompt_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            prompt_embeds: [B, 120, 4096]
            prompt_mask: [B, 120] - 1 for valid tokens, 0 for padding
        Returns:
            [B, 512]
        """
        # Step 1: Compress embedding dim
        x = self.compress(prompt_embeds)  # [B, 120, 512]

        # Create attention mask (True = ignore)
        attn_mask = (prompt_mask == 0)  # [B, 120]

        # Step 2: Self-Attention with residual
        x_norm = self.norm1(x)
        attn_out, _ = self.attn(
            x_norm, x_norm, x_norm,
            key_padding_mask=attn_mask,
            need_weights=False,
        )
        x = x + attn_out

        # Step 3: FFN with residual
        x = x + self.ffn(self.norm2(x))

        # Step 4: Global pooling (masked mean)
        mask = prompt_mask.unsqueeze(-1)  # [B, 120, 1]
        x_sum = (x * mask).sum(dim=1)  # [B, 512]
        mask_sum = mask.sum(dim=1).clamp(min=1e-8)  # [B, 1]
        pooled = x_sum / mask_sum  # [B, 512]

        return pooled


class CompressionLaterTextEncoder(nn.Module):
    """
    Latest Idea B: "Compression Later" (High Quality)

    Don't lose information early. Run Attention on the full 4096 vector to capture
    maximum nuance, then compress.

    Steps:
        1. Self-Attention on full 4096 dim -> [B, 120, 4096]
        2. FFN (4096→16384→4096 w/ SiLU) -> [B, 120, 4096]
        3. Linear 4096→512 -> [B, 120, 512]
        4. Global pooling (collapse 120→1) -> [B, 512]
    """

    def __init__(
        self,
        embed_dim: int = 4096,
        output_dim: int = 512,
        num_heads: int = 16,
        ffn_mult: int = 4,
    ):
        super().__init__()

        # Step 1: Self-Attention on full dim
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            batch_first=True,
        )

        # Step 2: FFN
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * ffn_mult),
            nn.SiLU(),
            nn.Linear(embed_dim * ffn_mult, embed_dim),
        )

        # Step 3: Compression
        self.compress = nn.Linear(embed_dim, output_dim)

        self._output_dim = output_dim

    @property
    def output_dim(self) -> int:
        return self._output_dim

    def forward(
        self,
        prompt_embeds: torch.Tensor,
        prompt_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            prompt_embeds: [B, 120, 4096]
            prompt_mask: [B, 120] - 1 for valid tokens, 0 for padding
        Returns:
            [B, 512]
        """
        x = prompt_embeds

        # Create attention mask (True = ignore)
        attn_mask = (prompt_mask == 0)  # [B, 120]

        # Step 1: Self-Attention with residual
        x_norm = self.norm1(x)
        attn_out, _ = self.attn(
            x_norm, x_norm, x_norm,
            key_padding_mask=attn_mask,
            need_weights=False,
        )
        x = x + attn_out

        # Step 2: FFN with residual
        x = x + self.ffn(self.norm2(x))

        # Step 3: Compress
        x = self.compress(x)  # [B, 120, 512]

        # Step 4: Global pooling (masked mean)
        mask = prompt_mask.unsqueeze(-1)  # [B, 120, 1]
        x_sum = (x * mask).sum(dim=1)  # [B, 512]
        mask_sum = mask.sum(dim=1).clamp(min=1e-8)  # [B, 1]
        pooled = x_sum / mask_sum  # [B, 512]

        return pooled


def get_text_encoder(name: str, **kwargs) -> nn.Module:
    """Factory function to create text encoder by name."""
    encoders = {
        'pertokenscalar': PerTokenScalarTextEncoder,
        'pooling': PoolingTextEncoder,
        'compression_first': CompressionFirstTextEncoder,
        'compression_later': CompressionLaterTextEncoder,
    }

    if name not in encoders:
        raise ValueError(f"Unknown text encoder: {name}. Available: {list(encoders.keys())}")

    return encoders[name](**kwargs)
