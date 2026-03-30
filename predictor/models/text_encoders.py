import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class AttnPool(nn.Module):

    def __init__(
        self,
        embed_dim: int = 4096,
        output_dim: int = 1024,
        ffn_ratio: float = 0.5,
        dropout: float = 0.1,
        seq_len: int = 120,
        pos_encoding: str = 'none',  
    ):
        super().__init__()

        self.embed_dim = embed_dim
        self.pos_encoding_type = pos_encoding

        num_heads = 16

        # Learnable summary token
        self.summary_token = nn.Parameter(torch.randn(1, 1, embed_dim))

        # Positional encoding
        if pos_encoding == 'sinusoidal':
            pe = self._build_sinusoidal_pe(seq_len + 1, embed_dim)
            self.register_buffer('pos_encoding', pe)  # [1, seq_len+1, embed_dim]

        # Attention block 1
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn1 = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            batch_first=True,
        )
        self.dropout1 = nn.Dropout(dropout)

        # FFN block 1
        ffn_dim = int(embed_dim * ffn_ratio)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ffn1 = nn.Sequential(
            nn.Linear(embed_dim, ffn_dim),
            nn.SiLU(),
            nn.Linear(ffn_dim, embed_dim),
        )
        self.dropout2 = nn.Dropout(dropout)

        # Attention block 2
        self.norm3 = nn.LayerNorm(embed_dim)
        self.attn2 = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            batch_first=True,
        )
        self.dropout3 = nn.Dropout(dropout)

        # FFN block 2
        self.norm4 = nn.LayerNorm(embed_dim)
        self.ffn2 = nn.Sequential(
            nn.Linear(embed_dim, ffn_dim),
            nn.SiLU(),
            nn.Linear(ffn_dim, embed_dim),
        )
        self.dropout4 = nn.Dropout(dropout)

        self.compress = nn.Linear(embed_dim, output_dim)
        self.norm_out = nn.LayerNorm(output_dim)

        self._output_dim = output_dim

    @staticmethod
    def _build_sinusoidal_pe(max_len: int, embed_dim: int) -> torch.Tensor:
        pe = torch.zeros(max_len, embed_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, embed_dim, 2, dtype=torch.float) * (-math.log(10000.0) / embed_dim)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0)  # [1, max_len, embed_dim]

    @property
    def output_dim(self) -> int:
        return self._output_dim

    def forward(
        self,
        prompt_embeds: torch.Tensor,
        prompt_mask: torch.Tensor,
    ) -> torch.Tensor:
        x = prompt_embeds  # [B, seq_len, embed_dim]

        summary = self.summary_token.expand(x.size(0), -1, -1)  # [B, 1, embed_dim]
        x = torch.cat([x, summary], dim=1)  # [B, seq_len+1, embed_dim]

        attn_mask = (prompt_mask == 0)  # [B, seq_len]
        attn_mask = F.pad(attn_mask, (0, 1), value=False)  # [B, seq_len+1]

        # Positional encoding
        if self.pos_encoding_type != 'none':
            x = x + self.pos_encoding[:, :x.size(1), :]

  
        x_norm = self.norm1(x)
        attn_out, _ = self.attn1(
            x_norm, x_norm, x_norm,
            key_padding_mask=attn_mask,
            need_weights=False,
        )
        x = x + self.dropout1(attn_out)
        x = x + self.dropout2(self.ffn1(self.norm2(x)))


        x_norm = self.norm3(x)
        attn_out, _ = self.attn2(
            x_norm, x_norm, x_norm,
            key_padding_mask=attn_mask,
            need_weights=False,
        )
        x = x + self.dropout3(attn_out)
        x = x + self.dropout4(self.ffn2(self.norm4(x)))

        x = x[:, -1, :]

        x = self.norm_out(self.compress(x))  # [B, output_dim]
        return x


class LightAttnPool(nn.Module):

    def __init__(
        self,
        embed_dim: int = 4096,
        output_dim: int = 1024,
        compressed_dim: int = 1024,
        dropout: float = 0.1,
        seq_len: int = 300,
        pos_encoding: str = 'none',
    ):
        super().__init__()

        self.embed_dim = embed_dim
        self.compressed_dim = compressed_dim
        self.pos_encoding_type = pos_encoding

        num_heads = 8  # 1024 / 8 = 128 head_dim

        # Early compression: 4096 → 1024
        self.compress_input = nn.Linear(embed_dim, compressed_dim)
        self.compress_norm = nn.LayerNorm(compressed_dim)

        # Learnable summary token (at compressed dim)
        self.summary_token = nn.Parameter(torch.randn(1, 1, compressed_dim))

        # Positional encoding
        if pos_encoding == 'sinusoidal':
            pe = AttnPool._build_sinusoidal_pe(seq_len + 1, compressed_dim)
            self.register_buffer('pos_encoding', pe)

        # Attention block 1
        self.norm1 = nn.LayerNorm(compressed_dim)
        self.attn1 = nn.MultiheadAttention(
            embed_dim=compressed_dim,
            num_heads=num_heads,
            batch_first=True,
        )
        self.dropout1 = nn.Dropout(dropout)

        # FFN block 1
        self.norm2 = nn.LayerNorm(compressed_dim)
        self.ffn1 = nn.Sequential(
            nn.Linear(compressed_dim, compressed_dim),
            nn.SiLU(),
            nn.Linear(compressed_dim, compressed_dim),
        )
        self.dropout2 = nn.Dropout(dropout)

        # Attention block 2
        self.norm3 = nn.LayerNorm(compressed_dim)
        self.attn2 = nn.MultiheadAttention(
            embed_dim=compressed_dim,
            num_heads=num_heads,
            batch_first=True,
        )
        self.dropout3 = nn.Dropout(dropout)

        # FFN block 2
        self.norm4 = nn.LayerNorm(compressed_dim)
        self.ffn2 = nn.Sequential(
            nn.Linear(compressed_dim, compressed_dim),
            nn.SiLU(),
            nn.Linear(compressed_dim, compressed_dim),
        )
        self.dropout4 = nn.Dropout(dropout)

        # Output projection
        self.compress = nn.Linear(compressed_dim, output_dim)
        self.norm_out = nn.LayerNorm(output_dim)

        self._output_dim = output_dim

    @property
    def output_dim(self) -> int:
        return self._output_dim

    def forward(
        self,
        prompt_embeds: torch.Tensor,
        prompt_mask: torch.Tensor,
    ) -> torch.Tensor:
        # Early compression: [B, seq, 4096] → [B, seq, 1024]
        x = self.compress_norm(self.compress_input(prompt_embeds))

        # Append summary token
        summary = self.summary_token.expand(x.size(0), -1, -1)
        x = torch.cat([x, summary], dim=1)  # [B, seq+1, compressed_dim]

        attn_mask = (prompt_mask == 0)
        attn_mask = F.pad(attn_mask, (0, 1), value=False)

        # Positional encoding
        if self.pos_encoding_type != 'none':
            x = x + self.pos_encoding[:, :x.size(1), :]

        # Block 1
        x_norm = self.norm1(x)
        attn_out, _ = self.attn1(
            x_norm, x_norm, x_norm,
            key_padding_mask=attn_mask,
            need_weights=False,
        )
        x = x + self.dropout1(attn_out)
        x = x + self.dropout2(self.ffn1(self.norm2(x)))

        # Block 2
        x_norm = self.norm3(x)
        attn_out, _ = self.attn2(
            x_norm, x_norm, x_norm,
            key_padding_mask=attn_mask,
            need_weights=False,
        )
        x = x + self.dropout3(attn_out)
        x = x + self.dropout4(self.ffn2(self.norm4(x)))

        # Extract summary token
        x = x[:, -1, :]  # [B, compressed_dim]

        # Output
        x = self.norm_out(self.compress(x))  # [B, output_dim]
        return x


class PerTokenScalarTextEncoder(nn.Module):

    def __init__(self, embed_dim: int = 4096, seq_len: int = 120, **kwargs):
        super().__init__()
        self.project = nn.Sequential(
            nn.Linear(embed_dim, 2048),
            nn.ReLU(),
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )
        self._output_dim = seq_len

    @property
    def output_dim(self) -> int:
        return self._output_dim

    def forward(
        self,
        prompt_embeds: torch.Tensor,
        prompt_mask: torch.Tensor,
    ) -> torch.Tensor:
        # [B, seq_len, embed_dim] → [B, seq_len, 1] → [B, seq_len]
        return self.project(prompt_embeds).squeeze(-1)


def get_text_encoder(name: str, embed_dim: int = 4096, seq_len: int = 120, **kwargs) -> nn.Module:

    encoders = {
        'attnpool': AttnPool,
        'lightattnpool': LightAttnPool,
        'pertokenscalar': PerTokenScalarTextEncoder,
        'summarytoken': AttnPool,
        'lightsummary': LightAttnPool,
    }

    if name not in encoders:
        raise ValueError(f"Unknown text encoder: {name}. Available: {list(encoders.keys())}")

    return encoders[name](embed_dim=embed_dim, seq_len=seq_len, **kwargs)
