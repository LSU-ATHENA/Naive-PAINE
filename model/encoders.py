import torch
import torch.nn as nn


class CustomNoiseEncoder(nn.Module):

    def __init__(self, in_channels: int = 4, spatial_size: int = 128):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=5, stride=1, padding=2)
        self.bn1 = nn.BatchNorm2d(64)
        self.act1 = nn.SiLU()
        self.do1 = nn.Dropout2d(0.3)
        self.skip_1 = nn.Conv2d(in_channels, 64, kernel_size=1, stride=1, padding=0)

        self.ds1 = nn.Conv2d(64, 64, kernel_size=5, stride=2, padding=2)
        self.ds_bn1 = nn.BatchNorm2d(64)
        self.ds_act1 = nn.SiLU()
        self.ds_do1 = nn.Dropout2d(0.3)

        self.conv2 = nn.Conv2d(64, 64, kernel_size=5, stride=1, padding=2)
        self.bn2 = nn.BatchNorm2d(64)
        self.act2 = nn.SiLU()
        self.do2 = nn.Dropout2d(0.3)

        self.ds2 = nn.Conv2d(64, 128, kernel_size=5, stride=2, padding=2)
        self.ds_bn2 = nn.BatchNorm2d(128)
        self.ds_act2 = nn.SiLU()
        self.ds_do2 = nn.Dropout2d(0.3)

        self.conv3 = nn.Conv2d(128, 128, kernel_size=5, stride=1, padding=2)
        self.bn3 = nn.BatchNorm2d(128)
        self.act3 = nn.SiLU()
        self.do3 = nn.Dropout2d(0.3)

        self.ds3 = nn.Conv2d(128, 256, kernel_size=5, stride=2, padding=2)
        self.ds_bn3 = nn.BatchNorm2d(256)
        self.ds_act3 = nn.SiLU()
        self.ds_do3 = nn.Dropout2d(0.3)

        self.conv4 = nn.Conv2d(256, 256, kernel_size=5, stride=1, padding=2)
        self.bn4 = nn.BatchNorm2d(256)
        self.act4 = nn.SiLU()
        self.do4 = nn.Dropout2d(0.3)

        self.ds4 = nn.Conv2d(256, 1024, kernel_size=5, stride=2, padding=2)
        self.ds_bn4 = nn.BatchNorm2d(1024)
        self.ds_act4 = nn.SiLU()
        self.ds_do4 = nn.Dropout2d(0.3)

        self.pool = nn.AdaptiveMaxPool2d((1, 1))
        self.final_do = nn.Dropout(0.3)

        self._output_dim = 1024

    @property
    def output_dim(self) -> int:
        return self._output_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.skip_1(x)
        x = self.do1(self.act1(self.bn1(self.conv1(x)))) + identity
        x = self.ds_do1(self.ds_act1(self.ds_bn1(self.ds1(x))))
        x = self.do2(self.act2(self.bn2(self.conv2(x)))) + x
        x = self.ds_do2(self.ds_act2(self.ds_bn2(self.ds2(x))))
        x = self.do3(self.act3(self.bn3(self.conv3(x)))) + x
        x = self.ds_do3(self.ds_act3(self.ds_bn3(self.ds3(x))))
        x = self.do4(self.act4(self.bn4(self.conv4(x)))) + x
        x = self.ds_do4(self.ds_act4(self.ds_bn4(self.ds4(x))))

        x = self.pool(x)
        x = x.flatten(start_dim=1)
        x = self.final_do(x)
        return x


class _BasicBlock(nn.Module):
    """ResNet BasicBlock: two 3x3 convs + projection shortcut, SiLU activations."""

    def __init__(self, in_ch: int, out_ch: int, stride: int = 1, dropout: float = 0.0):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.act = nn.SiLU()
        self.drop = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.drop(self.act(self.bn1(self.conv1(x))))
        out = self.bn2(self.conv2(out))
        return self.act(out + self.shortcut(x))


class ResNetNoiseEncoder(nn.Module):
    """4-stage ResNet: downsamples the latent while widening channels, adaptive-max-pool to a vector."""

    def __init__(self, in_channels: int = 4, spatial_size: int = 128,
                 widths=(64, 128, 256, 512), dropout: float = 0.1):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, widths[0], 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(widths[0]),
            nn.SiLU(),
        )
        stages, in_ch = [], widths[0]
        for w in widths:
            stages.append(_BasicBlock(in_ch, w, stride=2, dropout=dropout))
            in_ch = w
        self.stages = nn.Sequential(*stages)
        self.pool = nn.AdaptiveMaxPool2d((1, 1))
        self.final_do = nn.Dropout(dropout)
        self._output_dim = widths[-1]

    @property
    def output_dim(self) -> int:
        return self._output_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stages(self.stem(x))
        x = self.pool(x).flatten(start_dim=1)
        return self.final_do(x)


def get_noise_encoder(name: str = 'custom', spatial_size: int = 128,
                      in_channels: int = 4, dropout: float = 0.1, **kwargs) -> nn.Module:
    if name == 'resnet':
        return ResNetNoiseEncoder(in_channels=in_channels, spatial_size=spatial_size, dropout=dropout)
    return CustomNoiseEncoder(in_channels=in_channels, spatial_size=spatial_size)


import math
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

        self.summary_token = nn.Parameter(torch.randn(1, 1, embed_dim))

        if pos_encoding == 'sinusoidal':
            pe = self._build_sinusoidal_pe(seq_len + 1, embed_dim)
            self.register_buffer('pos_encoding', pe)

        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn1 = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            batch_first=True,
        )
        self.dropout1 = nn.Dropout(dropout)

        ffn_dim = int(embed_dim * ffn_ratio)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ffn1 = nn.Sequential(
            nn.Linear(embed_dim, ffn_dim),
            nn.SiLU(),
            nn.Linear(ffn_dim, embed_dim),
        )
        self.dropout2 = nn.Dropout(dropout)

        self.norm3 = nn.LayerNorm(embed_dim)
        self.attn2 = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            batch_first=True,
        )
        self.dropout3 = nn.Dropout(dropout)

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
        return pe.unsqueeze(0)

    @property
    def output_dim(self) -> int:
        return self._output_dim

    def forward(
        self,
        prompt_embeds: torch.Tensor,
        prompt_mask: torch.Tensor,
    ) -> torch.Tensor:
        x = prompt_embeds

        summary = self.summary_token.expand(x.size(0), -1, -1)
        x = torch.cat([x, summary], dim=1)

        attn_mask = (prompt_mask == 0)
        attn_mask = F.pad(attn_mask, (0, 1), value=False)

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

        x = self.norm_out(self.compress(x))
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

        num_heads = 8

        self.compress_input = nn.Linear(embed_dim, compressed_dim)
        self.compress_norm = nn.LayerNorm(compressed_dim)

        self.summary_token = nn.Parameter(torch.randn(1, 1, compressed_dim))

        if pos_encoding == 'sinusoidal':
            pe = AttnPool._build_sinusoidal_pe(seq_len + 1, compressed_dim)
            self.register_buffer('pos_encoding', pe)

        self.norm1 = nn.LayerNorm(compressed_dim)
        self.attn1 = nn.MultiheadAttention(
            embed_dim=compressed_dim,
            num_heads=num_heads,
            batch_first=True,
        )
        self.dropout1 = nn.Dropout(dropout)

        self.norm2 = nn.LayerNorm(compressed_dim)
        self.ffn1 = nn.Sequential(
            nn.Linear(compressed_dim, compressed_dim),
            nn.SiLU(),
            nn.Linear(compressed_dim, compressed_dim),
        )
        self.dropout2 = nn.Dropout(dropout)

        self.norm3 = nn.LayerNorm(compressed_dim)
        self.attn2 = nn.MultiheadAttention(
            embed_dim=compressed_dim,
            num_heads=num_heads,
            batch_first=True,
        )
        self.dropout3 = nn.Dropout(dropout)

        self.norm4 = nn.LayerNorm(compressed_dim)
        self.ffn2 = nn.Sequential(
            nn.Linear(compressed_dim, compressed_dim),
            nn.SiLU(),
            nn.Linear(compressed_dim, compressed_dim),
        )
        self.dropout4 = nn.Dropout(dropout)

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
        x = self.compress_norm(self.compress_input(prompt_embeds))

        summary = self.summary_token.expand(x.size(0), -1, -1)
        x = torch.cat([x, summary], dim=1)

        attn_mask = (prompt_mask == 0)
        attn_mask = F.pad(attn_mask, (0, 1), value=False)

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

        x = self.norm_out(self.compress(x))
        return x


def get_text_encoder(name: str, embed_dim: int = 4096, seq_len: int = 120, **kwargs) -> nn.Module:
    encoders = {
        'summarytoken': AttnPool,      # SDXL, DreamShaper, Hunyuan, Sana
        'lightsummary': LightAttnPool,  # PixArt-Sigma
    }
    if name not in encoders:
        raise ValueError(f"Unknown text encoder: {name}. Available: {list(encoders.keys())}")
    return encoders[name](embed_dim=embed_dim, seq_len=seq_len, **kwargs)
