"""
PixArt Score Predictor Models

Factory function to create models from configuration strings.

Text encoders (4):
    - 'pertokenscalar': Linear 4096→1 per token, output [B, 120]
    - 'pooling': Global pool over tokens, output [B, 4096]
    - 'compression_first': Compress→Attn→FFN→Pool, output [B, 512]
    - 'compression_later': Attn→FFN→Compress→Pool, output [B, 512]

Noise encoders (6):
    - 'spatial_shrink': Conv downsample 64→16, output [B, 1024]
    - 'channel_squash': 1x1 conv 4→1, flatten, output [B, 4096]
    - 'basic_cnn_avg': Basic CNN + GlobalAvgPool, output [B, 256]
    - 'basic_cnn_max': Basic CNN + GlobalMaxPool, output [B, 256]
    - 'resnet_avgpool': ResNet + AvgPool skip connections, output [B, 256]
    - 'resnet_maxpool': ResNet + MaxPool skip connections, output [B, 256]
"""

from .noise_encoders import get_noise_encoder
from .text_encoders import get_text_encoder
from .model import ScorePredictor


NOISE_ENCODERS = [
    'spatial_shrink',
    'channel_squash',
    'basic_cnn_avg',
    'basic_cnn_max',
    'resnet_avgpool',
    'resnet_maxpool',
]
TEXT_ENCODERS = ['pertokenscalar', 'pooling', 'compression_first', 'compression_later']


def get_model(
    noise_enc: str = 'basic_cnn_avg',
    text_enc: str = 'compression_first',
    dropout: float = 0.1,
    num_heads: int = 1,
) -> ScorePredictor:
    """
    Factory function to create model from configuration strings.

    Args:
        noise_enc: Noise encoder type
        text_enc: Text encoder type
        dropout: Dropout rate in fusion MLP
        num_heads: Number of output heads (1=single target, 3=multi-head)

    Returns:
        ScorePredictor model instance
    """
    if noise_enc not in NOISE_ENCODERS:
        raise ValueError(f"Unknown noise encoder: {noise_enc}. Available: {NOISE_ENCODERS}")
    if text_enc not in TEXT_ENCODERS:
        raise ValueError(f"Unknown text encoder: {text_enc}. Available: {TEXT_ENCODERS}")

    noise_encoder = get_noise_encoder(noise_enc)
    text_encoder = get_text_encoder(text_enc)

    return ScorePredictor(
        noise_encoder=noise_encoder,
        text_encoder=text_encoder,
        dropout=dropout,
        num_heads=num_heads,
    )


__all__ = [
    'get_model',
    'ScorePredictor',
    'get_text_encoder',
    'get_noise_encoder',
    'NOISE_ENCODERS',
    'TEXT_ENCODERS',
]
