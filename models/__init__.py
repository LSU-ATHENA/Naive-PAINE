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
