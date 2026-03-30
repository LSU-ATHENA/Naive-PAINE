from .noise_encoders import get_noise_encoder
from .text_encoders import get_text_encoder
from .model import ScorePredictor


NOISE_ENCODERS = ['custom']
TEXT_ENCODERS = ['summarytoken', 'lightsummary', 'pertokenscalar']


def get_model(
    noise_enc: str = 'custom',
    text_enc: str = 'summarytoken',
    dropout: float = 0.1,
    num_heads: int = 1,
    spatial_size: int = 128,
    in_channels: int = 4,
    embed_dim: int = 2048,
    seq_len: int = 77,
    pos_encoding: str = 'none',
) -> ScorePredictor:
    if noise_enc not in NOISE_ENCODERS:
        raise ValueError(f"Unknown noise encoder: {noise_enc}. Available: {NOISE_ENCODERS}")
    if text_enc not in TEXT_ENCODERS:
        raise ValueError(f"Unknown text encoder: {text_enc}. Available: {TEXT_ENCODERS}")

    text_encoder = get_text_encoder(text_enc, embed_dim=embed_dim, seq_len=seq_len, pos_encoding=pos_encoding)
    noise_encoder = get_noise_encoder(spatial_size=spatial_size, in_channels=in_channels)

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
