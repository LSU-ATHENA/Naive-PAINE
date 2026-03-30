MODEL_DIMS = {
    'sdxl': {
        'latent_shape': (4, 128, 128),
        'spatial_size': 128,
        'embed_dim': 2048,
        'seq_len': 77,
    },
    'dreamshaper': {
        'latent_shape': (4, 128, 128),
        'spatial_size': 128,
        'embed_dim': 2048,
        'seq_len': 77,
    },
    'hunyuan_dit': {
        'latent_shape': (4, 128, 128),
        'spatial_size': 128,
        'embed_dim': 2048,
        'seq_len': 256,
    },
    'pixart_sigma': {
        'latent_shape': (4, 128, 128),
        'spatial_size': 128,
        'embed_dim': 4096,
        'seq_len': 300,
    },
'sana_sprint': {
        'latent_shape': (32, 32, 32),   # DC-AE f32c32: 32 channels, 32x spatial compression
        'spatial_size': 32,
        'embed_dim': 2304,              # Gemma-2-2B-IT caption_channels
        'seq_len': 300,
    },
}

# Override embed_dim/seq_len when using a non-default text encoder.
# Models with a single text encoder don't need entries here.
TEXT_EMBED_OVERRIDES = {
    'hunyuan_dit': {
        't5+clip': {'embed_dim': 2048, 'seq_len': 333},
    },
}


def get_dims(model_type: str, text_embed_type: str = 'default') -> dict:
    if model_type not in MODEL_DIMS:
        raise ValueError(
            f"Unknown model_type: {model_type}. "
            f"Available: {list(MODEL_DIMS.keys())}"
        )
    dims = MODEL_DIMS[model_type].copy()

    if text_embed_type not in ('default', 't5') and model_type in TEXT_EMBED_OVERRIDES:
        overrides = TEXT_EMBED_OVERRIDES[model_type].get(text_embed_type)
        if overrides is None:
            raise ValueError(
                f"Unknown text_embed_type '{text_embed_type}' for {model_type}. "
                f"Available: {list(TEXT_EMBED_OVERRIDES[model_type].keys())}"
            )
        dims.update(overrides)

    return dims
