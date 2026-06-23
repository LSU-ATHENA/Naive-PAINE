import torch

SDXL_CONFIG = {
    'model_id': 'stabilityai/stable-diffusion-xl-base-1.0',
    'resolution': 1024,
    'latent_shape': (4, 128, 128),
    'embed_dim': 2048,
    'pooled_dim': 1280,
    'max_seq_len': 77,
    'guidance_scale': 5.5,
    'num_inference_steps': 50,
}


def load_sdxl_pipeline(device='cuda'):
    from diffusers import StableDiffusionXLPipeline
    pipe = StableDiffusionXLPipeline.from_pretrained(
        SDXL_CONFIG['model_id'], torch_dtype=torch.float16,
    )
    pipe.to(device)
    pipe.upcast_vae()
    return pipe


HUNYUAN_CONFIG = {
    'model_id': 'Tencent-Hunyuan/HunyuanDiT-v1.2-Diffusers',
    'resolution': 1024,
    'latent_shape': (4, 128, 128),
    'clip_dim': 1024,
    'clip_seq_len': 77,
    't5_dim': 2048,
    't5_seq_len': 256,
    'guidance_scale': 5.0,
    'num_inference_steps': 50,
}


def load_hunyuan_pipeline(device='cuda'):
    from diffusers import HunyuanDiTPipeline
    pipe = HunyuanDiTPipeline.from_pretrained(
        HUNYUAN_CONFIG['model_id'], torch_dtype=torch.float16,
    )
    return pipe.to(device)


DREAMSHAPER_CONFIG = {
    'model_id': 'Lykon/dreamshaper-xl-v2-turbo',
    'resolution': 1024,
    'latent_shape': (4, 128, 128),
    'embed_dim': 2048,
    'pooled_dim': 1280,
    'max_seq_len': 77,
    'guidance_scale': 2.0,
    'num_inference_steps': 8,
}


def load_dreamshaper_pipeline(device='cuda'):
    from diffusers import StableDiffusionXLPipeline
    pipe = StableDiffusionXLPipeline.from_pretrained(
        DREAMSHAPER_CONFIG['model_id'], torch_dtype=torch.float16,
    )
    pipe.to(device)
    pipe.upcast_vae()
    return pipe


PIXART_SIGMA_CONFIG = {
    'model_id': 'PixArt-alpha/PixArt-Sigma-XL-2-1024-MS',
    'resolution': 1024,
    'latent_shape': (4, 128, 128),
    't5_dim': 4096,
    't5_seq_len': 300,
    'guidance_scale': 4.5,
    'num_inference_steps': 20,
}


def load_pixart_sigma_pipeline(device='cuda'):
    from diffusers import PixArtSigmaPipeline
    pipe = PixArtSigmaPipeline.from_pretrained(
        PIXART_SIGMA_CONFIG['model_id'], torch_dtype=torch.float16,
    )
    return pipe.to(device)


# --- SANA-Sprint 0.6B (matches Noise Hypernetworks paper exactly) ---
SANA_SPRINT_CONFIG = {
    'model_id': 'Efficient-Large-Model/Sana_Sprint_0.6B_1024px_diffusers',
    'resolution': 1024,
    'latent_shape': (32, 32, 32),     # DC-AE f32c32: 32 channels, 32x spatial compression
    'text_dim': 2304,                 # Gemma-2-2B-IT caption_channels
    'text_seq_len': 300,              # max_sequence_length
    'guidance_scale': 4.5,            # Embedded CFG (guidance_embeds_scale=0.1)
    'num_inference_steps': 4,         # Matching HyperNoise paper's SANA-Sprint inference setting
}


def load_sana_sprint_pipeline(device='cuda'):
    from diffusers import SanaSprintPipeline
    pipe = SanaSprintPipeline.from_pretrained(
        SANA_SPRINT_CONFIG['model_id'], torch_dtype=torch.bfloat16,
    )
    return pipe.to(device)
