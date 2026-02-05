import os
import json
import math
import hashlib
from pathlib import Path
from typing import Optional, Dict, Any

import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import hpsv2
import ImageReward as RM
from torchmetrics.multimodal.clip_score import CLIPScore

from diffusers import PixArtAlphaPipeline
from diffusers.models.attention_processor import Attention


class Config:
    MODEL_ID = "PixArt-alpha/PixArt-XL-2-512x512"
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    DTYPE = torch.float16

    HEIGHT = 512
    WIDTH = 512
    NUM_INFERENCE_STEPS = 20
    GUIDANCE_SCALE = 4.5

    MAX_SEQUENCE_LENGTH = 120
    NEGATIVE_PROMPT = ""

    NUM_IMAGES_PER_PROMPT = 20
    SAVE_DIR = Path("./out")
    PROMPTS_FILE = "prompts.txt"

    CAPTURE_STEP = 0
    POSITIVE_BATCH_INDEX = 1

    # Master seed for reproducible pseudo-random seed generation
    MASTER_SEED = 42


def derive_seed(master_seed, prompt_id, sample_idx):
    """Derive a 63-bit seed from (master_seed, prompt_id, sample_idx) via SHA256."""
    data = f"{master_seed}:{prompt_id}:{sample_idx}".encode()
    h = hashlib.sha256(data).digest()
    return int.from_bytes(h[:8], "big") & ((1 << 63) - 1)


def _to_additive_mask_f32(attention_mask, k_len, device):
    if attention_mask is None:
        return None

    mask = attention_mask.to(device)

    if mask.dim() == 2:
        mask = mask[:, None, None, :]
    elif mask.dim() == 3:
        mask = mask[:, None, :, :]
    elif mask.dim() == 4:
        pass
    else:
        raise ValueError(f"Unexpected attention_mask dims: {mask.dim()}")

    if mask.shape[-1] != k_len:
        if mask.shape[-1] > k_len:
            mask = mask[..., :k_len]
        else:
            raise ValueError(f"Mask K dim mismatch: got {mask.shape[-1]}, expected {k_len}")

    if mask.dtype == torch.bool:
        add = torch.zeros_like(mask, dtype=torch.float32)
        return add.masked_fill(~mask, float("-inf"))

    if not mask.is_floating_point():
        keep = mask.to(torch.float32)
        return (1.0 - keep) * -10000.0

    with torch.no_grad():
        m_min, m_max = float(mask.min()), float(mask.max())
    if 0.0 <= m_min and m_max <= 1.0:
        keep = mask.to(torch.float32)
        return (1.0 - keep) * -10000.0

    return mask.to(torch.float32)


class AttentionCaptureProcessor:
    def __init__(self):
        self.is_capturing = False
        self.step_counter = 0
        self.captured_probs_fp16_cpu: Optional[torch.Tensor] = None

    def __call__(
        self,
        attn,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
        **kwargs,
    ):
        bsz, q_len, _ = hidden_states.shape

        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads
        scale = 1.0 / math.sqrt(head_dim)

        query = query.view(bsz, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(bsz, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(bsz, -1, attn.heads, head_dim).transpose(1, 2)

        k_len = key.shape[-2]
        add_mask_f32 = _to_additive_mask_f32(attention_mask, k_len=k_len, device=query.device)

        sdpa_mask = add_mask_f32.to(dtype=query.dtype) if add_mask_f32 is not None else None
        official_out = F.scaled_dot_product_attention(
            query, key, value,
            attn_mask=sdpa_mask,
            dropout_p=0.0,
            scale=scale,
        )

        if (
            self.is_capturing
            and self.step_counter == Config.CAPTURE_STEP
            and self.captured_probs_fp16_cpu is None
        ):
            with torch.no_grad():
                q32, k32 = query.float(), key.float()
                scores = torch.matmul(q32, k32.transpose(-2, -1)) * scale
                if add_mask_f32 is not None:
                    scores = scores + add_mask_f32
                probs = torch.softmax(scores, dim=-1)
                self.captured_probs_fp16_cpu = probs.detach().cpu().half()

        out = official_out.transpose(1, 2).reshape(bsz, q_len, attn.heads * head_dim)
        out = attn.to_out[0](out)
        out = attn.to_out[1](out)
        return out


def setup_environment():
    for d in ["images", "noise", "attn", "embeds"]:
        (Config.SAVE_DIR / d).mkdir(parents=True, exist_ok=True)
    Path("shared_assets/checkpoints").mkdir(parents=True, exist_ok=True)

    print(f"Pipeline: {Config.MODEL_ID}")
    pipe = PixArtAlphaPipeline.from_pretrained(Config.MODEL_ID, torch_dtype=Config.DTYPE).to(Config.DEVICE)

    scorer_ir = RM.load("ImageReward-v1.0", download_root="shared_assets/checkpoints")
    scorer_clip = CLIPScore(model_name_or_path="openai/clip-vit-base-patch16").to(Config.DEVICE)

    return pipe, scorer_ir, scorer_clip


def inject_custom_processor(pipe):
    processor = AttentionCaptureProcessor()
    pipe.transformer.transformer_blocks[0].attn2.set_processor(processor)
    return processor


def generate_raw_noise(seed):
    gen = torch.Generator(device=Config.DEVICE).manual_seed(seed)
    return torch.randn(
        (1, 4, Config.HEIGHT // 8, Config.WIDTH // 8),
        generator=gen,
        device=Config.DEVICE,
        dtype=Config.DTYPE,
    )


def encode_and_save_embeddings(pipe, prompt_str, embed_path):
    with torch.inference_mode():
        pe, pm, ne, nm = pipe.encode_prompt(
            prompt=prompt_str,
            negative_prompt=Config.NEGATIVE_PROMPT,
            max_sequence_length=Config.MAX_SEQUENCE_LENGTH,
            device=Config.DEVICE,
            num_images_per_prompt=1,
        )
    embeds = {
        "prompt_embeds": pe.detach().cpu().half(),
        "prompt_mask": pm.detach().cpu(),
        "neg_embeds": ne.detach().cpu().half(),
        "neg_mask": nm.detach().cpu(),
    }
    torch.save(embeds, embed_path)
    return embeds


def compute_scores(image, prompt_str, scorer_ir, scorer_clip):
    prompt = prompt_str.strip()
    image_rgb = image.convert("RGB")

    with torch.inference_mode():
        hps = round(float(hpsv2.score([image_rgb], prompt, hps_version="v2.1")[0]), 4)
        ir = round(float(scorer_ir.score(prompt, image_rgb)), 4)
        img = torch.from_numpy(np.array(image_rgb)).permute(2, 0, 1).unsqueeze(0).float().to(Config.DEVICE)
        clip_raw = float(scorer_clip(img, [prompt]).detach().cpu())  # ~100 * cosine
        clip_cos = round(clip_raw / 100.0, 4)   

    return {
        "hpsv2": hps,
        "image_reward": ir,
        "clip_score": clip_cos,
    }


def main():
    pipe, scorer_ir, scorer_clip = setup_environment()
    processor = inject_custom_processor(pipe)

    with open(Config.PROMPTS_FILE, "r") as f:
        prompts = [line.strip() for line in f if line.strip()]
        #prompts = prompts[:2]

    meta_path = Config.SAVE_DIR / "metadata.jsonl"
    existing = set()
    if meta_path.exists():
        with open(meta_path, "r") as f:
            for line in f:
                if line.strip():
                    d = json.loads(line)
                    existing.add((d["prompt_id"], d["sample_idx"]))

    def step_callback(step, timestep, latents):
        processor.step_counter = step
        return latents

    with open(meta_path, "a") as meta_file:
        for p_idx, prompt_str in enumerate(tqdm(prompts, desc="Prompts")):
            embed_path = Config.SAVE_DIR / "embeds" / f"p{p_idx:04d}.pt"
            if embed_path.exists():
                embeds = torch.load(embed_path, map_location="cpu", weights_only=False)
            else:
                embeds = encode_and_save_embeddings(pipe, prompt_str, embed_path)

            for i in range(Config.NUM_IMAGES_PER_PROMPT):
                if (p_idx, i) in existing:
                    continue

                # Derive pseudo-random 63-bit seed
                seed = derive_seed(Config.MASTER_SEED, p_idx, i)
                name = f"p{p_idx:04d}_s{i:02d}"

                raw = generate_raw_noise(seed)
                scaled = raw * pipe.scheduler.init_noise_sigma

                processor.is_capturing = True
                processor.step_counter = 0
                processor.captured_probs_fp16_cpu = None

                with torch.inference_mode():
                    result = pipe(
                        prompt=None,
                        negative_prompt=None,
                        prompt_embeds=embeds["prompt_embeds"].to(Config.DEVICE),
                        prompt_attention_mask=embeds["prompt_mask"].to(Config.DEVICE),
                        negative_prompt_embeds=embeds["neg_embeds"].to(Config.DEVICE),
                        negative_prompt_attention_mask=embeds["neg_mask"].to(Config.DEVICE),
                        latents=scaled.to(Config.DEVICE),
                        num_inference_steps=Config.NUM_INFERENCE_STEPS,
                        guidance_scale=Config.GUIDANCE_SCALE,
                        callback=step_callback,
                        callback_steps=1,
                    )

                image = result.images[0]
                scores = compute_scores(image, prompt_str, scorer_ir, scorer_clip)

                image.save(Config.SAVE_DIR / "images" / f"{name}.jpg")
                torch.save(raw.detach().cpu().half(), Config.SAVE_DIR / "noise" / f"{name}.pt")

                if processor.captured_probs_fp16_cpu is not None:
                    probs = processor.captured_probs_fp16_cpu
                    if probs.shape[0] > Config.POSITIVE_BATCH_INDEX:
                        probs = probs[Config.POSITIVE_BATCH_INDEX:Config.POSITIVE_BATCH_INDEX + 1]
                    else:
                        probs = probs[0:1]
                    torch.save(probs, Config.SAVE_DIR / "attn" / f"{name}.pt")

                meta = {
                    "prompt_id": p_idx,
                    "sample_idx": i,
                    "seed": seed,
                    "prompt": prompt_str,
                    **scores,
                }
                meta_file.write(json.dumps(meta) + "\n")
                meta_file.flush()

                del result, image, raw, scaled
                torch.cuda.empty_cache()


if __name__ == "__main__":
    main()