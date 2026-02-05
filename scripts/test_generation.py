"""
Quick test: load predictor, score noise candidates, generate one image.

Usage:
    python scripts/test_generation.py
    python scripts/test_generation.py --config pnm/configs/srcc_champion.yaml
"""

import argparse
import sys
from pathlib import Path

import torch

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from diffusers import PixArtAlphaPipeline
from pnm.models import ScorePredictor
from pnm.inference import generate_noise_candidates, select_top_k_noise


def main():
    parser = argparse.ArgumentParser(description="Test PNM generation pipeline")
    parser.add_argument("--config", type=str, default="pnm/configs/ndcg_champion.yaml")
    parser.add_argument("--prompt", type=str, default="A cat wearing a hat, digital art")
    parser.add_argument("--N", type=int, default=50, help="Noise candidates")
    parser.add_argument("--B", type=int, default=2, help="Images to generate")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output-dir", type=str, default="output/test")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load pipeline
    print("Loading PixArt-Alpha pipeline...")
    pipe = PixArtAlphaPipeline.from_pretrained(
        "PixArt-alpha/PixArt-XL-2-512x512",
        torch_dtype=torch.float16,
    ).to(args.device)

    # 2. Load predictor
    print(f"Loading predictor from {args.config}...")
    predictor, norm_info = ScorePredictor.from_config(args.config, device=args.device)
    print(f"  num_heads={predictor.num_heads}, norm={norm_info}")

    # 3. Encode prompt
    print(f"Encoding prompt: '{args.prompt}'")
    prompt_embeds, prompt_mask, neg_embeds, neg_mask = pipe.encode_prompt(
        prompt=args.prompt,
        do_classifier_free_guidance=True,
        num_images_per_prompt=1,
        device=args.device,
        clean_caption=True,
        max_sequence_length=120,
    )
    print(f"  prompt_embeds: {prompt_embeds.shape}")

    # Use positive embeddings for predictor
    if prompt_embeds.shape[0] == 2:
        pred_embeds = prompt_embeds[1:2]
        pred_mask = prompt_mask[1:2] if prompt_mask is not None else None
    else:
        pred_embeds = prompt_embeds
        pred_mask = prompt_mask

    if pred_mask is None:
        pred_mask = torch.ones(pred_embeds.shape[:2], device=args.device, dtype=torch.long)

    # 4. Generate and score noise
    print(f"Generating {args.N} noise candidates...")
    noises = generate_noise_candidates(
        num_candidates=args.N,
        latent_shape=(4, 64, 64),
        device=args.device,
        dtype=torch.float16,
    )

    print(f"Selecting top {args.B} from {args.N} candidates...")
    selected = select_top_k_noise(
        predictor=predictor,
        noises=noises,
        prompt_embeds=pred_embeds,
        prompt_mask=pred_mask,
        num_select=args.B,
    )
    print(f"  Selected noise shape: {selected.shape}")

    # 5. Generate images
    latents = selected * pipe.scheduler.init_noise_sigma
    print(f"Generating {args.B} images...")
    result = pipe(
        prompt=None,
        prompt_embeds=prompt_embeds,
        prompt_attention_mask=prompt_mask,
        negative_prompt=None,
        negative_prompt_embeds=neg_embeds,
        negative_prompt_attention_mask=neg_mask,
        latents=latents,
        num_images_per_prompt=1,
        num_inference_steps=20,
        guidance_scale=4.5,
    )

    for i, img in enumerate(result.images):
        path = output_dir / f"test_{i:02d}.png"
        img.save(path)
        print(f"  Saved: {path}")

    print("\nTest passed!")


if __name__ == "__main__":
    main()
