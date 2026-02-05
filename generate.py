"""
Generate images using PNM predictor-selected noise with PixArt-Alpha.

The predictor scores N random noise candidates and selects the top B
for image generation via the standard PixArt-Alpha pipeline.

Usage:
    python generate.py --prompt "A cat wearing a hat" --N 100 --B 4
    python generate.py --prompt "A sunset over mountains" --N 50 --B 2 --compare
"""

import argparse
import sys
from pathlib import Path

import torch
from diffusers import PixArtAlphaPipeline

from pnm.models import ScorePredictor
from pnm.inference import generate_noise_candidates, select_top_k_noise


def encode_prompt(pipe, prompt, device):
    """Encode prompt using the pipeline's text encoder.

    Returns:
        (prompt_embeds, prompt_mask, negative_embeds, negative_mask)
    """
    # PixArt encode_prompt returns 4 values
    (
        prompt_embeds,
        prompt_attention_mask,
        negative_prompt_embeds,
        negative_prompt_attention_mask,
    ) = pipe.encode_prompt(
        prompt=prompt,
        do_classifier_free_guidance=True,
        num_images_per_prompt=1,
        device=device,
        clean_caption=True,
        max_sequence_length=120,
    )
    return prompt_embeds, prompt_attention_mask, negative_prompt_embeds, negative_prompt_attention_mask


def generate_with_predictor(pipe, predictor, prompt, N, B, head_index, device, steps, guidance_scale, seed=None):
    """Generate B images using predictor to select from N noise candidates."""
    # Encode prompt once (returns separate positive and negative embeddings)
    prompt_embeds, prompt_mask, neg_embeds, neg_mask = encode_prompt(pipe, prompt, device)

    # prompt_embeds is [1, seq, dim] (positive only), use directly for predictor
    pred_embeds = prompt_embeds
    pred_mask = prompt_mask
    if pred_mask is None:
        pred_mask = torch.ones(pred_embeds.shape[:2], device=device, dtype=torch.long)

    # Generate N candidates and select top B
    generator = torch.Generator(device=device).manual_seed(seed) if seed is not None else None
    noises = generate_noise_candidates(
        num_candidates=N,
        latent_shape=(4, 64, 64),  # PixArt-Alpha 512x512
        device=device,
        dtype=pipe.transformer.dtype,
        generator=generator,
    )

    selected = select_top_k_noise(
        predictor=predictor,
        noises=noises,
        prompt_embeds=pred_embeds,
        prompt_mask=pred_mask,
        num_select=B,
        head_index=head_index,
    )

    # Scale noise by scheduler's init sigma
    latents = selected * pipe.scheduler.init_noise_sigma

    # Expand embeddings from [1, seq, dim] to [B, seq, dim] to match B latents
    prompt_embeds_b = prompt_embeds.expand(B, -1, -1)
    neg_embeds_b = neg_embeds.expand(B, -1, -1)
    prompt_mask_b = prompt_mask.expand(B, -1) if prompt_mask is not None else None
    neg_mask_b = neg_mask.expand(B, -1) if neg_mask is not None else None

    # Generate images using standard pipeline
    result = pipe(
        prompt=None,
        prompt_embeds=prompt_embeds_b,
        prompt_attention_mask=prompt_mask_b,
        negative_prompt=None,
        negative_prompt_embeds=neg_embeds_b,
        negative_prompt_attention_mask=neg_mask_b,
        latents=latents,
        num_images_per_prompt=1,
        num_inference_steps=steps,
        guidance_scale=guidance_scale,
    )
    return result.images


def generate_random(pipe, prompt, B, device, steps, guidance_scale, seed=None):
    """Generate B images with random noise (baseline)."""
    generator = torch.Generator(device=device).manual_seed(seed + 999) if seed is not None else None
    result = pipe(
        prompt=prompt,
        num_images_per_prompt=B,
        num_inference_steps=steps,
        guidance_scale=guidance_scale,
        generator=generator,
    )
    return result.images


def main():
    parser = argparse.ArgumentParser(
        description="Generate images with PNM predictor-selected noise"
    )
    parser.add_argument("--prompt", type=str, required=True, help="Text prompt")
    parser.add_argument("--N", type=int, default=100, help="Number of noise candidates to score")
    parser.add_argument("--B", type=int, default=4, help="Number of images to generate")
    parser.add_argument("--config", type=str, default="pnm/configs/ndcg_champion.yaml",
                        help="Predictor config YAML path")
    parser.add_argument("--head", type=int, default=0,
                        help="Prediction head (0=hpsv2, 1=image_reward, 2=clip_score)")
    parser.add_argument("--steps", type=int, default=20, help="Inference steps")
    parser.add_argument("--guidance-scale", type=float, default=4.5, help="CFG scale")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument("--output-dir", type=str, default="output", help="Output directory")
    parser.add_argument("--compare", action="store_true",
                        help="Also generate B random baseline images for comparison")
    parser.add_argument("--device", type=str, default="cuda", help="Device (cuda or cpu)")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"{'='*60}")
    print(f"PNM Image Generation")
    print(f"{'='*60}")
    print(f"  Prompt:     {args.prompt}")
    print(f"  N:          {args.N} noise candidates")
    print(f"  B:          {args.B} images to generate")
    print(f"  Config:     {args.config}")
    print(f"  Head:       {args.head}")
    print(f"  Steps:      {args.steps}")
    print(f"  CFG scale:  {args.guidance_scale}")
    print(f"  Seed:       {args.seed}")
    print(f"  Output:     {output_dir}")
    print(f"{'='*60}")

    # Load pipeline
    print("\nLoading PixArt-Alpha pipeline...")
    pipe = PixArtAlphaPipeline.from_pretrained(
        "PixArt-alpha/PixArt-XL-2-512x512",
        torch_dtype=torch.float16,
    ).to(args.device)
    print("  Pipeline loaded.")

    # Load predictor
    print(f"Loading predictor from {args.config}...")
    predictor, norm_info = ScorePredictor.from_config(args.config, device=args.device)
    print(f"  Predictor loaded (num_heads={predictor.num_heads})")

    # Generate with predictor
    print(f"\nGenerating {args.B} images from top-{args.B} of {args.N} candidates...")
    images_pred = generate_with_predictor(
        pipe, predictor, args.prompt, args.N, args.B, args.head,
        args.device, args.steps, args.guidance_scale, args.seed
    )

    for i, img in enumerate(images_pred):
        path = output_dir / f"predictor_{i:02d}.png"
        img.save(path)
        print(f"  Saved: {path}")

    # Optionally generate random baseline
    if args.compare:
        print(f"\nGenerating {args.B} baseline images (random noise)...")
        images_rand = generate_random(pipe, args.prompt, args.B, args.device, args.steps, args.guidance_scale, args.seed)

        for i, img in enumerate(images_rand):
            path = output_dir / f"random_{i:02d}.png"
            img.save(path)
            print(f"  Saved: {path}")

    print(f"\nImages saved to {output_dir}/")


if __name__ == "__main__":
    main()
