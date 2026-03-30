import argparse
import sys
from pathlib import Path

import torch

from predictor.inference.loader import load_predictor
from predictor.inference.noise_selection import generate_noise_candidates, select_top_k_noise
from predictor.configs.model_dims import MODEL_DIMS, get_dims


PIPELINE_CONFIG = {
    'sdxl': {
        'class': 'StableDiffusionXLPipeline',
        'pretrained': 'stabilityai/stable-diffusion-xl-base-1.0',
    },
    'dreamshaper': {
        'class': 'StableDiffusionXLPipeline',
        'pretrained': 'Lykon/dreamshaper-xl-v2-turbo',
    },
    'hunyuan_dit': {
        'class': 'HunyuanDiTPipeline',
        'pretrained': 'Tencent-Hunyuan/HunyuanDiT-v1.2-Diffusers',
    },
    'pixart_sigma': {
        'class': 'PixArtSigmaPipeline',
        'pretrained': 'PixArt-alpha/PixArt-Sigma-XL-2-1024-MS',
    },
    'sana_sprint': {
        'class': 'SanaSprintPipeline',
        'pretrained': 'Efficient-Large-Model/Sana_Sprint_0.6B_1024px_diffusers',
        'dtype': torch.bfloat16,
    },
}


def load_pipeline(model_type: str, device: str = 'cuda', dtype=torch.float16):
    import diffusers

    config = PIPELINE_CONFIG[model_type]
    pipe_class = getattr(diffusers, config['class'])
    pipe_dtype = config.get('dtype', dtype)
    pipe = pipe_class.from_pretrained(
        config['pretrained'],
        torch_dtype=pipe_dtype,
    ).to(device)
    return pipe


def encode_prompt_for_model(pipe, prompt: str, model_type: str, device: str = 'cuda'):
    if model_type in ('sdxl', 'dreamshaper'):
        (
            prompt_embeds,
            negative_prompt_embeds,
            pooled_prompt_embeds,
            negative_pooled_prompt_embeds,
        ) = pipe.encode_prompt(
            prompt=prompt,
            prompt_2=prompt,
            device=device,
            num_images_per_prompt=1,
            do_classifier_free_guidance=True,
        )

        pred_embeds = prompt_embeds
        pred_mask = torch.ones(pred_embeds.shape[:2], device=device, dtype=torch.long)

        gen_kwargs = {
            'prompt_embeds': prompt_embeds,
            'negative_prompt_embeds': negative_prompt_embeds,
            'pooled_prompt_embeds': pooled_prompt_embeds,
            'negative_pooled_prompt_embeds': negative_pooled_prompt_embeds,
        }

    elif model_type == 'hunyuan_dit':
        result = pipe.encode_prompt(
            prompt=prompt,
            device=device,
            num_images_per_prompt=1,
            do_classifier_free_guidance=True,
        )

        has_t5_from_encode = (len(result) >= 4 and result[2].dtype in (torch.float16, torch.float32, torch.bfloat16))

        if has_t5_from_encode:
            prompt_embeds, negative_prompt_embeds = result[0], result[1]
            prompt_embeds_2, negative_prompt_embeds_2 = result[2], result[3]

            if len(result) >= 8:
                prompt_attention_mask = result[4]
                negative_prompt_attention_mask = result[5]
                prompt_attention_mask_2 = result[6]
                negative_prompt_attention_mask_2 = result[7]
            else:
                prompt_attention_mask = torch.ones(prompt_embeds.shape[:2], device=device, dtype=torch.long)
                negative_prompt_attention_mask = torch.ones(negative_prompt_embeds.shape[:2], device=device, dtype=torch.long)
                prompt_attention_mask_2 = torch.ones(prompt_embeds_2.shape[:2], device=device, dtype=torch.long)
                negative_prompt_attention_mask_2 = torch.ones(negative_prompt_embeds_2.shape[:2], device=device, dtype=torch.long)

            pred_embeds = prompt_embeds_2
            pred_mask = prompt_attention_mask_2

            gen_kwargs = {
                'prompt_embeds': prompt_embeds,
                'negative_prompt_embeds': negative_prompt_embeds,
                'prompt_embeds_2': prompt_embeds_2,
                'negative_prompt_embeds_2': negative_prompt_embeds_2,
                'prompt_attention_mask': prompt_attention_mask,
                'negative_prompt_attention_mask': negative_prompt_attention_mask,
                'prompt_attention_mask_2': prompt_attention_mask_2,
                'negative_prompt_attention_mask_2': negative_prompt_attention_mask_2,
            }
        else:
            max_seq_len = get_dims(model_type)['seq_len']
            tokens = pipe.tokenizer_2(
                prompt,
                max_length=max_seq_len,
                padding='max_length',
                truncation=True,
                return_tensors='pt',
            ).to(device)
            with torch.no_grad():
                t5_output = pipe.text_encoder_2(
                    tokens.input_ids,
                    attention_mask=tokens.attention_mask,
                )
            pred_embeds = t5_output[0].to(dtype=torch.float16)
            pred_mask = tokens.attention_mask

            gen_kwargs = {}

    elif model_type == 'pixart_sigma':
        max_seq_len = get_dims(model_type)['seq_len']
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
            clean_caption=False,
            max_sequence_length=max_seq_len,
        )

        pred_embeds = prompt_embeds
        pred_mask = prompt_attention_mask if prompt_attention_mask is not None else \
            torch.ones(pred_embeds.shape[:2], device=device, dtype=torch.long)

        gen_kwargs = {
            'prompt_embeds': prompt_embeds,
            'prompt_attention_mask': prompt_attention_mask,
            'negative_prompt_embeds': negative_prompt_embeds,
            'negative_prompt_attention_mask': negative_prompt_attention_mask,
        }

    elif model_type == 'sana_sprint':
        prompt_embeds, prompt_attention_mask = pipe.encode_prompt(
            prompt=prompt,
            device=device,
            max_sequence_length=get_dims(model_type)['seq_len'],
        )

        pred_embeds = prompt_embeds
        pred_mask = prompt_attention_mask

        gen_kwargs = {
            'prompt_embeds': prompt_embeds,
            'prompt_attention_mask': prompt_attention_mask,
        }

    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    return pred_embeds, pred_mask, gen_kwargs


def main():
    parser = argparse.ArgumentParser(
        description="Generate images with multi-model predictor-selected noise"
    )
    parser.add_argument("--model_type", type=str, required=True,
                        choices=list(MODEL_DIMS.keys()), help="Diffusion model type")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to predictor checkpoint (.pth)")
    parser.add_argument("--prompt", type=str, required=True, help="Text prompt")
    parser.add_argument("--N", type=int, default=100, help="Number of noise candidates")
    parser.add_argument("--B", type=int, default=4, help="Number of images to generate")
    parser.add_argument("--head", type=int, default=0,
                        help="Prediction head (0=hpsv2, 1=image_reward, 2=clip_score)")
    parser.add_argument("--steps", type=int, default=20, help="Inference steps")
    parser.add_argument("--guidance-scale", type=float, default=4.5, help="CFG scale")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument("--output-dir", type=str, default="output", help="Output directory")
    parser.add_argument("--compare", action="store_true",
                        help="Also generate B random baseline images")
    parser.add_argument("--device", type=str, default="cuda", help="Device")
    args = parser.parse_args()

    dims = get_dims(args.model_type)
    latent_shape = dims['latent_shape']
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"{'='*60}")
    print(f"PNM Multi-Model Image Generation")
    print(f"{'='*60}")
    print(f"  Model type: {args.model_type}")
    print(f"  Latent:     {latent_shape}")
    print(f"  Prompt:     {args.prompt}")
    print(f"  N:          {args.N} noise candidates")
    print(f"  B:          {args.B} images")
    print(f"  Checkpoint: {args.checkpoint}")
    print(f"  Output:     {output_dir}")
    print(f"{'='*60}")

    print(f"\nLoading {args.model_type} pipeline...")
    pipe = load_pipeline(args.model_type, device=args.device)

    print(f"Loading predictor from {args.checkpoint}...")
    predictor, norm_info = load_predictor(args.checkpoint, device=args.device)
    print(f"  num_heads={predictor.num_heads}")

    pred_embeds, pred_mask, gen_kwargs = encode_prompt_for_model(
        pipe, args.prompt, args.model_type, args.device
    )

    generator = torch.Generator(device=args.device).manual_seed(args.seed) if args.seed else None
    noises = generate_noise_candidates(
        num_candidates=args.N,
        latent_shape=latent_shape,
        device=args.device,
        dtype=pipe.unet.dtype if hasattr(pipe, 'unet') else pipe.transformer.dtype,
        generator=generator,
    )

    selected = select_top_k_noise(
        predictor=predictor,
        noises=noises,
        prompt_embeds=pred_embeds,
        prompt_mask=pred_mask,
        num_select=args.B,
        head_index=args.head,
    )

    latents = selected

    B = args.B
    expanded_kwargs = {}
    for k, v in gen_kwargs.items():
        if isinstance(v, torch.Tensor) and v.dim() >= 2:
            expanded_kwargs[k] = v.expand(B, *[-1] * (v.dim() - 1))
        else:
            expanded_kwargs[k] = v

    print(f"\nGenerating {B} images from top-{B} of {args.N} candidates...")
    result = pipe(
        prompt=None,
        **expanded_kwargs,
        latents=latents,
        num_images_per_prompt=1,
        num_inference_steps=args.steps,
        guidance_scale=args.guidance_scale,
    )

    for i, img in enumerate(result.images):
        path = output_dir / f"{args.model_type}_predictor_{i:02d}.png"
        img.save(path)
        print(f"  Saved: {path}")

    if args.compare:
        print(f"\nGenerating {B} baseline images (random noise)...")
        gen_random = torch.Generator(device=args.device).manual_seed(args.seed + 999) if args.seed else None
        result_rand = pipe(
            prompt=args.prompt,
            num_images_per_prompt=B,
            num_inference_steps=args.steps,
            guidance_scale=args.guidance_scale,
            generator=gen_random,
        )
        for i, img in enumerate(result_rand.images):
            path = output_dir / f"{args.model_type}_random_{i:02d}.png"
            img.save(path)
            print(f"  Saved: {path}")

    print(f"\nDone! Images saved to {output_dir}/")


if __name__ == "__main__":
    main()
