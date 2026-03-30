import argparse
from pathlib import Path

import torch
from diffusers import StableDiffusionXLPipeline, PixArtSigmaPipeline

from predictor.inference.loader import load_predictor
from predictor.inference.noise_selection import generate_noise_candidates, select_top_k_noise
from predictor.configs.model_dims import get_dims


def encode_prompt(pipe, prompt, model_type, device):
    if model_type in ('sdxl', 'dreamshaper'):
        embeds, neg_embeds, pooled, neg_pooled = pipe.encode_prompt(
            prompt=prompt,
            prompt_2=prompt,
            device=device,
            num_images_per_prompt=1,
            do_classifier_free_guidance=True,
        )
        pred_embeds = embeds
        pred_mask = torch.ones(embeds.shape[:2], device=device, dtype=torch.long)
        gen_kwargs = {
            "prompt_embeds": embeds,
            "negative_prompt_embeds": neg_embeds,
            "pooled_prompt_embeds": pooled,
            "negative_pooled_prompt_embeds": neg_pooled,
        }

    elif model_type == 'pixart_sigma':
        seq_len = get_dims(model_type)["seq_len"]
        embeds, mask, neg_embeds, neg_mask = pipe.encode_prompt(
            prompt=prompt,
            do_classifier_free_guidance=True,
            num_images_per_prompt=1,
            device=device,
            clean_caption=False,
            max_sequence_length=seq_len,
        )
        pred_embeds = embeds
        pred_mask = mask if mask is not None else torch.ones(
            embeds.shape[:2], device=device, dtype=torch.long,
        )
        gen_kwargs = {
            "prompt_embeds": embeds,
            "prompt_attention_mask": mask,
            "negative_prompt_embeds": neg_embeds,
            "negative_prompt_attention_mask": neg_mask,
        }

    else:
        raise ValueError(f"Unsupported model_type: {model_type}")

    return pred_embeds, pred_mask, gen_kwargs


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--pipeline', default='sdxl',
                        choices=['sdxl', 'dreamshaper', 'pixart_sigma'], type=str)
    parser.add_argument('--prompt', default='A banana on the left of an apple.', type=str)
    parser.add_argument('--pretrained-path', type=str, required=True)
    parser.add_argument('--inference-step', default=20, type=int)
    parser.add_argument('--cfg', default=4.5, type=float)
    parser.add_argument('--N', default=100, type=int)
    parser.add_argument('--B', default=1, type=int)
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--size', default=1024, type=int)

    args = parser.parse_args()
    return args


def main(args):
    dtype = torch.float16
    device = torch.device('cuda')

    if args.pipeline == 'sdxl':
        pipe = StableDiffusionXLPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0",
            variant="fp16", use_safetensors=True,
            torch_dtype=dtype).to(device)

    elif args.pipeline == 'dreamshaper':
        pipe = StableDiffusionXLPipeline.from_pretrained(
            "lykon/dreamshaper-xl-v2-turbo",
            variant="fp16",
            torch_dtype=dtype).to(device)

    elif args.pipeline == 'pixart_sigma':
        pipe = PixArtSigmaPipeline.from_pretrained(
            "PixArt-alpha/PixArt-Sigma-XL-2-1024-MS",
            torch_dtype=dtype).to(device)

    dims = get_dims(args.pipeline)

    predictor, norm_info = load_predictor(args.pretrained_path, device=device)

    pred_embeds, pred_mask, gen_kwargs = encode_prompt(
        pipe, args.prompt, args.pipeline, device,
    )

    generator = torch.Generator(device=device).manual_seed(args.seed)
    denoiser = pipe.unet if hasattr(pipe, "unet") else pipe.transformer
    noises = generate_noise_candidates(
        num_candidates=args.N,
        latent_shape=dims["latent_shape"],
        device=device,
        dtype=denoiser.dtype,
        generator=generator,
    )

    selected_noises, scores = select_top_k_noise(
        predictor=predictor,
        noises=noises,
        prompt_embeds=pred_embeds,
        prompt_mask=pred_mask,
        num_select=args.B,
    )

    B = selected_noises.shape[0]
    expanded_kwargs = {}
    for k, v in gen_kwargs.items():
        if isinstance(v, torch.Tensor) and v.dim() >= 2:
            expanded_kwargs[k] = v.expand(B, *[-1] * (v.dim() - 1))
        else:
            expanded_kwargs[k] = v

    # generate images with PAINE-selected noise
    paine_result = pipe(
        prompt=None,
        **expanded_kwargs,
        latents=selected_noises,
        height=args.size,
        width=args.size,
        num_images_per_prompt=1,
        num_inference_steps=args.inference_step,
        guidance_scale=args.cfg,
    )

    for i, img in enumerate(paine_result.images):
        img.save(f"{args.pipeline}_{args.prompt}_paine_{i:02d}.png")

    # generate images with standard random noise
    latent = torch.randn(1, *dims["latent_shape"], dtype=dtype, device=device)
    standard_result = pipe(
        prompt=args.prompt,
        height=args.size,
        width=args.size,
        num_inference_steps=args.inference_step,
        guidance_scale=args.cfg,
        latents=latent,
    )

    standard_result.images[0].save(f"{args.pipeline}_{args.prompt}_standard.png")


if __name__ == '__main__':
    args = get_args()
    main(args)
