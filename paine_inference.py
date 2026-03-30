import argparse
from pathlib import Path

import torch
from diffusers import StableDiffusionXLPipeline, HunyuanDiTPipeline

from predictor.inference.loader import load_predictor, denormalize_prediction
from predictor.inference.noise_selection import generate_noise_candidates, select_top_k_noise
from predictor.configs.model_dims import get_dims

GENERATION_DEFAULTS = {
    'sdxl':         {'steps': 50, 'guidance_scale': 5.5},
    'hunyuan_dit':  {'steps': 50, 'guidance_scale': 5.0},
}


def encode_prompt(pipe, prompt, model_type, device):
    if model_type == 'sdxl':
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
            'prompt_embeds': embeds,
            'negative_prompt_embeds': neg_embeds,
            'pooled_prompt_embeds': pooled,
            'negative_pooled_prompt_embeds': neg_pooled,
        }

    elif model_type == 'hunyuan_dit':
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

    else:
        raise ValueError(f"Unsupported model_type: {model_type}")

    return pred_embeds, pred_mask, gen_kwargs


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--pipeline', default='sdxl',
                        choices=['sdxl', 'hunyuan_dit'], type=str)
    parser.add_argument('--prompt', type=str, required=True,
                        help='Text prompt for image generation')
    parser.add_argument('--pretrained-path', type=str, required=True)
    parser.add_argument('--inference-step', default=None, type=int)
    parser.add_argument('--cfg', default=None, type=float)
    parser.add_argument('--N', default=100, type=int)
    parser.add_argument('--B', default=1, type=int)
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--output-dir', default='output', type=str)

    return parser.parse_args()


def run_single_prompt(pipe, predictor, norm_info, prompt, args, dims, device, output_dir):
    pred_embeds, pred_mask, gen_kwargs = encode_prompt(
        pipe, prompt, args.pipeline, device,
    )

    generator = torch.Generator(device=device).manual_seed(args.seed)
    denoiser = pipe.unet if hasattr(pipe, 'unet') else pipe.transformer
    noises = generate_noise_candidates(
        num_candidates=args.N,
        latent_shape=dims['latent_shape'],
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

    raw_scores = denormalize_prediction(scores, norm_info)
    target_name = norm_info.get('target', 'score')

    print(f'Prompt: {prompt}')
    print(f'  PAINE predicted {target_name}: {[f"{s:.4f}" for s in raw_scores.tolist()]}')

    B = selected_noises.shape[0]
    expanded_kwargs = {}
    for k, v in gen_kwargs.items():
        if isinstance(v, torch.Tensor) and v.dim() >= 2:
            expanded_kwargs[k] = v.expand(B, *[-1] * (v.dim() - 1))
        else:
            expanded_kwargs[k] = v

    paine_result = pipe(
        prompt=None if gen_kwargs else prompt,
        **expanded_kwargs,
        latents=selected_noises,
        height=1024,
        width=1024,
        num_images_per_prompt=1,
        num_inference_steps=args.inference_step,
        guidance_scale=args.cfg,
    )

    for i, img in enumerate(paine_result.images):
        path = output_dir / f'paine_{i:02d}.jpg'
        img.save(path)


def main(args):
    defaults = GENERATION_DEFAULTS.get(args.pipeline, {'steps': 50, 'guidance_scale': 5.5})
    if args.inference_step is None:
        args.inference_step = defaults['steps']
    if args.cfg is None:
        args.cfg = defaults['guidance_scale']

    dtype = torch.float16
    device = torch.device('cuda')

    if args.pipeline == 'sdxl':
        pipe = StableDiffusionXLPipeline.from_pretrained(
            'stabilityai/stable-diffusion-xl-base-1.0',
            variant='fp16', use_safetensors=True,
            torch_dtype=dtype).to(device)

    elif args.pipeline == 'hunyuan_dit':
        pipe = HunyuanDiTPipeline.from_pretrained(
            'Tencent-Hunyuan/HunyuanDiT-v1.2-Diffusers',
            torch_dtype=dtype).to(device)

    dims = get_dims(args.pipeline)
    predictor, norm_info = load_predictor(args.pretrained_path, device=device)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    run_single_prompt(pipe, predictor, norm_info, args.prompt, args, dims, device, output_dir)

    print(f'Done. Output: {output_dir}/')


if __name__ == '__main__':
    args = get_args()
    main(args)
