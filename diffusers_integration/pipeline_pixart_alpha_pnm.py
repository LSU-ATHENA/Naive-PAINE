"""
PixArt-Alpha Pipeline with PNM Predictor Support.

This module provides a PixArt-Alpha pipeline that integrates the PNM
ScorePredictor for noise-optimized image generation.
"""

import torch
from typing import Callable, List, Optional, Tuple, Union

from diffusers import PixArtAlphaPipeline
from diffusers.utils import logging

from .predictors import PredictorMixin

logger = logging.get_logger(__name__)


class PixArtAlphaPNMPipeline(PixArtAlphaPipeline, PredictorMixin):
    """
    PixArt-Alpha pipeline with PNM predictor integration.

    Extends the standard PixArtAlphaPipeline with the ability to use
    a trained ScorePredictor to select optimal initial noise.

    Example:
        ```python
        from pnm.diffusers_integration import PixArtAlphaPNMPipeline

        pipe = PixArtAlphaPNMPipeline.from_pretrained(
            "PixArt-alpha/PixArt-XL-2-512x512",
            torch_dtype=torch.float16
        ).to("cuda")

        # Load predictor
        pipe.load_predictor("pnm/configs/ndcg_champion.yaml")

        # Generate with predictor-selected noise
        images = pipe(
            prompt="A cat wearing a hat",
            use_predictor=True,
            num_candidates=100,
            num_images_per_prompt=4,
        ).images
        ```
    """

    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        negative_prompt: str = "",
        num_inference_steps: int = 20,
        timesteps: List[int] = None,
        sigmas: List[float] = None,
        guidance_scale: float = 4.5,
        num_images_per_prompt: int = 1,
        height: int = None,
        width: int = None,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.Tensor] = None,
        prompt_embeds: Optional[torch.Tensor] = None,
        prompt_attention_mask: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_attention_mask: Optional[torch.Tensor] = None,
        output_type: str = "pil",
        return_dict: bool = True,
        callback: Optional[Callable[[int, int, torch.Tensor], None]] = None,
        callback_steps: int = 1,
        clean_caption: bool = True,
        use_resolution_binning: bool = True,
        max_sequence_length: int = 120,
        # PNM-specific parameters
        use_predictor: bool = False,
        num_candidates: int = 100,
        predictor_head_index: int = 0,
        **kwargs,
    ):
        """
        Generate images with optional predictor-based noise selection.

        Args:
            prompt: The prompt(s) to guide image generation
            negative_prompt: The negative prompt to guide generation
            num_inference_steps: Number of denoising steps
            timesteps: Custom timesteps (overrides num_inference_steps)
            sigmas: Custom sigmas (overrides timesteps)
            guidance_scale: CFG scale
            num_images_per_prompt: Number of images to generate per prompt
            height: Image height (defaults to 512)
            width: Image width (defaults to 512)
            eta: DDIM eta parameter
            generator: Random generator for reproducibility
            latents: Pre-generated latents (overrides noise generation)
            prompt_embeds: Pre-computed prompt embeddings
            prompt_attention_mask: Attention mask for prompt embeddings
            negative_prompt_embeds: Pre-computed negative prompt embeddings
            negative_prompt_attention_mask: Attention mask for negative embeddings
            output_type: Output format ("pil", "latent", "pt", "np")
            return_dict: Whether to return a dict or tuple
            callback: Callback function for each step
            callback_steps: Frequency of callback calls
            clean_caption: Whether to clean the caption
            use_resolution_binning: Whether to use resolution binning
            max_sequence_length: Maximum sequence length for text encoder

            use_predictor: Whether to use predictor for noise selection
            num_candidates: Number of noise candidates to evaluate (N)
            predictor_head_index: Which prediction head to use (0=hpsv2)

        Returns:
            Generated images and optionally other outputs
        """
        # Set default height/width
        # Get default sample size from transformer config or use 512 fallback
        default_size = getattr(self.transformer.config, 'sample_size', 64)  # 64 latent = 512 pixel
        height = height or default_size * self.vae_scale_factor
        width = width or default_size * self.vae_scale_factor

        # If use_predictor=True and no latents provided, use predictor to select noise
        if use_predictor and latents is None and self.has_predictor:
            (
                latents,
                prompt_embeds,
                prompt_attention_mask,
                negative_prompt_embeds,
                negative_prompt_attention_mask,
            ) = self._generate_predictor_latents(
                prompt=prompt,
                negative_prompt=negative_prompt,
                num_images_per_prompt=num_images_per_prompt,
                num_candidates=num_candidates,
                height=height,
                width=width,
                generator=generator,
                prompt_embeds=prompt_embeds,
                prompt_attention_mask=prompt_attention_mask,
                negative_prompt_embeds=negative_prompt_embeds,
                negative_prompt_attention_mask=negative_prompt_attention_mask,
                clean_caption=clean_caption,
                max_sequence_length=max_sequence_length,
                predictor_head_index=predictor_head_index,
            )
        elif use_predictor and not self.has_predictor:
            logger.warning(
                "use_predictor=True but no predictor loaded. "
                "Call load_predictor() first. Falling back to random noise."
            )

        # Call parent pipeline
        return super().__call__(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=num_inference_steps,
            timesteps=timesteps,
            sigmas=sigmas,
            guidance_scale=guidance_scale,
            num_images_per_prompt=num_images_per_prompt,
            height=height,
            width=width,
            eta=eta,
            generator=generator,
            latents=latents,
            prompt_embeds=prompt_embeds,
            prompt_attention_mask=prompt_attention_mask,
            negative_prompt_embeds=negative_prompt_embeds,
            negative_prompt_attention_mask=negative_prompt_attention_mask,
            output_type=output_type,
            return_dict=return_dict,
            callback=callback,
            callback_steps=callback_steps,
            clean_caption=clean_caption,
            use_resolution_binning=use_resolution_binning,
            max_sequence_length=max_sequence_length,
            **kwargs,
        )

    def _generate_predictor_latents(
        self,
        prompt: Union[str, List[str]],
        negative_prompt: str,
        num_images_per_prompt: int,
        num_candidates: int,
        height: int,
        width: int,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]],
        prompt_embeds: Optional[torch.Tensor],
        prompt_attention_mask: Optional[torch.Tensor],
        negative_prompt_embeds: Optional[torch.Tensor],
        negative_prompt_attention_mask: Optional[torch.Tensor],
        clean_caption: bool,
        max_sequence_length: int,
        predictor_head_index: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Generate latents using predictor to select best noise.

        This method encodes the prompt, generates N noise candidates,
        scores them with the predictor, and returns the top B along
        with the encoded embeddings to avoid double encoding.

        Returns:
            Tuple of (latents, prompt_embeds, prompt_attention_mask,
                      negative_prompt_embeds, negative_prompt_attention_mask)
        """
        device = self._execution_device
        dtype = self.transformer.dtype

        # Encode prompt if not provided
        if prompt_embeds is None:
            (
                prompt_embeds,
                prompt_attention_mask,
                negative_prompt_embeds,
                negative_prompt_attention_mask,
            ) = self.encode_prompt(
                prompt=prompt,
                negative_prompt=negative_prompt,
                do_classifier_free_guidance=self.do_classifier_free_guidance,
                num_images_per_prompt=1,  # Single embedding, we'll expand later
                device=device,
                clean_caption=clean_caption,
                max_sequence_length=max_sequence_length,
            )

        # For CFG, prompt_embeds is [2, seq_len, dim] - use positive (second half)
        if prompt_embeds.shape[0] == 2:
            prompt_embeds_for_pred = prompt_embeds[1:2]
            prompt_mask_for_pred = prompt_attention_mask[1:2] if prompt_attention_mask is not None else None
        else:
            prompt_embeds_for_pred = prompt_embeds
            prompt_mask_for_pred = prompt_attention_mask

        # Default mask if not provided
        if prompt_mask_for_pred is None:
            prompt_mask_for_pred = torch.ones(
                prompt_embeds_for_pred.shape[:2],
                device=device,
                dtype=torch.long,
            )

        # Calculate latent dimensions
        latent_height = height // self.vae_scale_factor
        latent_width = width // self.vae_scale_factor
        latent_shape = (4, latent_height, latent_width)

        # Select best noise using predictor
        selected_noise = self.select_best_noise(
            prompt_embeds=prompt_embeds_for_pred.to(dtype=torch.float32),
            prompt_mask=prompt_mask_for_pred,
            num_candidates=num_candidates,
            num_select=num_images_per_prompt,
            latent_shape=latent_shape,
            generator=generator,
            dtype=dtype,
            head_index=predictor_head_index,
        )

        # Scale by scheduler init noise sigma
        latents = selected_noise * self.scheduler.init_noise_sigma

        return latents, prompt_embeds, prompt_attention_mask, negative_prompt_embeds, negative_prompt_attention_mask

    @property
    def do_classifier_free_guidance(self) -> bool:
        """Check if CFG is enabled based on guidance scale."""
        return hasattr(self, '_guidance_scale') and self._guidance_scale > 1

    def generate_with_comparison(
        self,
        prompt: Union[str, List[str]],
        num_candidates: int = 100,
        num_images_per_prompt: int = 4,
        **kwargs,
    ) -> dict:
        """
        Generate images with both predictor and random noise for comparison.

        Args:
            prompt: The prompt(s) to guide image generation
            num_candidates: Number of noise candidates for predictor
            num_images_per_prompt: Number of images per method
            **kwargs: Additional arguments passed to __call__

        Returns:
            Dictionary with 'predictor' and 'random' image sets
        """
        if not self.has_predictor:
            raise RuntimeError("Predictor not loaded. Call load_predictor() first.")

        # Generate with predictor
        result_pred = self(
            prompt=prompt,
            use_predictor=True,
            num_candidates=num_candidates,
            num_images_per_prompt=num_images_per_prompt,
            **kwargs,
        )

        # Generate with random noise (same number of images)
        result_rand = self(
            prompt=prompt,
            use_predictor=False,
            num_images_per_prompt=num_images_per_prompt,
            **kwargs,
        )

        return {
            "predictor": result_pred.images,
            "random": result_rand.images,
            "prompt": prompt,
            "num_candidates": num_candidates,
        }
