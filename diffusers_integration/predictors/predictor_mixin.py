"""
Predictor Mixin for Diffusers Pipelines.

Provides methods to load and use PNM ScorePredictor for noise selection
in diffusion pipelines.
"""

import torch
from typing import Optional, Tuple, Union

from .utils import generate_noise_candidates, select_top_k_noise


class PredictorMixin:
    predictor = None
    predictor_norm_info = None
    predictor_device = None

    def load_predictor(
        self,
        config_path: str,
        device: Optional[str] = None,
    ) -> None:
        """
        Load ScorePredictor from YAML config.

        Args:
            config_path: Path to YAML config file containing predictor settings
            device: Device to load predictor on (defaults to pipeline device)
        """
        from pnm.models import ScorePredictor

        # Determine device
        if device is None:
            device = str(self.device) if hasattr(self, 'device') else 'cuda'

        self.predictor, self.predictor_norm_info = ScorePredictor.from_config(
            config_path, device=device
        )
        self.predictor_device = device

    def unload_predictor(self) -> None:
        """Unload predictor to free memory."""
        if self.predictor is not None:
            del self.predictor
            self.predictor = None
            self.predictor_norm_info = None
            torch.cuda.empty_cache()

    @property
    def has_predictor(self) -> bool:
        """Check if predictor is loaded."""
        return self.predictor is not None

    def select_best_noise(
        self,
        prompt_embeds: torch.Tensor,
        prompt_mask: torch.Tensor,
        num_candidates: int = 100,
        num_select: int = 1,
        latent_shape: Tuple[int, ...] = (4, 64, 64),
        generator: Optional[Union[torch.Generator, list]] = None,
        dtype: torch.dtype = torch.float16,
        head_index: int = 0,
    ) -> torch.Tensor:
        """
        Generate N noise candidates and return top B scored by predictor.

        Args:
            prompt_embeds: [B, seq_len, embed_dim] T5 text embeddings
            prompt_mask: [B, seq_len] attention mask (1=valid, 0=padding)
            num_candidates: Number of noise candidates to generate (N)
            num_select: Number of top candidates to select (B)
            latent_shape: Shape of latent noise (C, H, W)
            generator: Optional random generator for reproducibility
            dtype: Data type for generated noise
            head_index: Which prediction head to use for scoring (0=hpsv2)

        Returns:
            Selected noise tensors [num_select, C, H, W]
        """
        if self.predictor is None:
            raise RuntimeError("Predictor not loaded. Call load_predictor() first.")

        device = self.predictor_device or str(self.device)

        # Generate N noise candidates
        noises = generate_noise_candidates(
            num_candidates=num_candidates,
            latent_shape=latent_shape,
            device=device,
            dtype=dtype,
            generator=generator,
        )

        # Score and select top B
        selected = select_top_k_noise(
            predictor=self.predictor,
            noises=noises,
            prompt_embeds=prompt_embeds,
            prompt_mask=prompt_mask,
            num_select=num_select,
            head_index=head_index,
        )

        return selected

    def score_noise(
        self,
        noise: torch.Tensor,
        prompt_embeds: torch.Tensor,
        prompt_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Score a single noise tensor or batch of noise tensors.

        Args:
            noise: [B, C, H, W] noise tensor(s) to score
            prompt_embeds: [B, seq_len, embed_dim] text embeddings
            prompt_mask: [B, seq_len] attention mask

        Returns:
            scores: [B, num_heads] predicted scores
        """
        if self.predictor is None:
            raise RuntimeError("Predictor not loaded. Call load_predictor() first.")

        return self.predictor.predict(prompt_embeds, noise, prompt_mask)
