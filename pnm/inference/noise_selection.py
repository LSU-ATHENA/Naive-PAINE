"""Noise candidate generation and selection for PNM predictor."""

import torch
from typing import Optional, Tuple, Union


def generate_noise_candidates(
    num_candidates: int,
    latent_shape: Tuple[int, ...] = (4, 64, 64),
    device: str = "cuda",
    dtype: torch.dtype = torch.float16,
    generator: Optional[Union[torch.Generator, list]] = None,
) -> torch.Tensor:
    """Generate N random noise candidates.

    Args:
        num_candidates: Number of noise samples (N)
        latent_shape: Shape of each latent (C, H, W)
        device: Device to create tensors on
        dtype: Data type for noise tensors
        generator: Random generator for reproducibility

    Returns:
        [N, C, H, W] noise tensor
    """
    if isinstance(generator, list):
        generator = generator[0] if generator else None

    return torch.randn(
        (num_candidates, *latent_shape),
        generator=generator,
        device=device,
        dtype=dtype,
    )


def select_top_k_noise(
    predictor: torch.nn.Module,
    noises: torch.Tensor,
    prompt_embeds: torch.Tensor,
    prompt_mask: torch.Tensor,
    num_select: int = 1,
    head_index: int = 0,
    batch_size: int = 64,
) -> torch.Tensor:
    """Score noise candidates with predictor and return top K.

    Args:
        predictor: ScorePredictor model
        noises: [N, C, H, W] noise candidates
        prompt_embeds: [1, seq_len, embed_dim] text embeddings
        prompt_mask: [1, seq_len] attention mask
        num_select: Number of top candidates to return (B)
        head_index: Which head to use (0=hpsv2, 1=image_reward, 2=clip_score)
        batch_size: Batch size for scoring (memory efficiency)

    Returns:
        [B, C, H, W] selected noise tensors
    """
    num_candidates = noises.shape[0]
    device = noises.device
    prompt_embeds = prompt_embeds.to(device)
    prompt_mask = prompt_mask.to(device)

    all_scores = []

    with torch.no_grad():
        for i in range(0, num_candidates, batch_size):
            batch = noises[i:i + batch_size]
            bs = batch.shape[0]

            # Expand prompt to match batch
            pe = prompt_embeds.expand(bs, -1, -1).float()
            pm = prompt_mask.expand(bs, -1)

            scores = predictor.predict(pe, batch.float(), pm)

            # Extract score from correct head
            if scores.dim() == 2 and scores.shape[1] > 1:
                scores = scores[:, head_index]
            else:
                scores = scores.squeeze(-1)

            all_scores.append(scores)

    all_scores = torch.cat(all_scores, dim=0)
    top_indices = torch.topk(all_scores, min(num_select, num_candidates)).indices
    return noises[top_indices]
