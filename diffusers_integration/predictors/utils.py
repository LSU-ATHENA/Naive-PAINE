"""
Utility functions for predictor-based noise selection.
"""

import torch
from typing import Optional, Tuple, Union


def generate_noise_candidates(
    num_candidates: int,
    latent_shape: Tuple[int, ...] = (4, 64, 64),
    device: str = "cuda",
    dtype: torch.dtype = torch.float16,
    generator: Optional[Union[torch.Generator, list]] = None,
) -> torch.Tensor:
    """
    Generate N random noise candidates for predictor scoring.

    Args:
        num_candidates: Number of noise samples to generate
        latent_shape: Shape of each latent (C, H, W)
        device: Device to create tensors on
        dtype: Data type for noise tensors
        generator: Optional random generator for reproducibility

    Returns:
        Tensor of shape [num_candidates, C, H, W]
    """
    shape = (num_candidates, *latent_shape)

    if generator is not None:
        # Handle single generator or list
        if isinstance(generator, list):
            generator = generator[0] if generator else None

    noises = torch.randn(
        shape,
        generator=generator,
        device=device,
        dtype=dtype,
    )

    return noises


def select_top_k_noise(
    predictor: torch.nn.Module,
    noises: torch.Tensor,
    prompt_embeds: torch.Tensor,
    prompt_mask: torch.Tensor,
    num_select: int = 1,
    head_index: int = 0,
    batch_size: int = 64,
) -> torch.Tensor:
    """
    Score noise candidates and select top K.

    Args:
        predictor: ScorePredictor model
        noises: [N, C, H, W] noise candidates
        prompt_embeds: [1, seq_len, embed_dim] text embeddings
        prompt_mask: [1, seq_len] attention mask
        num_select: Number of top candidates to return
        head_index: Which prediction head to use (0=hpsv2, 1=image_reward, 2=clip_score)
        batch_size: Process candidates in batches for memory efficiency

    Returns:
        Selected noise tensors [num_select, C, H, W]
    """
    num_candidates = noises.shape[0]
    device = noises.device

    # Ensure prompt embeddings are on the same device
    prompt_embeds = prompt_embeds.to(device)
    prompt_mask = prompt_mask.to(device)

    # Collect all scores
    all_scores = []

    with torch.no_grad():
        for i in range(0, num_candidates, batch_size):
            batch_noises = noises[i:i + batch_size]
            batch_size_actual = batch_noises.shape[0]

            # Expand prompt embeddings to match batch
            prompt_embeds_exp = prompt_embeds.expand(batch_size_actual, -1, -1)
            prompt_mask_exp = prompt_mask.expand(batch_size_actual, -1)

            # Convert to float32 for predictor (model weights are float32)
            batch_noises_f32 = batch_noises.float()
            prompt_embeds_f32 = prompt_embeds_exp.float()

            # Get predictions
            scores = predictor.predict(prompt_embeds_f32, batch_noises_f32, prompt_mask_exp)

            # Handle multi-head output
            if scores.dim() == 2 and scores.shape[1] > 1:
                scores = scores[:, head_index]
            else:
                scores = scores.squeeze(-1)

            all_scores.append(scores)

    # Concatenate all scores
    all_scores = torch.cat(all_scores, dim=0)

    # Select top K
    top_indices = torch.topk(all_scores, min(num_select, num_candidates)).indices
    selected_noises = noises[top_indices]

    return selected_noises


def compute_noise_statistics(
    predictor: torch.nn.Module,
    prompt_embeds: torch.Tensor,
    prompt_mask: torch.Tensor,
    num_samples: int = 1000,
    latent_shape: Tuple[int, ...] = (4, 64, 64),
    device: str = "cuda",
    dtype: torch.dtype = torch.float16,
) -> dict:
    """
    Compute statistics of predicted scores over random noise samples.

    Useful for understanding the score distribution for a given prompt.

    Args:
        predictor: ScorePredictor model
        prompt_embeds: [1, seq_len, embed_dim] text embeddings
        prompt_mask: [1, seq_len] attention mask
        num_samples: Number of random samples to evaluate
        latent_shape: Shape of latent noise
        device: Device for computation
        dtype: Data type for noise

    Returns:
        Dictionary with mean, std, min, max, median of scores
    """
    noises = generate_noise_candidates(
        num_candidates=num_samples,
        latent_shape=latent_shape,
        device=device,
        dtype=dtype,
    )

    all_scores = []
    batch_size = 64

    with torch.no_grad():
        for i in range(0, num_samples, batch_size):
            batch_noises = noises[i:i + batch_size]
            batch_size_actual = batch_noises.shape[0]

            prompt_embeds_exp = prompt_embeds.expand(batch_size_actual, -1, -1).to(device)
            prompt_mask_exp = prompt_mask.expand(batch_size_actual, -1).to(device)

            # Convert to float32 for predictor
            batch_noises_f32 = batch_noises.float()
            prompt_embeds_f32 = prompt_embeds_exp.float()

            scores = predictor.predict(prompt_embeds_f32, batch_noises_f32, prompt_mask_exp)

            if scores.dim() == 2:
                scores = scores[:, 0]  # Use first head
            else:
                scores = scores.squeeze(-1)

            all_scores.append(scores.cpu())

    all_scores = torch.cat(all_scores, dim=0)

    return {
        "mean": float(all_scores.mean()),
        "std": float(all_scores.std()),
        "min": float(all_scores.min()),
        "max": float(all_scores.max()),
        "median": float(all_scores.median()),
        "num_samples": num_samples,
    }
