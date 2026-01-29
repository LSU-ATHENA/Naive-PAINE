"""
PixArt Score Predictor - Model Loader

This module provides utilities to load a trained score predictor model
from a checkpoint file. The checkpoint is self-contained with all
configuration needed to reconstruct the model.

Usage:
    from loader import load_predictor, denormalize_prediction

    # Load model
    model, norm_info = load_predictor('best_model.pth')

    # Run inference (after extracting noise and prompt_embeds from pipeline)
    with torch.no_grad():
        pred_normalized = model(noise, prompt_embeds, prompt_mask)

    # Denormalize to get actual score
    pred_score = denormalize_prediction(pred_normalized, norm_info)
"""

import torch
import torch.nn as nn
from typing import Tuple, Dict, Any

# Import model factory - adjust path if needed
from pnm.models import get_model


def load_predictor(
    checkpoint_path: str,
    device: str = 'cuda',
) -> Tuple[nn.Module, Dict[str, Any]]:
    """
    Load a PixArt Score Predictor from a checkpoint file.

    The checkpoint contains:
    - model_state_dict: The trained weights
    - model_config: Architecture configuration (noise_enc, text_enc, dropout, num_heads)
    - normalization: Stats for converting predictions back to original scale

    Args:
        checkpoint_path: Path to the checkpoint file (best_model.pth)
        device: Device to load the model on ('cuda' or 'cpu')

    Returns:
        model: The loaded model in eval mode
        normalization: Dict with denormalization info
            For multi_target=True:
                {'multi_target': True, 'target_stats': {
                    'hpsv2': {'mean': float, 'std': float},
                    'image_reward': {'mean': float, 'std': float},
                    'clip_score': {'mean': float, 'std': float},
                }}
            For multi_target=False:
                {'multi_target': False, 'target': str, 'y_mean': float, 'y_std': float}

    Example:
        >>> model, norm = load_predictor('experiments/best_run/best_model.pth')
        >>> print(f"Noise encoder: {norm.get('noise_enc', 'see checkpoint')}")
        >>> print(f"Multi-target: {norm['multi_target']}")
    """
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Extract model configuration
    cfg = checkpoint['model_config']

    # Reconstruct the model
    model = get_model(
        noise_enc=cfg['noise_enc'],
        text_enc=cfg['text_enc'],
        dropout=cfg['dropout'],
        num_heads=cfg['num_heads'],
    )

    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    # Return model and normalization info
    return model, checkpoint['normalization']


def denormalize_prediction(
    pred_normalized: torch.Tensor,
    normalization: Dict[str, Any],
    target: str = 'hpsv2',
) -> torch.Tensor:
    """
    Convert normalized predictions back to original scale.

    The model outputs z-score normalized predictions. Use this function
    to convert them to actual scores (e.g., HPSv2 scores in [0, 1] range).

    Args:
        pred_normalized: Model output tensor
            For multi_target: [B, 3] where dims are [hpsv2, image_reward, clip_score]
            For single_target: [B, 1] or [B]
        normalization: The normalization dict from load_predictor()
        target: Which target to denormalize (for multi_target mode)
                Options: 'hpsv2', 'image_reward', 'clip_score'

    Returns:
        Denormalized predictions in original scale

    Example:
        >>> pred_norm = model(noise, prompt_embeds, prompt_mask)  # [B, 3]
        >>> hpsv2_score = denormalize_prediction(pred_norm[:, 0], norm, 'hpsv2')
        >>> image_reward = denormalize_prediction(pred_norm[:, 1], norm, 'image_reward')
    """
    if normalization['multi_target']:
        stats = normalization['target_stats'][target]
        mean, std = stats['mean'], stats['std']
    else:
        mean = normalization['y_mean']
        std = normalization['y_std']

    return pred_normalized * std + mean


def get_checkpoint_info(checkpoint_path: str) -> Dict[str, Any]:
    """
    Get information about a checkpoint without loading the full model.

    Useful for inspecting what's in a checkpoint file.

    Args:
        checkpoint_path: Path to the checkpoint file

    Returns:
        Dict with keys: 'model_config', 'normalization', 'param_count'

    Example:
        >>> info = get_checkpoint_info('best_model.pth')
        >>> print(f"Noise encoder: {info['model_config']['noise_enc']}")
        >>> print(f"Text encoder: {info['model_config']['text_enc']}")
        >>> print(f"Num heads: {info['model_config']['num_heads']}")
    """
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

    # Count parameters
    param_count = sum(
        p.numel() for p in checkpoint['model_state_dict'].values()
    )

    return {
        'model_config': checkpoint['model_config'],
        'normalization': checkpoint['normalization'],
        'param_count': param_count,
    }


# =============================================================================
# Integration Guide for PixartAlphaPipeline
# =============================================================================
#
# The undergraduate integrator needs to:
#
# 1. At Load Time (__init__):
#    - Add argument: predictor_checkpoint_path
#    - Call load_predictor() and store model in self.predictor
#
# 2. At Run Time (__call__):
#    - Add argument: N_steps (number of diffusion steps to predict for)
#    - Extract noise latent and prompt_embeds from pipeline
#    - Call self.predictor(noise, prompt_embeds, prompt_mask)
#    - Denormalize to get actual score
#
# Example Integration:
#
# class PixartAlphaPipelineWithPredictor(PixartAlphaPipeline):
#     def __init__(self, ..., predictor_checkpoint_path=None):
#         super().__init__(...)
#         if predictor_checkpoint_path:
#             self.predictor, self.predictor_norm = load_predictor(
#                 predictor_checkpoint_path,
#                 device=self.device
#             )
#         else:
#             self.predictor = None
#
#     def predict_score(self, prompt, noise_latent):
#         """Predict quality score for given noise and prompt."""
#         if self.predictor is None:
#             raise ValueError("Predictor not loaded")
#
#         # Encode prompt (reuse pipeline's text encoder)
#         prompt_embeds, prompt_mask = self.encode_prompt(prompt)
#
#         # Get prediction
#         with torch.no_grad():
#             pred_norm = self.predictor(noise_latent, prompt_embeds, prompt_mask)
#
#         # Denormalize
#         score = denormalize_prediction(pred_norm, self.predictor_norm)
#         return score
#
# =============================================================================


if __name__ == '__main__':
    # Example usage - run this to verify a checkpoint
    import sys

    if len(sys.argv) < 2:
        print("Usage: python loader.py <checkpoint_path>")
        print("\nExample: python loader.py experiments/best_run/best_model.pth")
        sys.exit(1)

    ckpt_path = sys.argv[1]
    print(f"Loading checkpoint: {ckpt_path}")

    info = get_checkpoint_info(ckpt_path)

    print(f"\n{'='*50}")
    print("Checkpoint Information")
    print(f"{'='*50}")
    print(f"\nModel Config:")
    for k, v in info['model_config'].items():
        print(f"  {k}: {v}")

    print(f"\nNormalization:")
    norm = info['normalization']
    print(f"  multi_target: {norm['multi_target']}")
    if norm['multi_target']:
        print(f"  targets: {list(norm['target_stats'].keys())}")
        for tgt, stats in norm['target_stats'].items():
            print(f"    {tgt}: mean={stats['mean']:.4f}, std={stats['std']:.4f}")
    else:
        print(f"  target: {norm['target']}")
        print(f"  y_mean: {norm['y_mean']:.4f}")
        print(f"  y_std: {norm['y_std']:.4f}")

    print(f"\nParameter count: {info['param_count']:,}")
    print(f"{'='*50}")
