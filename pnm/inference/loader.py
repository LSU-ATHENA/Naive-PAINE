"""Load trained PNM predictor from checkpoint."""

import torch
import torch.nn as nn
from typing import Tuple, Dict, Any

from pnm.models import get_model


def load_predictor(
    checkpoint_path: str,
    device: str = "cuda",
) -> Tuple[nn.Module, Dict[str, Any]]:
    """Load a ScorePredictor from a checkpoint file.

    Args:
        checkpoint_path: Path to .pth checkpoint
        device: Device to load on ('cuda' or 'cpu')

    Returns:
        (model, normalization_info) tuple
    """
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    cfg = checkpoint["model_config"]
    model = get_model(
        noise_enc=cfg["noise_enc"],
        text_enc=cfg["text_enc"],
        dropout=cfg["dropout"],
        num_heads=cfg["num_heads"],
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    return model, checkpoint["normalization"]


def denormalize_prediction(
    pred_normalized: torch.Tensor,
    normalization: Dict[str, Any],
    target: str = "hpsv2",
) -> torch.Tensor:
    """Convert normalized predictions back to original scale.

    Args:
        pred_normalized: Model output (z-score normalized)
        normalization: Normalization dict from load_predictor()
        target: Which target to denormalize (for multi_target mode)

    Returns:
        Denormalized predictions
    """
    if normalization["multi_target"]:
        stats = normalization["target_stats"][target]
        mean, std = stats["mean"], stats["std"]
    else:
        mean = normalization["y_mean"]
        std = normalization["y_std"]

    return pred_normalized * std + mean


def get_checkpoint_info(checkpoint_path: str) -> Dict[str, Any]:
    """Inspect checkpoint without loading full model.

    Returns dict with 'model_config', 'normalization', 'param_count'.
    """
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    param_count = sum(p.numel() for p in checkpoint["model_state_dict"].values())

    return {
        "model_config": checkpoint["model_config"],
        "normalization": checkpoint["normalization"],
        "param_count": param_count,
    }
