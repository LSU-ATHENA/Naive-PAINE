import torch
import torch.nn as nn
from typing import Tuple, Dict, Any

from model import get_model
from model.config import MODEL_DIMS


def load_predictor(
    checkpoint_path: str,
    device: str = "cuda",
) -> Tuple[nn.Module, Dict[str, Any]]:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    cfg = checkpoint["model_config"]

    model_type = cfg.get("model_type", None)
    if model_type and model_type in MODEL_DIMS:
        dims = MODEL_DIMS[model_type]
        spatial_size = cfg.get("spatial_size", dims["spatial_size"])
        in_channels = cfg.get("in_channels", dims["latent_shape"][0])
        embed_dim = cfg.get("embed_dim", dims["embed_dim"])
        seq_len = cfg.get("seq_len", dims["seq_len"])
    else:
        spatial_size = cfg.get("spatial_size", 64)
        in_channels = cfg.get("in_channels", 4)
        embed_dim = cfg.get("embed_dim", 4096)
        seq_len = cfg.get("seq_len", 120)

    model = get_model(
        noise_enc=cfg["noise_enc"],
        text_enc=cfg["text_enc"],
        dropout=cfg["dropout"],
        num_heads=cfg.get("num_heads", 1),
        spatial_size=spatial_size,
        in_channels=in_channels,
        embed_dim=embed_dim,
        seq_len=seq_len,
        pos_encoding=cfg.get("pos_encoding", "none"),
    )

    state_dict = {k: v.float() for k, v in checkpoint["model_state_dict"].items()}
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    return model, checkpoint["normalization"]


def denormalize_prediction(
    pred_normalized: torch.Tensor,
    normalization: Dict[str, Any],
) -> torch.Tensor:

    mean = normalization["y_mean"]
    std = normalization["y_std"]
    return pred_normalized * std + mean


def get_checkpoint_info(checkpoint_path: str) -> Dict[str, Any]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    param_count = sum(p.numel() for p in checkpoint["model_state_dict"].values())

    return {
        "model_config": checkpoint["model_config"],
        "normalization": checkpoint["normalization"],
        "param_count": param_count,
    }