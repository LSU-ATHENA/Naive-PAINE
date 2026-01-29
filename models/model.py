"""
Score Predictor model for PixArt.

Combines noise encoder and text encoder with a fusion MLP head.
Supports single-head (one target) or multi-head (all 3 targets) prediction.
"""

import torch
import torch.nn as nn
import yaml
from pathlib import Path
from typing import Optional, Dict, Any

class ScorePredictor(nn.Module):
    """
    Main predictor: (noise, prompt) -> score(s)

    Architecture:
        noise_encoder: [B, 4, 64, 64] -> [B, noise_dim]
        text_encoder: [B, 120, 4096] + mask -> [B, text_dim]
        fusion: concat -> MLP -> [B, num_heads]
    """

    def __init__(
        self,
        noise_encoder: nn.Module,
        text_encoder: nn.Module,
        dropout: float = 0.1,
        num_heads: int = 1,
    ):
        """
        Args:
            noise_encoder: Noise encoder module
            text_encoder: Text encoder module
            dropout: Dropout rate in fusion MLP
            num_heads: Number of output heads
                - 1: Single target mode (default)
                - 3: Multi-head mode (hpsv2, image_reward, clip_score)
        """
        super().__init__()

        self.noise_encoder = noise_encoder
        self.text_encoder = text_encoder
        self.num_heads = num_heads
        self.dropout = dropout

        # Get output dimensions from encoders
        noise_dim = noise_encoder.output_dim
        text_dim = text_encoder.output_dim
        fusion_dim = noise_dim + text_dim

        # Fusion MLP head (shared backbone)
        self.fusion_backbone = nn.Sequential(
            nn.Linear(fusion_dim, 512),
            nn.SiLU(),
            nn.Dropout(dropout),

            nn.Linear(512, 256),
            nn.SiLU(),
            nn.Dropout(dropout),

            nn.Linear(256, 64),
            nn.SiLU(),
        )

        # Output head(s)
        if num_heads == 1:
            self.head = nn.Linear(64, 1)
        else:
            # Separate heads for each target (hpsv2, image_reward, clip_score)
            self.heads = nn.ModuleList([nn.Linear(64, 1) for _ in range(num_heads)])

    def forward(
        self,
        noise: torch.Tensor,
        prompt_embeds: torch.Tensor,
        prompt_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            noise: [B, 4, 64, 64] - initial latent noise
            prompt_embeds: [B, 120, 4096] - T5 text embeddings
            prompt_mask: [B, 120] - attention mask (1=valid, 0=padding)
        Returns:
            [B, 1] for single-head mode
            [B, num_heads] for multi-head mode
        """
        noise_feat = self.noise_encoder(noise)
        text_feat = self.text_encoder(prompt_embeds, prompt_mask)

        combined = torch.cat([noise_feat, text_feat], dim=1)
        backbone_out = self.fusion_backbone(combined)

        if self.num_heads == 1:
            return self.head(backbone_out)
        else:
            # Stack outputs from all heads
            outputs = [head(backbone_out) for head in self.heads]
            return torch.cat(outputs, dim=1)  # [B, num_heads]

    @classmethod
    def from_config(cls, config_path: str, device: str = None):
        """
        Load a predictor from a YAML config file.

        Args:
            config_path: Path to YAML config file
            device: Override device (optional, defaults to config or 'cuda')

        Returns:
            Tuple of (model, normalization_info)
        """
        # Load YAML config
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        pred_config = config['predictor']
        arch = pred_config['architecture']
        weights = pred_config['weights']

        # Determine device
        device = device or weights.get('device', 'cuda')

        # Load checkpoint
        checkpoint_path = weights['checkpoint_path']
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

        # Get model config from checkpoint (overrides YAML if present)
        model_config = checkpoint.get('model_config', {})

        # Build encoders
        from .text_encoders import get_text_encoder
        from .noise_encoders import get_noise_encoder

        text_enc_name = model_config.get('text_enc', arch['text_encoder'])
        noise_enc_name = model_config.get('noise_enc', arch['noise_encoder'])
        dropout = model_config.get('dropout', arch.get('dropout', 0.1))
        num_heads = model_config.get('num_heads', arch.get('num_heads', 1))

        text_encoder = get_text_encoder(text_enc_name)
        noise_encoder = get_noise_encoder(noise_enc_name)

        # Create model
        model = cls(
            noise_encoder=noise_encoder,
            text_encoder=text_encoder,
            dropout=dropout,
            num_heads=num_heads
        )

        # Load weights
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        model.eval()

        # Get normalization info
        normalization_info = checkpoint.get('normalization', {})

        return model, normalization_info

    @torch.no_grad()
    def predict(self, prompt_embeds, noise, prompt_mask=None):
        """
        Convenience method for inference.

        Args:
            prompt_embeds: [B, seq_len, embed_dim] text embeddings
            noise: [B, 4, 64, 64] latent noise
            prompt_mask: [B, seq_len] attention mask (optional)

        Returns:
            scores: [B, num_heads] predicted scores
        """
        was_training = self.training
        self.eval()

        if prompt_mask is None:
            prompt_mask = torch.ones(
                prompt_embeds.shape[:2],
                device=prompt_embeds.device
            )

        scores = self.forward(noise, prompt_embeds, prompt_mask)

        if was_training:
            self.train()

        return scores

    def save(
        self,
        path: str,
        normalization: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Save the predictor to a checkpoint file.

        Args:
            path: Path to save the checkpoint
            normalization: Optional normalization info to include

        The checkpoint includes:
            - model_state_dict: Model weights
            - model_config: Architecture configuration for reconstruction
            - normalization: Score normalization parameters (if provided)
        """
        checkpoint = {
            'model_state_dict': self.state_dict(),
            'model_config': {
                'text_enc': self.text_encoder.__class__.__name__.replace('TextEncoder', '').lower(),
                'noise_enc': self.noise_encoder.__class__.__name__.replace('NoiseEncoder', '').lower(),
                'dropout': self.dropout,
                'num_heads': self.num_heads,
            },
            'normalization': normalization or {},
        }
        torch.save(checkpoint, path)
