"""
Verify that ScorePredictor loads and produces valid outputs.

Runs on CPU with dummy inputs (no GPU or pipeline needed).

Usage:
    python scripts/verify_predictor.py
    python scripts/verify_predictor.py --config pnm/configs/srcc_champion.yaml
"""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from pnm.models import ScorePredictor
from pnm.inference import load_predictor, generate_noise_candidates, select_top_k_noise


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Verify predictor loading and inference")
    parser.add_argument("--config", type=str, default="pnm/configs/ndcg_champion.yaml")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Direct checkpoint path (alternative to --config)")
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    print(f"{'='*50}")
    print("PNM Predictor Verification")
    print(f"{'='*50}")

    # Test 1: Load model
    print("\n[1] Loading predictor...")
    if args.checkpoint:
        model, norm_info = load_predictor(args.checkpoint, device=args.device)
        print(f"  Loaded from checkpoint: {args.checkpoint}")
    else:
        model, norm_info = ScorePredictor.from_config(args.config, device=args.device)
        print(f"  Loaded from config: {args.config}")

    print(f"  num_heads: {model.num_heads}")
    print(f"  normalization: {norm_info}")
    param_count = sum(p.numel() for p in model.parameters())
    print(f"  parameters: {param_count:,}")

    # Test 2: Forward pass with dummy inputs
    print("\n[2] Running forward pass (dummy inputs)...")
    batch_size = 4
    noise = torch.randn(batch_size, 4, 64, 64, device=args.device)
    prompt_embeds = torch.randn(batch_size, 120, 4096, device=args.device)
    prompt_mask = torch.ones(batch_size, 120, device=args.device)

    scores = model.predict(prompt_embeds, noise, prompt_mask)
    print(f"  Input:  noise={noise.shape}, embeds={prompt_embeds.shape}")
    print(f"  Output: scores={scores.shape}")
    print(f"  Values: {scores.squeeze().tolist()}")

    # Test 3: Noise selection
    print("\n[3] Testing noise selection...")
    N, B = 20, 3
    noises = generate_noise_candidates(N, (4, 64, 64), device=args.device, dtype=torch.float32)
    selected = select_top_k_noise(
        predictor=model,
        noises=noises,
        prompt_embeds=prompt_embeds[:1],
        prompt_mask=prompt_mask[:1],
        num_select=B,
    )
    print(f"  Generated {N} candidates, selected top {B}")
    print(f"  Selected shape: {selected.shape}")

    print(f"\n{'='*50}")
    print("All checks passed!")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
