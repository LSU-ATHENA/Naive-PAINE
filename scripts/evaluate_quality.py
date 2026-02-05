#!/usr/bin/env python
"""
Evaluate image quality metrics for predictor vs random noise comparison.

Computes HPSv2, ImageReward, and CLIP scores for generated images
and provides statistical comparison.
"""

import os
import sys
import argparse
import json
from pathlib import Path
from typing import List, Dict

import torch
import numpy as np
from PIL import Image

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def load_metrics():
    """Load metric scorers."""
    import hpsv2
    import ImageReward as RM
    from torchmetrics.multimodal.clip_score import CLIPScore

    device = "cuda" if torch.cuda.is_available() else "cpu"

    scorer_ir = RM.load("ImageReward-v1.0")
    scorer_clip = CLIPScore(model_name_or_path="openai/clip-vit-base-patch16").to(device)

    return {
        "hpsv2": hpsv2,
        "image_reward": scorer_ir,
        "clip_score": scorer_clip,
        "device": device,
    }


def compute_scores(
    image: Image.Image,
    prompt: str,
    scorers: dict,
) -> Dict[str, float]:
    """Compute all quality metrics for an image."""
    device = scorers["device"]
    image_rgb = image.convert("RGB")

    with torch.inference_mode():
        # HPSv2
        hps = float(scorers["hpsv2"].score([image_rgb], prompt, hps_version="v2.1")[0])

        # ImageReward
        ir = float(scorers["image_reward"].score(prompt, image_rgb))

        # CLIP Score
        img_tensor = torch.from_numpy(np.array(image_rgb)).permute(2, 0, 1).unsqueeze(0).float().to(device)
        clip_raw = float(scorers["clip_score"](img_tensor, [prompt]).detach().cpu())
        clip_cos = clip_raw / 100.0  # Convert to cosine similarity

    return {
        "hpsv2": round(hps, 4),
        "image_reward": round(ir, 4),
        "clip_score": round(clip_cos, 4),
    }


def evaluate_images(
    image_paths: List[Path],
    prompt: str,
    scorers: dict,
    label: str,
) -> List[Dict]:
    """Evaluate a list of images."""
    results = []

    for path in image_paths:
        print(f"  Scoring: {path.name}")
        image = Image.open(path)
        scores = compute_scores(image, prompt, scorers)
        scores["image"] = str(path)
        scores["label"] = label
        results.append(scores)

    return results


def compute_statistics(results: List[Dict], metric: str) -> Dict:
    """Compute statistics for a metric across results."""
    values = [r[metric] for r in results]
    return {
        "mean": round(np.mean(values), 4),
        "std": round(np.std(values), 4),
        "min": round(np.min(values), 4),
        "max": round(np.max(values), 4),
        "n": len(values),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate quality metrics for generated images"
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        default="../output",
        help="Directory containing generated images"
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="A green dog and a red cat",
        help="Prompt used for generation"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output JSON file for results"
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir)

    # Find images
    pred_images = sorted(input_dir.glob("predictor_*.png"))
    rand_images = sorted(input_dir.glob("random_*.png"))

    if not pred_images:
        print(f"No predictor images found in {input_dir}")
        sys.exit(1)

    print("=" * 60)
    print("Quality Evaluation")
    print("=" * 60)
    print(f"Prompt: {args.prompt}")
    print(f"Predictor images: {len(pred_images)}")
    print(f"Random images: {len(rand_images)}")
    print("=" * 60)

    # Load metrics
    print("\nLoading metric scorers...")
    scorers = load_metrics()

    # Evaluate predictor images
    print("\nEvaluating predictor images...")
    pred_results = evaluate_images(pred_images, args.prompt, scorers, "predictor")

    # Evaluate random images
    rand_results = []
    if rand_images:
        print("\nEvaluating random images...")
        rand_results = evaluate_images(rand_images, args.prompt, scorers, "random")

    # Compute statistics
    metrics = ["hpsv2", "image_reward", "clip_score"]

    print("\n" + "=" * 60)
    print("Results Summary")
    print("=" * 60)

    summary = {
        "prompt": args.prompt,
        "predictor": {},
        "random": {},
        "improvement": {},
    }

    for metric in metrics:
        pred_stats = compute_statistics(pred_results, metric)
        summary["predictor"][metric] = pred_stats

        print(f"\n{metric.upper()}:")
        print(f"  Predictor: {pred_stats['mean']:.4f} +/- {pred_stats['std']:.4f}")

        if rand_results:
            rand_stats = compute_statistics(rand_results, metric)
            summary["random"][metric] = rand_stats

            improvement = pred_stats['mean'] - rand_stats['mean']
            pct_improvement = (improvement / abs(rand_stats['mean'])) * 100 if rand_stats['mean'] != 0 else 0

            summary["improvement"][metric] = {
                "absolute": round(improvement, 4),
                "percentage": round(pct_improvement, 2),
            }

            print(f"  Random:    {rand_stats['mean']:.4f} +/- {rand_stats['std']:.4f}")
            print(f"  Delta:     {improvement:+.4f} ({pct_improvement:+.1f}%)")

    # Per-image results
    print("\n" + "=" * 60)
    print("Per-Image Scores")
    print("=" * 60)

    print("\nPredictor images:")
    for r in pred_results:
        print(f"  {Path(r['image']).name}: HPSv2={r['hpsv2']:.4f}, IR={r['image_reward']:.4f}, CLIP={r['clip_score']:.4f}")

    if rand_results:
        print("\nRandom images:")
        for r in rand_results:
            print(f"  {Path(r['image']).name}: HPSv2={r['hpsv2']:.4f}, IR={r['image_reward']:.4f}, CLIP={r['clip_score']:.4f}")

    # Save results
    if args.output:
        output_path = Path(args.output)
        full_results = {
            "summary": summary,
            "predictor_images": pred_results,
            "random_images": rand_results,
        }
        with open(output_path, "w") as f:
            json.dump(full_results, f, indent=2)
        print(f"\nResults saved to: {output_path}")

    print("\n" + "=" * 60)
    print("Evaluation complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
