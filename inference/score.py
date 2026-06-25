# HPSv2 / HPSv3 / ImageReward / PickScore. Pass image paths explicitly via --images.
import argparse
import json
import sys
from pathlib import Path

from PIL import Image

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / 'gen_dataset'))
from metrics.scorer import MultiMetricScorer

METRICS = ['hpsv2', 'hpsv3', 'image_reward', 'pick_score']


def main():
    ap = argparse.ArgumentParser(description="Reward-score specific images for one prompt.")
    ap.add_argument('--prompt', required=True)
    ap.add_argument('--images', nargs='+', required=True, help="Image paths to score.")
    ap.add_argument('--metrics', nargs='+', default=METRICS)
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--out', default='scores.json', help="JSON output path.")
    args = ap.parse_args()

    scorer = MultiMetricScorer(metrics=args.metrics, device=args.device)

    rows = []
    for p in (Path(x) for x in args.images):
        img = Image.open(p).convert('RGB')
        s = scorer.score(img, args.prompt, image_path=str(p))
        rows.append({'image': p.name, **{m: float(s[m]) for m in scorer.metrics}})
        print(f"  {p.name}: " + "  ".join(f"{m}={float(s[m]):.4f}" for m in scorer.metrics))

    mean = {m: sum(r[m] for r in rows) / len(rows) for m in scorer.metrics}
    print(f"\n{'=' * 60}\nmean over {len(rows)} image(s)  (prompt: {args.prompt[:50]})\n{'=' * 60}")
    print("           " + "  ".join(f"{m:>12}" for m in scorer.metrics))
    print("mean       " + "  ".join(f"{mean[m]:>12.4f}" for m in scorer.metrics))

    Path(args.out).write_text(json.dumps({'prompt': args.prompt, 'per_image': rows, 'mean': mean}, indent=2))
    print(f"\nwrote {args.out}")


if __name__ == '__main__':
    main()
