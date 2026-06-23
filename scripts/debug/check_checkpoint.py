# Q: what config / normalization / param-count / module-naming does each original PAINE checkpoint hold?
# Standalone — no predictor imports, runs anywhere with torch. Reveals how the per-DM weights differ.
import argparse
import json
from pathlib import Path

import torch


def summarize(path):
    ck = torch.load(path, map_location='cpu', weights_only=False)
    sd = ck['model_state_dict']
    return {
        'file': str(path),
        'top_level_keys': sorted(ck.keys()),
        'model_config': ck.get('model_config'),
        'normalization': ck.get('normalization'),
        'param_M': round(sum(v.numel() for v in sd.values()) / 1e6, 3),
        'n_state_keys': len(sd),
        'top_level_modules': sorted({k.split('.')[0] for k in sd}),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('paths', nargs='+')
    args = ap.parse_args()
    for p in args.paths:
        print(json.dumps(summarize(p), indent=2, default=str))
        print()


if __name__ == '__main__':
    main()
