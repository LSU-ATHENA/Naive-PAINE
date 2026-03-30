import torch
from PIL import Image
from typing import Dict


class MultiMetricScorer:
    METRICS = ['hpsv2', 'hpsv3', 'image_reward', 'pick_score']

    def __init__(self, metrics=None, device='cuda'):
        self.device = device
        self.metrics = metrics or self.METRICS
        self._models = {}

        if 'hpsv2' in self.metrics:
            import hpsv2 as _hpsv2
            self._models['hpsv2'] = _hpsv2

        if 'hpsv3' in self.metrics:
            from hpsv3 import HPSv3RewardInferencer
            self._models['hpsv3'] = HPSv3RewardInferencer(device=device)

        if 'image_reward' in self.metrics:
            import ImageReward as RM
            self._models['image_reward'] = RM.load("ImageReward-v1.0")

        if 'pick_score' in self.metrics:
            from .pick_score import PickScorer
            self._models['pick_score'] = PickScorer(device=device)


    @torch.inference_mode()
    def score(self, image: Image.Image, prompt: str, image_path: str = None) -> Dict[str, float]:
        prompt = prompt.strip()
        image_rgb = image.convert("RGB")
        results = {}

        if 'hpsv2' in self.metrics:
            val = float(self._models['hpsv2'].score([image_rgb], prompt, hps_version="v2.1")[0])
            results['hpsv2'] = round(val, 4)

        if 'hpsv3' in self.metrics:
            rewards = self._models['hpsv3'].reward([image_path], [prompt])
            val = float(rewards[0][0].item())
            results['hpsv3'] = round(val, 4)

        if 'image_reward' in self.metrics:
            val = float(self._models['image_reward'].score(prompt, image_rgb))
            results['image_reward'] = round(val, 4)

        if 'pick_score' in self.metrics:
            val = self._models['pick_score'].score(image_rgb, prompt)
            results['pick_score'] = round(val, 4)

        return results
