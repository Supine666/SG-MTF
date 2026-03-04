# datasets/transforms.py
from typing import Tuple, Optional
import torch


class IdentityTransform:
    """啥也不做，用于占位。"""
    def __call__(self, img: torch.Tensor, mask: torch.Tensor):
        return img, mask


class Normalize01ToMeanStd:
    """
    对 img 做 (x-mean)/std。mask 不变。
    注意：img 应该是 [3,H,W] 且已在 [0,1]（你 dataset 已满足）。
    """
    def __init__(self, mean: Tuple[float, float, float], std: Tuple[float, float, float], eps: float = 1e-6):
        self.mean = torch.tensor(mean, dtype=torch.float32).view(3, 1, 1)
        self.std = torch.tensor(std, dtype=torch.float32).view(3, 1, 1)
        self.eps = float(eps)

    def __call__(self, img: torch.Tensor, mask: torch.Tensor):
        mean = self.mean.to(img.device)
        std = self.std.to(img.device)
        img = (img - mean) / (std + self.eps)
        return img, mask


class Compose:
    """顺序组合：每个 transform 都必须是 (img, mask)->(img, mask)。"""
    def __init__(self, transforms):
        self.transforms = list(transforms)

    def __call__(self, img: torch.Tensor, mask: torch.Tensor):
        for t in self.transforms:
            img, mask = t(img, mask)
        return img, mask