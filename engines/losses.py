# engines/losses.py
from typing import Callable
import torch
import torch.nn as nn

from models.sgmtf import SGMTFModel

class DiceLossPerSample(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)
        inter = (probs * targets).sum(dim=(2, 3))
        union = probs.sum(dim=(2, 3)) + targets.sum(dim=(2, 3))
        dice = (2 * inter + self.eps) / (union + self.eps)  # [B,1]
        return (1.0 - dice).view(dice.size(0))              # [B]

def make_nomissing_loss_fn(
    cls_criterion: nn.Module,
    seg_bce_weight: float,
    force_full_observed_mask: bool,
) -> Callable:
    bce = nn.BCEWithLogitsLoss(reduction="none")
    dice_ps = DiceLossPerSample()

    def _loss_fn(
        model: SGMTFModel,
        x_img: torch.Tensor,
        seg_gt: torch.Tensor,
        has_mask: torch.Tensor,
        y_gt: torch.Tensor,
        c_obs: torch.Tensor,
        m: torch.Tensor,
        **kwargs,
    ):
        if force_full_observed_mask:
            m = torch.ones_like(m)

        seg_logits, cls_logits, _ = model(x_img, c_obs=c_obs, m=m, task="both")

        bce_map = bce(seg_logits, seg_gt)
        bce_per = bce_map.view(bce_map.size(0), -1).mean(1)
        dice_per = dice_ps(seg_logits, seg_gt)
        seg_per = seg_bce_weight * bce_per + (1.0 - seg_bce_weight) * dice_per
        Lseg = (seg_per * has_mask).sum() / has_mask.sum().clamp_min(1.0)

        Lcls = cls_criterion(cls_logits, y_gt)
        Ltotal = Lseg + Lcls

        log = {"Lseg": float(Lseg.detach().cpu()), "Lcls": float(Lcls.detach().cpu()), "Limp": 0.0, "Lcons": 0.0}
        return Ltotal, log

    return _loss_fn