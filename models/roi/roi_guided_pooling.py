import torch.nn as nn
import torch.nn.functional as F


class ROIGuidedPooling(nn.Module):
    """
    Eq.(2): v_img = Σ(F_cls ⊙ P_roi) / (Σ(P_roi)+eps)
    """
    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, feat_map, roi_prob):
        # ensure roi_prob is [B,1,h,w]
        if roi_prob.dim() != 4 or roi_prob.size(1) != 1:
            raise ValueError(f"roi_prob must be [B,1,h,w], got {tuple(roi_prob.shape)}")

        if roi_prob.shape[-2:] != feat_map.shape[-2:]:
            roi_prob = F.interpolate(roi_prob, size=feat_map.shape[-2:], mode="bilinear", align_corners=False)

        w = roi_prob.clamp(0, 1)                      # [B,1,h,w]
        num = (feat_map * w).sum(dim=(2, 3))          # [B,C]
        den = w.sum(dim=(2, 3)).clamp_min(self.eps)   # [B,1]
        return num / den                               # [B,C]