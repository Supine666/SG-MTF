import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# -------------------------
# Utils
# -------------------------
def _make_divisible(v, divisor=8):
    return int(math.ceil(v / divisor) * divisor)


def _make_gn(num_channels: int, num_groups: int = 32, eps: float = 1e-5, affine: bool = True) -> nn.GroupNorm:
    """
    Create GroupNorm with a safe group count.
    - Prefer num_groups (default 32)
    - If num_channels not divisible by groups, reduce groups until divisible
    - Fallback to 1 group (equivalent to LayerNorm over channels for conv features)
    """
    g = int(min(num_groups, num_channels))
    while g > 1 and (num_channels % g != 0):
        g -= 1
    return nn.GroupNorm(num_groups=g, num_channels=num_channels, eps=eps, affine=affine)


class DropPath(nn.Module):
    """Stochastic Depth (per-sample)."""
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = float(drop_prob)

    def forward(self, x):
        if self.drop_prob == 0.0 or (not self.training):
            return x
        keep_prob = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        return x.div(keep_prob) * random_tensor


class ConvGNAct(nn.Module):
    """
    Replace BatchNorm2d with GroupNorm to stabilize small-batch multitask training.
    """
    def __init__(self, in_ch, out_ch, k=3, s=1, p=None, groups=1, act=True, gn_groups=32):
        super().__init__()
        if p is None:
            p = k // 2
        self.conv = nn.Conv2d(in_ch, out_ch, k, s, p, groups=groups, bias=False)
        self.norm = _make_gn(out_ch, num_groups=gn_groups)
        self.act = nn.SiLU(inplace=True) if act else nn.Identity()

    def forward(self, x):
        return self.act(self.norm(self.conv(x)))


class SqueezeExcite(nn.Module):
    """SE block for channel attention."""
    def __init__(self, ch, rd_ratio=0.25):
        super().__init__()
        rd_ch = max(8, int(ch * rd_ratio))
        self.fc1 = nn.Conv2d(ch, rd_ch, kernel_size=1)
        self.act = nn.SiLU(inplace=True)
        self.fc2 = nn.Conv2d(rd_ch, ch, kernel_size=1)

    def forward(self, x):
        w = F.adaptive_avg_pool2d(x, 1)
        w = self.fc2(self.act(self.fc1(w)))
        w = torch.sigmoid(w)
        return x * w


class ResSEBlock(nn.Module):
    """
    Performance-oriented residual block:
      - 1x1 -> 3x3 -> 1x1 (bottleneck-like)
      - SE attention
      - DropPath
    """
    def __init__(self, in_ch, out_ch, stride=1, expand=2, se_ratio=0.25, drop_path=0.0, gn_groups=32):
        super().__init__()
        mid = _make_divisible(out_ch // expand, 8)

        self.conv1 = ConvGNAct(in_ch, mid, k=1, s=1, p=0, gn_groups=gn_groups)
        self.conv2 = ConvGNAct(mid, mid, k=3, s=stride, p=1, gn_groups=gn_groups)
        self.conv3 = ConvGNAct(mid, out_ch, k=1, s=1, p=0, act=False, gn_groups=gn_groups)
        self.se = SqueezeExcite(out_ch, rd_ratio=se_ratio) if se_ratio and se_ratio > 0 else nn.Identity()
        self.drop_path = DropPath(drop_path) if drop_path and drop_path > 0 else nn.Identity()
        self.act = nn.SiLU(inplace=True)

        if stride != 1 or in_ch != out_ch:
            self.short = ConvGNAct(in_ch, out_ch, k=1, s=stride, p=0, act=False, gn_groups=gn_groups)
        else:
            self.short = nn.Identity()

    def forward(self, x):
        identity = self.short(x)
        out = self.conv3(self.conv2(self.conv1(x)))
        out = self.se(out)
        out = identity + self.drop_path(out)
        return self.act(out)


class Stage(nn.Module):
    def __init__(self, in_ch, out_ch, depth, stride, drop_path_rate=0.0, se_ratio=0.25, gn_groups=32):
        super().__init__()
        blocks = []
        for i in range(depth):
            s = stride if i == 0 else 1
            dpr = drop_path_rate * (i / max(1, depth - 1)) if depth > 1 else drop_path_rate
            blocks.append(
                ResSEBlock(
                    in_ch if i == 0 else out_ch,
                    out_ch,
                    stride=s,
                    expand=2,
                    se_ratio=se_ratio,
                    drop_path=dpr,
                    gn_groups=gn_groups,
                )
            )
        self.blocks = nn.Sequential(*blocks)

    def forward(self, x):
        return self.blocks(x)


# -------------------------
# Encoder: outputs f1,f2,f3,f4 with channels [120,240,480,960]
# resolution: f1=1/4, f2=1/8, f3=1/16, f4=1/32 (typical)
# -------------------------
class SGMTFPerformanceEncoder(nn.Module):
    def __init__(self, in_ch=3, drop_path_rate=0.1, se_ratio=0.25, gn_groups=32):
        super().__init__()

        # stem: 1/2 then 1/4
        self.stem1 = ConvGNAct(in_ch, 64, k=3, s=2, p=1, gn_groups=gn_groups)
        self.stem2 = ConvGNAct(64, 120, k=3, s=2, p=1, gn_groups=gn_groups)  # -> f1 channels 120 at 1/4

        # stages
        self.stage2 = Stage(120, 240, depth=3, stride=2, drop_path_rate=drop_path_rate, se_ratio=se_ratio, gn_groups=gn_groups)  # 1/8
        self.stage3 = Stage(240, 480, depth=5, stride=2, drop_path_rate=drop_path_rate, se_ratio=se_ratio, gn_groups=gn_groups)  # 1/16
        self.stage4 = Stage(480, 960, depth=3, stride=2, drop_path_rate=drop_path_rate, se_ratio=se_ratio, gn_groups=gn_groups)  # 1/32

    def forward(self, x):
        x = self.stem1(x)
        f1 = self.stem2(x)        # [B,120,H/4,W/4]
        f2 = self.stage2(f1)      # [B,240,H/8,W/8]
        f3 = self.stage3(f2)      # [B,480,H/16,W/16]
        f4 = self.stage4(f3)      # [B,960,H/32,W/32]
        return f1, f2, f3, f4
