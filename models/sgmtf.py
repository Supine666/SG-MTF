import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbones import SGMTFPerformanceEncoder
from .heads import SGMTFPerformanceDecoder, EnhancedClassifier, MissingnessRobustClinicalModule
from .roi import ROIGuidedPooling
from .fusion import AdaptiveFeatureFusion, ReliabilityAwareCrossModalGating


class SGMTFModel(nn.Module):
    """
    SG-MTF (core):
      - shared encoder/decoder (seg)
      - Eq.(1) seg_head -> roi_prob
      - Eq.(2) ROI-guided pooling
      - Eq.(3)-(7) clinical imputation + reliability
      - Eq.(8)-(10) reliability-aware gating + fusion
      - uncertainty-aware tri-task weights (Eq.(14))
    """
    def __init__(
        self,
        clinical_dim: int,
        numeric_slice,
        onehot_slices_dict,
        num_classes: int = 3,
        use_pca: bool = False,
        pca_dim: int = 100,
        clin_embed_dim: int = 128,
        cls_hidden_dims=(512, 256),
        cls_dropout: float = 0.3,
        detach_roi_in_cls: bool = True,
        detach_segfeat_in_cls: bool = False,
        lambda_cons: float = 0.5,
        drop_path_rate: float = 0.1,
        se_ratio: float = 0.25,
        gn_groups: int = 32,
    ):
        super().__init__()
        self.clinical_dim = int(clinical_dim)
        self.num_classes = int(num_classes)
        self.detach_roi_in_cls = bool(detach_roi_in_cls)
        self.detach_segfeat_in_cls = bool(detach_segfeat_in_cls)
        self.lambda_cons = float(lambda_cons)

        # 1) segmentation encoder/decoder
        self.encoder = SGMTFPerformanceEncoder(in_ch=3, drop_path_rate=drop_path_rate, se_ratio=se_ratio, gn_groups=gn_groups)
        self.decoder = SGMTFPerformanceDecoder(gn_groups=gn_groups)

        # 2) Eq.(1): S = Conv1×1(Fseg) -> Proi = sigmoid(S)
        self.seg_head = nn.Conv2d(120, 1, kernel_size=1, bias=True)

        # 3) ROI pooling (Eq.(2))
        self.roi_pool = ROIGuidedPooling()
        self.segfeat_to_imgdim = nn.Sequential(
            nn.Conv2d(120, 960, kernel_size=1, bias=False),
            nn.GroupNorm(gn_groups, 960),
            nn.ReLU(inplace=True),
        )

        # 4) feature fusion (optional PCA)
        self.feature_fusion = AdaptiveFeatureFusion(
            img_feat_dim=960,
            clinical_embed_dim=clin_embed_dim,
            use_pca=use_pca,
            pca_dim=pca_dim,
        )

        # 5) clinical module (Eq.(3)-(7))
        self.clin_module = MissingnessRobustClinicalModule(
            clinical_dim=self.clinical_dim,
            img_dim=self.feature_fusion.img_out_dim,
            numeric_slice=numeric_slice,
            onehot_slices_dict=onehot_slices_dict,
            h_dim=256,
            embed_dim=clin_embed_dim,
            dropout=min(0.3, cls_dropout),
        )

        # 6) reliability-aware gating (Eq.(8)(9))
        self.gating = ReliabilityAwareCrossModalGating(
            img_dim=self.feature_fusion.img_out_dim,
            clin_dim=clin_embed_dim,
        )

        # 7) classifier
        self.cls_head = EnhancedClassifier(
            feature_dim=self.feature_fusion.fused_feat_dim,
            output_size=self.num_classes,
            hidden_dims=cls_hidden_dims,
            dropout_rate=cls_dropout,
        )

        # 8) uncertainty weights (Eq.(14))  log σ^2
        self.log_var_seg = nn.Parameter(torch.tensor(0.0))
        self.log_var_cls = nn.Parameter(torch.tensor(0.0))
        self.log_var_imp = nn.Parameter(torch.tensor(0.0))

        self._init_basic_weights()

    def forward(self, x_img, c_obs=None, m=None, task="both"):
        """
        Returns:
          seg_logits: [B,1,H,W] or None
          cls_logits: [B,num_classes] or None
          aux: dict (roi_prob, seg_logits_low, seg_prob_low, plus clinical recon outputs in cls mode)
        """
        if x_img.dim() != 4:
            raise ValueError(f"x_img must be [B,C,H,W], got {tuple(x_img.shape)}")
        B, _, H, W = x_img.shape

        need_seg = task in ["seg", "both"]
        need_cls = task in ["cls", "both"]

        f1, f2, f3, f4 = self.encoder(x_img)
        feat_seg = self.decoder(f1, f2, f3, f4)              # [B,120,H/4,W/4]

        seg_logits_low = self.seg_head(feat_seg)              # [B,1,H/4,W/4]
        seg_prob_low = torch.sigmoid(seg_logits_low)          # [B,1,H/4,W/4]
        roi_prob = seg_prob_low

        seg_logits = None
        if need_seg:
            seg_logits = F.interpolate(seg_logits_low, size=(H, W), mode="bilinear", align_corners=False)

        cls_logits = None
        aux = {
            "roi_prob": roi_prob,
            "seg_logits_low": seg_logits_low,
            "seg_prob_low": seg_prob_low,
        }

        if need_cls:
            if c_obs is None or m is None:
                raise ValueError("Classification requires c_obs and m.")
            if c_obs.shape != m.shape or c_obs.shape[0] != B or c_obs.shape[1] != self.clinical_dim:
                raise ValueError(f"c_obs/m must be [B,{self.clinical_dim}] and match batch size.")

            roi_for_cls = roi_prob.detach() if self.detach_roi_in_cls else roi_prob
            feat_for_cls = feat_seg.detach() if self.detach_segfeat_in_cls else feat_seg

            Fcls = self.segfeat_to_imgdim(feat_for_cls)         # [B,960,H/4,W/4]
            v_img = self.roi_pool(Fcls, roi_for_cls)            # [B,960]
            v_img = self.feature_fusion.forward_img(v_img)      # -> [B,img_out_dim]

            clin_out = self.clin_module(c_obs.float(), m.float(), v_img)
            v_clin_star = clin_out["v_clin_star"]
            r_bar = clin_out["r_bar"]

            v_img_tilde, v_clin_tilde = self.gating(v_img, v_clin_star, r_bar)
            vfused = self.feature_fusion.forward_fuse(v_img_tilde, v_clin_tilde)

            cls_logits = self.cls_head(vfused)
            aux.update({"v_img": v_img, "vfused": vfused, **clin_out})

        if task == "seg":
            return seg_logits, None, aux
        if task == "cls":
            return None, cls_logits, aux
        return seg_logits, cls_logits, aux

    # -------------------------- tri-task uncertainty loss (Eq.(14)) --------------------------
    def get_total_loss(self, Lseg, Lcls, Limp, Lcons=0.0, clamp=(-5.0, 5.0)):
        lo, hi = clamp
        lv_seg = self.log_var_seg.clamp(lo, hi)
        lv_cls = self.log_var_cls.clamp(lo, hi)
        lv_imp = self.log_var_imp.clamp(lo, hi)

        loss = (
            0.5 * torch.exp(-lv_seg) * Lseg + 0.5 * lv_seg
            + 0.5 * torch.exp(-lv_cls) * Lcls + 0.5 * lv_cls
            + 0.5 * torch.exp(-lv_imp) * Limp + 0.5 * lv_imp
            + self.lambda_cons * Lcons
        )

        weights = {
            "w_seg": float((0.5 * torch.exp(-lv_seg)).detach().cpu()),
            "w_cls": float((0.5 * torch.exp(-lv_cls)).detach().cpu()),
            "w_imp": float((0.5 * torch.exp(-lv_imp)).detach().cpu()),
            "lambda_cons": float(self.lambda_cons),
        }
        return loss, weights

    def _init_basic_weights(self):
        for mm in self.modules():
            if isinstance(mm, nn.Conv2d):
                nn.init.kaiming_normal_(mm.weight, mode="fan_out", nonlinearity="relu")
                if mm.bias is not None:
                    nn.init.zeros_(mm.bias)
            elif isinstance(mm, nn.Linear):
                nn.init.xavier_uniform_(mm.weight)
                if mm.bias is not None:
                    nn.init.zeros_(mm.bias)
            elif isinstance(mm, (nn.LayerNorm, nn.GroupNorm, nn.BatchNorm2d, nn.BatchNorm1d)):
                if hasattr(mm, "weight") and mm.weight is not None:
                    nn.init.constant_(mm.weight, 1.0)
                if hasattr(mm, "bias") and mm.bias is not None:
                    nn.init.constant_(mm.bias, 0.0)