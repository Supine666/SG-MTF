import torch
import torch.nn as nn


class ReliabilityAwareCrossModalGating(nn.Module):
    """
    Eq.(8)(9) reliability-aware cross-modal gating
    """
    def __init__(self, img_dim: int, clin_dim: int):
        super().__init__()
        self.img_gate = nn.Linear(clin_dim + 1, img_dim, bias=True)
        self.clin_gate = nn.Linear(img_dim, clin_dim, bias=True)

    def forward(self, v_img, v_clin_star, r_bar):
        g_img = torch.sigmoid(self.img_gate(torch.cat([v_clin_star, r_bar], dim=1)))
        g_clin = torch.sigmoid(self.clin_gate(v_img))
        v_img_tilde = v_img * g_img + v_img
        v_clin_tilde = v_clin_star * g_clin + v_clin_star
        return v_img_tilde, v_clin_tilde


class AdaptiveFeatureFusion(nn.Module):
    """
    (optional) PCA-like projection for image vector, then concat with clinical vector.
    """
    def __init__(self, img_feat_dim=960, clinical_embed_dim=0, use_pca=False, pca_dim=100):
        super().__init__()
        self.use_pca = use_pca
        if self.use_pca:
            self.pca_linear = nn.Linear(img_feat_dim, pca_dim, bias=True)
            self.pca_norm = nn.LayerNorm(pca_dim)
            self.pca_act = nn.ReLU(inplace=True)
            self.img_out_dim = pca_dim
        else:
            self.img_out_dim = img_feat_dim

        self.fused_feat_dim = self.img_out_dim + clinical_embed_dim

    def forward_img(self, img_vec):
        if not self.use_pca:
            return img_vec
        return self.pca_act(self.pca_norm(self.pca_linear(img_vec)))

    def forward_fuse(self, img_vec_out, clinical_vec=None):
        if clinical_vec is None:
            return img_vec_out
        return torch.cat([img_vec_out, clinical_vec], dim=1)