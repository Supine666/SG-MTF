import torch
import torch.nn as nn


class MissingnessRobustClinicalModule(nn.Module):
    """
    Missingness-robust clinical modeling:
      Eq.(3)-(7): psi, imputation, reliability, phi_clin
    """
    def __init__(
        self,
        clinical_dim: int,
        img_dim: int,
        numeric_slice,                 # (0, Nn)
        onehot_slices_dict,            # dict: col -> (s,e)
        h_dim: int = 256,
        embed_dim: int = 128,
        dropout: float = 0.2,
    ):
        super().__init__()
        if clinical_dim <= 0:
            raise ValueError("clinical_dim must be >0")

        self.clinical_dim = int(clinical_dim)
        self.embed_dim = int(embed_dim)

        self.numeric_slice = (int(numeric_slice[0]), int(numeric_slice[1]))
        self.onehot_slices_dict = {str(k): (int(v[0]), int(v[1])) for k, v in dict(onehot_slices_dict).items()}

        n0, n1 = self.numeric_slice
        if not (0 <= n0 <= n1 <= self.clinical_dim):
            raise ValueError(f"Invalid numeric_slice {self.numeric_slice} for clinical_dim={self.clinical_dim}")
        self.num_dim = n1 - n0

        # Eq.(3) psi([cobs⊙m, m]) -> h_clin
        self.psi = nn.Sequential(
            nn.Linear(self.clinical_dim * 2, h_dim),
            nn.LayerNorm(h_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(h_dim, h_dim),
            nn.LayerNorm(h_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

        # shared fusion -> z
        self.hm_proj = nn.Sequential(
            nn.Linear(h_dim + img_dim, h_dim),
            nn.LayerNorm(h_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

        # numeric head
        self.f_imp_num = nn.Linear(h_dim, self.num_dim, bias=True) if self.num_dim > 0 else None

        # categorical heads
        self.f_imp_cat = nn.ModuleDict()
        for col, (s, e) in self.onehot_slices_dict.items():
            K = int(e - s)
            if K > 0:
                self.f_imp_cat[col] = nn.Linear(h_dim, K, bias=True)

        # reliability
        self.f_conf = nn.Sequential(
            nn.Linear(h_dim + img_dim, h_dim),
            nn.LayerNorm(h_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(h_dim, self.clinical_dim),
        )

        # Eq.(7) phi_clin([c*, m, r]) -> v_clin*
        self.phi_clin = nn.Sequential(
            nn.Linear(self.clinical_dim * 3, h_dim),
            nn.LayerNorm(h_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(h_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

    def forward(self, c_obs, m, v_img):
        if c_obs.dim() != 2 or m.dim() != 2:
            raise ValueError("c_obs and m must be [B,D]")
        if c_obs.shape != m.shape:
            raise ValueError(f"c_obs shape {tuple(c_obs.shape)} must match m {tuple(m.shape)}")
        if c_obs.size(1) != self.clinical_dim:
            raise ValueError(f"Expected clinical_dim={self.clinical_dim}, got {c_obs.size(1)}")

        # Eq.(3)
        x = torch.cat([c_obs * m, m], dim=1)   # [B,2D]
        h_clin = self.psi(x)                   # [B,h]

        hm = torch.cat([h_clin, v_img], dim=1) # [B,h+img]
        z = self.hm_proj(hm)                   # [B,h]

        # predictions -> c_hat_full [B,D]
        n0, n1 = self.numeric_slice
        c_hat_full = torch.zeros_like(c_obs)

        c_hat_num = None
        if self.num_dim > 0:
            c_hat_num = self.f_imp_num(z)      # [B,Nn]
            c_hat_full[:, n0:n1] = c_hat_num

        cat_logits = {}
        for col, (s, e) in self.onehot_slices_dict.items():
            if col not in self.f_imp_cat:
                continue
            logits = self.f_imp_cat[col](z)    # [B,K]
            cat_logits[col] = logits
            probs = torch.softmax(logits, dim=1)
            c_hat_full[:, s:e] = probs

        # reliability
        r = torch.sigmoid(self.f_conf(hm))     # [B,D]
        r_bar = r.mean(dim=1, keepdim=True)    # [B,1]

        # c*
        c_star = m * c_obs + (1.0 - m) * c_hat_full
        clin_pack = torch.cat([c_star, m, r], dim=1)  # [B,3D]
        v_clin_star = self.phi_clin(clin_pack)        # [B,embed_dim]

        return {
            "h_clin": h_clin,
            "c_hat": c_hat_full,
            "c_hat_num": c_hat_num,
            "cat_logits": cat_logits,
            "c_star": c_star,
            "r": r,
            "r_bar": r_bar,
            "v_clin_star": v_clin_star,
        }