# datasets/dualtask_dataset.py
import os
import warnings
from typing import Dict, Optional, Tuple, List, Any

import numpy as np
import cv2
import pandas as pd
import torch
from torch.utils.data import Dataset

warnings.filterwarnings("ignore")


class DualTaskDataset(Dataset):
    """
    SG-MTF 对齐版 Dataset（fold-wise 防泄露 + CE target + pid 规范化 + 尺寸语义明确 + seg 有效标记）

    默认返回（与你 train_dual.py 对齐）:
      img:        FloatTensor [3,H,W] in [0,1]
      seg_mask:   FloatTensor [1,H,W] (0/1). 若缺失则全0占位
      has_mask:   UInt8Tensor [] (0/1). loss 端用来 mask 掉 Lseg
      c_obs:      FloatTensor [D]. 缺失处置0
      m:          FloatTensor [D]. missingness mask (0/1)
      y:          LongTensor   []  分类标签
      cat_targets: dict[str, LongTensor[]] 每个分类变量一个标量 target，缺失/未知=-1（用于CE重建）

    重要约定：
      - image_size 语义为 (H, W)
      - cv2.resize 需要 (W, H)
      - onehot_slices 是 “在最终拼接后的 c_obs/m 向量里的全局切片索引”
        即：slice 已经包含 numeric offset（这是很多实现里最容易漏掉的点）
    """

    IMG_EXTS = (".bmp", ".png", ".jpg", ".jpeg", ".tif", ".tiff")

    def __init__(
        self,
        image_dir: str,
        mask_dir: Optional[str],
        clinical_excel: str,
        mode: str = "seg",                        # "seg" / "cls"
        transform=None,
        image_size: Tuple[int, int] = (256, 256), # (H, W)
        allow_missing_mask: bool = False,
        return_pid: bool = False,

        # ===== fold-wise 外部预处理器（训练折fit，val/test只transform）=====
        num_scaler: Optional[Dict[str, np.ndarray]] = None,      # {"mean":[Nn], "std":[Nn]}
        cat_maps: Optional[Dict[str, Dict[str, int]]] = None,    # {col: {category: idx}}
        unknown_cat_as_missing: bool = True,                     # val/test新类别 -> 缺失
        return_cat_targets: bool = True,                         # 返回 cat_targets 用于CE重建
    ):
        self.mode = str(mode)
        if self.mode not in ("seg", "cls"):
            raise ValueError(f"mode 必须是 'seg' 或 'cls'，当前为 {mode}")

        self.transform = transform

        # image_size 语义固定为 (H, W)
        self.H, self.W = int(image_size[0]), int(image_size[1])

        self.allow_missing_mask = bool(allow_missing_mask)
        self.return_pid = bool(return_pid)

        self.num_scaler = num_scaler
        self.cat_maps = cat_maps
        self.unknown_cat_as_missing = bool(unknown_cat_as_missing)
        self.return_cat_targets = bool(return_cat_targets)

        if not clinical_excel or (not os.path.exists(clinical_excel)):
            raise ValueError("必须指定有效的 clinical_excel 文件路径")
        self.clinical_excel = clinical_excel

        if not image_dir or (not os.path.exists(image_dir)):
            raise ValueError("必须指定有效的 image_dir 图像目录")
        self.image_dir = image_dir

        self.mask_dir = mask_dir
        if self.mode == "seg":
            if (not self.mask_dir) or (not os.path.exists(self.mask_dir)):
                raise ValueError("seg 模式必须指定有效的 mask_dir 掩码目录")

        # --- 读取临床（仅 transform，不在 Dataset 内 fit） ---
        (
            self.patient_ids,
            self.c_obs_all,
            self.m_all,
            self.all_labels,
            self.feature_names,
            self.numeric_feature_names,
            self.onehot_feature_names,
            self.onehot_slices,     # ✅ 全局切片（含 numeric offset）
            self.numeric_slice,     # (0, Nn)
            self.cat_targets_all,   # dict[col] -> np.ndarray [N] int64, 缺失=-1
        ) = self._extract_excel_features(
            self.clinical_excel,
            num_scaler=self.num_scaler,
            cat_maps=self.cat_maps,
            unknown_cat_as_missing=self.unknown_cat_as_missing,
            return_cat_targets=self.return_cat_targets,
        )

        self.pid_to_idx = {pid: idx for idx, pid in enumerate(self.patient_ids)}

        # --- 扫描图像与mask（pid统一规范化） ---
        self.image_path_dict = self._scan_paths(self.image_dir, exts=self.IMG_EXTS)

        if self.mask_dir and os.path.exists(self.mask_dir):
            self.mask_path_dict = self._scan_paths(
                self.mask_dir, exts=(".png", ".bmp", ".jpg", ".jpeg", ".tif", ".tiff")
            )
        else:
            self.mask_path_dict = {}

        # 有效样本：必须 image + clinical
        self.valid_pids = [pid for pid in self.image_path_dict.keys() if pid in self.pid_to_idx]

        # seg 模式：若不允许缺mask，则进一步要求mask存在
        if self.mode == "seg" and (not self.allow_missing_mask):
            self.valid_pids = [pid for pid in self.valid_pids if pid in self.mask_path_dict]

        if len(self.valid_pids) == 0:
            raise RuntimeError("有效样本数为 0：请检查 image/mask 与 clinical 的 pid 是否能匹配")

        self._print_stats()

    # ------------------------- pid 规范化 -------------------------
    @staticmethod
    def norm_pid(x: Any) -> str:
        return str(x).strip()

    # ------------------------- 对外接口 -------------------------
    def get_feature_dim(self) -> int:
        return int(self.c_obs_all.shape[1])

    def get_pid_list(self) -> List[str]:
        return list(self.valid_pids)

    # ------------------------- 扫描文件 -------------------------
    def _scan_paths(self, root_dir: str, exts: Tuple[str, ...]) -> Dict[str, str]:
        d: Dict[str, str] = {}
        for fn in os.listdir(root_dir):
            if fn.lower().endswith(exts):
                pid = self.norm_pid(os.path.splitext(fn)[0])
                d[pid] = os.path.join(root_dir, fn)
        if len(d) == 0:
            raise ValueError(f"在目录中未找到支持的文件：{root_dir}")
        return d

    # ------------------------- 统计打印 -------------------------
    def _print_stats(self):
        binc = np.bincount(self.all_labels, minlength=3)

        print(f"\n📊 {self.mode.upper()} 模式 - 数据集统计（SG-MTF fold-wise 对齐版）:")
        print(f"   - 临床数据总样本数：{len(self.patient_ids)}")
        print(f"   - 图像文件数：{len(self.image_path_dict)}")
        print(f"   - 掩码文件数：{len(self.mask_path_dict)}")
        print(f"   - 匹配后有效样本数：{len(self.valid_pids)}")
        print(f"   - 临床特征维度：{self.get_feature_dim()}")
        print(f"   - 分类标签分布：{binc} (对应标签0,1,2)")

        if self.num_scaler is None:
            print("   - 数值标准化：未应用（num_scaler=None；训练脚本需 fold-wise fit 后传入）")
        else:
            print("   - 数值标准化：已应用（外部 num_scaler transform）")

        if self.cat_maps is None:
            print("   - 分类词表：未指定 cat_maps（将用全数据 unique 构建 -> 有泄露风险，不推荐）")
        else:
            print("   - 分类词表：已指定 cat_maps（外部 fit，fold-wise 无泄露）")

    # ------------------------- 标签处理 -------------------------
    def _convert_labels(self, raw_labels: np.ndarray) -> np.ndarray:
        s = pd.Series(raw_labels)
        if s.isna().any():
            bad_idx = s[s.isna()].index.tolist()[:10]
            raise ValueError(f"标签列存在 NaN，示例索引: {bad_idx}")

        try:
            labels = s.astype(float).astype(int).to_numpy()
        except Exception:
            s_str = s.astype(str).str.strip().str.upper()
            mapping = {"LN0": 0, "LN1-3": 1, "LN1–3": 1, "LN1—3": 1, "LN4+": 2, "LN4 +": 2}
            mapped = s_str.map(mapping)
            if mapped.isna().any():
                bad = s_str[mapped.isna()].value_counts().head(20)
                raise ValueError("标签列存在无法识别的字符串标签。\n无法识别标签(Top20):\n" + str(bad))
            labels = mapped.astype(int).to_numpy()

        u = np.unique(labels)
        if not np.all(np.isin(u, [0, 1, 2])):
            raise ValueError(f"发现无效标签值 {u}，必须是 0/1/2")
        return labels.astype(np.int64)

    # ------------------------- fold-wise 数值标准化（transform only） -------------------------
    @staticmethod
    def _apply_num_scaler(
        num_obs: np.ndarray,
        num_mask: np.ndarray,
        num_scaler: Optional[Dict[str, np.ndarray]],
    ) -> np.ndarray:
        """
        只对 observed 位置做 (x-mean)/std；缺失仍保持 0（并由mask显式指示）。
        """
        if num_scaler is None:
            return num_obs.astype(np.float32)

        mean = num_scaler["mean"].astype(np.float32)
        std = num_scaler["std"].astype(np.float32)
        std = np.where(std < 1e-6, 1.0, std).astype(np.float32)

        z = (num_obs - mean[None, :]) / std[None, :]
        z = z * num_mask  # 缺失位置保持0
        return z.astype(np.float32)

    # ------------------------- 临床特征提取 -------------------------
    def _extract_excel_features(
        self,
        filename: str,
        num_scaler: Optional[Dict[str, np.ndarray]],
        cat_maps: Optional[Dict[str, Dict[str, int]]],
        unknown_cat_as_missing: bool,
        return_cat_targets: bool,
    ):
        df = pd.read_excel(filename, engine="openpyxl")
        print(f"\n🔍 处理临床数据：{filename}")
        print(f"   - Excel总行数：{len(df)}")
        print(f"   - Excel总列数：{len(df.columns)}")

        patient_ids = df.iloc[:, 0].astype(str).apply(self.norm_pid).to_numpy()
        labels = self._convert_labels(df.iloc[:, -1].to_numpy())

        features_df = df.iloc[:, 1:-1]
        print(f"   - 原始特征列数：{len(features_df.columns)}")

        numeric_df = features_df.select_dtypes(include=[np.number]).copy()
        cat_df = features_df.select_dtypes(exclude=[np.number]).copy()

        num_cols = list(numeric_df.columns)
        cat_cols = list(cat_df.columns)

        print(f"   - 数值型特征列数：{len(num_cols)}")
        print(f"   - 分类型特征列数：{len(cat_cols)}")

        # ---- numeric: NaN->0 + mask + (optional) fold-wise zscore transform ----
        if len(num_cols) > 0:
            num_arr = numeric_df.to_numpy(dtype=np.float32)                       # [N, Nn]
            num_mask = (~np.isnan(num_arr)).astype(np.float32)                    # [N, Nn]
            num_obs = np.nan_to_num(num_arr, nan=0.0).astype(np.float32)          # [N, Nn]
            num_obs = self._apply_num_scaler(num_obs, num_mask, num_scaler)
        else:
            num_obs = np.zeros((len(df), 0), dtype=np.float32)
            num_mask = np.zeros((len(df), 0), dtype=np.float32)

        numeric_slice = (0, len(num_cols))
        numeric_feature_names = [str(c) for c in num_cols]

        # ---- categorical: one-hot + group mask + CE target ----
        # 若未提供 cat_maps：兼容模式（但有泄露风险）
        if cat_maps is None:
            cat_maps = {}
            for col in cat_cols:
                s = cat_df[col]
                is_obs = ~s.isna()
                cats = pd.Series(s[is_obs].astype(str).unique()).sort_values().tolist()
                cat_maps[col] = {c: i for i, c in enumerate(cats)}

        onehot_obs_list: List[np.ndarray] = []
        onehot_mask_list: List[np.ndarray] = []
        onehot_feature_names: List[str] = []
        onehot_slices: Dict[str, Tuple[int, int]] = {}
        cat_targets_all: Dict[str, np.ndarray] = {}

        # ✅ 全局offset：one-hot 从 numeric_dim 之后开始
        cursor = len(num_cols)

        for col in cat_cols:
            mapper = cat_maps.get(col, {})
            K = int(len(mapper))
            if K <= 0:
                if return_cat_targets:
                    cat_targets_all[col] = (-1 * np.ones((len(df),), dtype=np.int64))
                continue

            s = cat_df[col]
            is_obs = ~s.isna()

            oh = np.zeros((len(df), K), dtype=np.float32)
            oh_m = np.zeros((len(df), K), dtype=np.float32)

            targets = (-1 * np.ones((len(df),), dtype=np.int64)) if return_cat_targets else None

            obs_rows = np.where(is_obs.to_numpy())[0]
            obs_vals = s[is_obs].astype(str).to_numpy()

            for r, v in zip(obs_rows, obs_vals):
                j = mapper.get(v, None)
                if j is None:
                    if unknown_cat_as_missing:
                        continue
                    raise ValueError(f"发现未知类别：col={col}, value={v}，请检查 cat_maps 或数据")
                oh[r, j] = 1.0
                oh_m[r, :] = 1.0  # group-wise observed mask
                if targets is not None:
                    targets[r] = int(j)

            # ✅ onehot_slices 是全局向量上的切片
            onehot_slices[col] = (cursor, cursor + K)
            cursor += K

            for c, j in sorted(mapper.items(), key=lambda x: x[1]):
                onehot_feature_names.append(f"{col}__{c}")

            onehot_obs_list.append(oh)
            onehot_mask_list.append(oh_m)

            if targets is not None:
                cat_targets_all[col] = targets

        onehot_obs = (
            np.concatenate(onehot_obs_list, axis=1).astype(np.float32)
            if len(onehot_obs_list) > 0 else np.zeros((len(df), 0), dtype=np.float32)
        )
        onehot_mask = (
            np.concatenate(onehot_mask_list, axis=1).astype(np.float32)
            if len(onehot_mask_list) > 0 else np.zeros((len(df), 0), dtype=np.float32)
        )

        # ---- concat clinical vector ----
        c_obs = np.concatenate([num_obs, onehot_obs], axis=1).astype(np.float32)
        m = np.concatenate([num_mask, onehot_mask], axis=1).astype(np.float32)

        feature_names = numeric_feature_names + onehot_feature_names

        miss_ratio = 1.0 - (m.sum() / max(1.0, m.size))
        print(f"   - 临床缺失率(按维度) ≈ {miss_ratio:.4f}")

        return (
            patient_ids,
            c_obs,
            m,
            labels,
            feature_names,
            numeric_feature_names,
            onehot_feature_names,
            onehot_slices,
            numeric_slice,
            cat_targets_all,
        )

    # ------------------------- 图像/掩码读取（cv2.resize: (W,H)） -------------------------
    def _read_image(self, img_path: str) -> torch.Tensor:
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(f"无法读取图像：{img_path}")
        img = cv2.resize(img, (self.W, self.H), interpolation=cv2.INTER_LINEAR)
        img = (img.astype(np.float32) / 255.0)
        img_t = torch.from_numpy(img).unsqueeze(0).repeat(3, 1, 1)  # [3,H,W]
        return img_t

    def _read_mask(self, mask_path: str) -> torch.Tensor:
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise FileNotFoundError(f"无法读取 mask：{mask_path}")
        mask = cv2.resize(mask, (self.W, self.H), interpolation=cv2.INTER_NEAREST)
        mask = (mask.astype(np.float32) / 255.0)
        mask = (mask > 0.5).astype(np.float32)
        return torch.from_numpy(mask).unsqueeze(0)  # [1,H,W]

    def __len__(self) -> int:
        return len(self.valid_pids)

    def __getitem__(self, idx: int):
        pid = self.valid_pids[idx]
        cidx = self.pid_to_idx[pid]

        img = self._read_image(self.image_path_dict[pid])

        # --- seg mask + has_mask ---
        if self.mode == "seg":
            mask_path = self.mask_path_dict.get(pid, None)
            if mask_path is not None:
                seg_mask = self._read_mask(mask_path)
                has_mask = torch.tensor(1, dtype=torch.uint8)
            else:
                if not self.allow_missing_mask:
                    raise FileNotFoundError(f"seg 模式缺少 mask：pid={pid}")
                seg_mask = torch.zeros((1, self.H, self.W), dtype=torch.float32)
                has_mask = torch.tensor(0, dtype=torch.uint8)
        else:
            seg_mask = torch.zeros((1, self.H, self.W), dtype=torch.float32)
            has_mask = torch.tensor(0, dtype=torch.uint8)

        c_obs = torch.tensor(self.c_obs_all[cidx], dtype=torch.float32)
        m = torch.tensor(self.m_all[cidx], dtype=torch.float32)
        y = torch.tensor(self.all_labels[cidx], dtype=torch.long)

        cat_targets = None
        if self.return_cat_targets:
            cat_targets = {
                col: torch.tensor(int(arr[cidx]), dtype=torch.long)
                for col, arr in self.cat_targets_all.items()
            }

        if self.transform is not None:
            # 约定：transform 接受 (img, seg_mask) -> (img, seg_mask)
            img, seg_mask = self.transform(img, seg_mask)

        # --- return (与你 train_dual.py 对齐) ---
        if self.return_pid:
            if self.return_cat_targets:
                return img, seg_mask, has_mask, c_obs, m, y, cat_targets, pid
            return img, seg_mask, has_mask, c_obs, m, y, pid

        if self.return_cat_targets:
            return img, seg_mask, has_mask, c_obs, m, y, cat_targets
        return img, seg_mask, has_mask, c_obs, m, y