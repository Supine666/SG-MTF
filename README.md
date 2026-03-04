# SG-MTF

### Segmentation-Guided Multi-Task Framework for Breast Lesion Segmentation and Molecular Subtyping

Official PyTorch implementation of the paper:

**“A Segmentation-Guided Multi-Task Framework Using Multimodal Ultrasound Images and Clinicopathological Data for Breast Lesion Segmentation and Molecular Subtyping”**

---

## Overview

Breast cancer diagnosis often requires integrating **ultrasound imaging** with **clinicopathological variables**. However, most multimodal learning frameworks rely on global image features without explicit spatial constraints, which limits the ability to capture lesion-specific morphological cues.

This repository implements **SG-MTF**, a **segmentation-guided multimodal multi-task framework** that performs:

* **Breast lesion segmentation**
* **Molecular subtype classification**
* **Missingness-robust clinical representation learning**

The framework follows a **lesion-localization-first diagnostic paradigm**, where segmentation-derived probability maps act as **soft spatial priors** to guide lesion-aware feature aggregation and multimodal fusion. 

Key ideas:

* **Soft ROI-guided pooling**
* **Lesion-conditioned clinical imputation**
* **Reliability-aware multimodal gating**
* **Uncertainty-aware tri-task optimization**

Experiments across **five datasets** demonstrate strong segmentation accuracy, class-balanced discrimination, and robustness to clinical missingness. 

---

# Repository Structure

```
SG-MTF
│
├── datasets
│   ├── dualtask_dataset.py       # multimodal dataset loader
│   └── transforms.py             # image augmentation & preprocessing
│
├── models
│   ├── backbones                 # segmentation encoder (GroupMixFormer)
│   ├── fusion                    # multimodal fusion modules
│   ├── heads                     # segmentation & classification heads
│   ├── roi                       # ROI-guided pooling
│   ├── __init__.py
│   └── sgmtf.py                  # main SG-MTF architecture
│
├── engines
│   ├── losses.py                 # multi-task loss functions
│   └── train_eval.py             # training & evaluation loops
│
├── utils
│   ├── meters.py                 # training meters
│   ├── metrics_seg.py            # Dice / mIoU
│   ├── metrics_cls.py            # ACC / F1 / AUC
│   ├── roc.py
│   ├── preprocess_fold.py        # fold-wise preprocessing
│   ├── optim.py                  # optimizer & scheduler
│   └── seed.py                   # reproducibility utilities
│
├── scripts
│   ├── train_dual.py             # main training script
│   └── checkpoints_sgmtf         # saved models
│
├── DualTask_Data                 # dataset directory (not included)
│
├── requirements.txt
└── README.md
```

---

# Installation

## 1. Clone repository

```bash
git clone https://github.com/xxx/SG-MTF.git
cd SG-MTF
```

---

## 2. Create environment

Recommended: **Python 3.10 + PyTorch ≥ 2.1**

```bash
conda create -n sgmtf python=3.10
conda activate sgmtf
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Example dependencies:

```
torch
torchvision
numpy
pandas
scikit-learn
opencv-python
matplotlib
tqdm
```

---

# Dataset Preparation

Due to clinical privacy, the datasets used in the paper are **not distributed**.

However, SG-MTF expects the following structure.

```
DualTask_Data
│
├── images
│   ├── 001.bmp
│   ├── 002.bmp
│   └── ...
│
├── masks
│   ├── 001.png
│   ├── 002.png
│   └── ...
│
└── clinical.xlsx
```

### images

Breast ultrasound images.

```
format: bmp / png / jpg
size: resized to 256×256 during preprocessing
channels: grayscale
```

---

### masks

Binary segmentation masks.

```
0 = background
1 = lesion
```

---

### clinical.xlsx

Each row corresponds to one patient.

Example:

| pid | age | tumor_size | ER | PR | Ki67 | histology | label |
| --- | --- | ---------- | -- | -- | ---- | --------- | ----- |

Notes:

* missing values should be **NaN**
* numerical features are **z-score normalized**
* categorical variables are **one-hot encoded**
* missingness mask is automatically constructed during loading

---

# Training

The model is trained using **5-fold stratified cross-validation**.

Example:

```bash
python scripts/train_dual.py \
    --data_root DualTask_Data \
    --image_dir images \
    --mask_dir masks \
    --clinical_file clinical.xlsx \
    --num_classes 3 \
    --batch_size 8 \
    --epochs 100 \
    --lr 1e-4
```

Training configuration (paper setting):

| Parameter        | Value             |
| ---------------- | ----------------- |
| image size       | 256×256           |
| optimizer        | AdamW             |
| learning rate    | 1e-4              |
| weight decay     | 1e-4              |
| epochs           | 100               |
| scheduler        | cosine annealing  |
| early stopping   | patience=12       |
| cross validation | 5-fold stratified |

---

# Evaluation

After training, evaluate using the saved checkpoint.

```
python scripts/train_dual.py \
    --eval \
    --checkpoint checkpoints_sgmtf/best_model_fold1.pth
```

Metrics:

### Segmentation

* Dice
* mIoU

### Classification

* Accuracy
* Macro-F1
* Macro-AUC

All results are reported as **mean ± std across 5 folds**.

---

# Method Overview

SG-MTF contains three main components.

### 1️⃣ Segmentation-guided spatial prior

A segmentation decoder generates a **probability map** that serves as a soft ROI prior.

```
Proi = sigmoid(conv1x1(Fseg))
```

---

### 2️⃣ Soft ROI-guided feature aggregation

Visual features are aggregated using ROI weights:

```
vimg = Σ(Fcls * Proi) / (Σ(Proi) + ε)
```

This suppresses background artifacts and enforces lesion-focused representation learning.

---

### 3️⃣ Missingness-robust clinical modeling

Clinical variables are encoded together with their observation mask.

```
hclin = ψ([cobs ⊙ m, m])
```

Missing variables are reconstructed using lesion features.

```
ĉ = fimp([hclin, vimg])
```

---

### 4️⃣ Reliability-aware multimodal fusion

Cross-modal gating dynamically reweights modality contributions:

```
gimg = σ(Wimg[vclin*, r])
gclin = σ(Wclin vimg)
```

---

### 5️⃣ Uncertainty-aware tri-task optimization

The network jointly optimizes:

* segmentation
* classification
* clinical reconstruction

```
Ltotal =
1/(2σ_seg²) Lseg +
1/(2σ_cls²) Lcls +
1/(2σ_imp²) Limp +
λcons Lcons
```

---

# Reproducing Paper Results

Train the full model:

```
python scripts/train_dual.py --config configs/sgmtf.yaml
```

Expected performance (HER2USC dataset):

| Metric   | Value |
| -------- | ----- |
| Dice     | 0.742 |
| mIoU     | 0.676 |
| ACC      | 0.832 |
| Macro-F1 | 0.820 |
| AUC      | 0.894 |

---

# Citation

If you use this code, please cite our paper.

```
@article{Ye2026SGMTF,
title={A Segmentation-Guided Multi-Task Framework Using Multimodal Ultrasound Images and Clinicopathological Data for Breast Lesion Segmentation and Molecular Subtyping},
author={Ye, Jinlin and Hu, Deming and Ge, Zhongyu and Liu, Yuhan and Yang, Liang and Ren, Shangjie and Wang, Changjun and Zhou, Yidong and Zhang, Wei},
journal={Information Fusion},
year={2026}
}
```

---

# Acknowledgements

This implementation builds upon several open-source libraries:

* PyTorch
* GroupMixFormer backbone
* Medical segmentation frameworks

---

# License

This project is released under the **MIT License**.

---