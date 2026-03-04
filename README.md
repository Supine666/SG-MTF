SG-MTF
Segmentation-Guided Multi-Task Framework for Breast Lesion Segmentation and Molecular Subtyping

Official PyTorch implementation of the paper:

“A Segmentation-Guided Multi-Task Framework Using Multimodal Ultrasound Images and Clinicopathological Data for Breast Lesion Segmentation and Molecular Subtyping”

Overview

Breast cancer diagnosis often requires integrating ultrasound imaging with clinicopathological variables. However, most multimodal learning frameworks rely on global image features without explicit spatial constraints, which limits their ability to capture lesion-specific morphological cues.

This repository implements SG-MTF, a segmentation-guided multimodal multi-task framework that jointly performs:

Breast lesion segmentation

Molecular subtype classification

Missingness-robust clinical representation learning

The framework follows a lesion-localization-first diagnostic paradigm, where segmentation-derived probability maps act as soft spatial priors to guide lesion-aware feature aggregation and multimodal fusion.

Key ideas of SG-MTF include:

Segmentation-guided multimodal reasoning, mimicking the clinical diagnostic workflow

Soft spatial priors for lesion-aware feature aggregation

Missingness-robust clinical representation learning for incomplete tabular inputs

Uncertainty-aware tri-task optimization for stable multimodal learning

Experiments across five datasets demonstrate strong segmentation accuracy, class-balanced subtype discrimination, and robustness to clinical missingness.

Repository Structure
SG-MTF
│
├── datasets
│   ├── __init__.py
│   ├── sgmtf_dataset.py        # multimodal dataset loader
│   └── transforms.py           # image augmentation & preprocessing
│
├── models
│   ├── backbones               # segmentation encoder
│   ├── fusion                  # multimodal fusion modules
│   ├── heads                   # segmentation & classification heads
│   ├── roi                     # ROI-guided pooling
│   ├── __init__.py
│   └── sgmtf.py                # main SG-MTF architecture
│
├── engines
│   ├── losses.py               # multi-task loss functions
│   └── train_eval.py           # training & evaluation loops
│
├── utils
│   ├── meters.py               # training meters
│   ├── metrics_seg.py          # Dice / mIoU
│   ├── metrics_cls.py          # ACC / F1 / AUC
│   ├── roc.py
│   ├── preprocess_fold.py      # fold-wise preprocessing
│   ├── optim.py                # optimizer & scheduler
│   └── seed.py                 # reproducibility utilities
│
├── scripts
│   └── run_cv.py               # 5-fold cross-validation runner
│
├── data                        # dataset directory (not included)
│
├── requirements.txt
└── README.md
Installation
1. Clone repository
git clone https://github.com/Supine666/SG-MTF.git
cd SG-MTF
2. Create environment

Recommended environment:

Python 3.10

PyTorch ≥ 2.1

conda create -n sgmtf python=3.10
conda activate sgmtf

Install dependencies:

pip install -r requirements.txt

Example dependencies include:

torch
torchvision
numpy
pandas
scikit-learn
opencv-python
matplotlib
tqdm
Dataset Preparation

Due to clinical privacy restrictions, the datasets used in the paper are not publicly distributed.

However, SG-MTF expects the following dataset structure:

data
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
Ultrasound Images

format: bmp / png / jpg

modality: breast ultrasound

resized to 256×256 during preprocessing

grayscale images

Segmentation Masks

Binary lesion segmentation masks:

0 = background
1 = lesion
Clinical Data

The clinical.xlsx file contains tabular clinicopathological variables.

Example:

pid	age	tumor_size	ER	PR	Ki67	histology	label

Notes:

missing values should be stored as NaN

numerical variables are z-score normalized

categorical variables are one-hot encoded

missingness masks are automatically constructed during loading

Training and Evaluation

The model is trained using 5-fold stratified cross-validation.

Each fold is automatically trained and evaluated, and the final results are reported as mean ± standard deviation across folds.

Example:

python scripts/run_cv.py \
    --data_root data \
    --image_dir images \
    --mask_dir masks \
    --clinical_file clinical.xlsx \
    --num_classes 3 \
    --batch_size 8 \
    --epochs 100 \
    --lr 1e-4
Training Configuration (Paper Setting)
Parameter	Value
image size	256×256
optimizer	AdamW
learning rate	1e-4
weight decay	1e-4
epochs	100
scheduler	cosine annealing
early stopping	patience = 12
cross validation	5-fold stratified
Evaluation Metrics
Segmentation

Dice

mIoU

Classification

Accuracy

Macro-F1

Macro-AUC

All results are reported as:

mean ± standard deviation across the five folds
Method Overview

SG-MTF consists of three major components.

1. Segmentation-Guided Spatial Prior

A segmentation decoder generates a probability map that serves as a soft ROI prior:

Proi = sigmoid(conv1x1(Fseg))
2. Soft ROI-Guided Feature Aggregation

Visual features are aggregated using ROI weights:

vimg = Σ(Fcls * Proi) / (Σ(Proi) + ε)

This suppresses background artifacts and enforces lesion-focused representation learning.

3. Missingness-Robust Clinical Modeling

Clinical variables are encoded together with their observation masks:

hclin = ψ([cobs ⊙ m, m])

Missing variables are reconstructed using lesion features:

ĉ = fimp([hclin, vimg])
4. Reliability-Aware Multimodal Fusion

Cross-modal gating dynamically reweights modality contributions:

gimg = σ(Wimg[vclin*, r])
gclin = σ(Wclin vimg)
5. Uncertainty-Aware Tri-Task Optimization

The network jointly optimizes:

segmentation

classification

clinical reconstruction

Ltotal =
1/(2σ_seg²) Lseg +
1/(2σ_cls²) Lcls +
1/(2σ_imp²) Limp +
λcons Lcons
Citation

If you find this code useful, please cite our work:

@article{YeSGMTF2026,
title={A Segmentation-Guided Multi-Task Framework Using Multimodal Ultrasound Images and Clinicopathological Data for Breast Lesion Segmentation and Molecular Subtyping},
author={Ye, Jinlin and Hu, Deming and Ge, Zhongyu and Liu, Yuhan and Yang, Liang and Ren, Shangjie and Wang, Changjun and Zhou, Yidong and Zhang, Wei},
note={Under review},
year={2026}
}
Acknowledgements

This implementation is developed using the PyTorch deep learning framework and standard scientific computing libraries in Python.

License

This project is released under the MIT License.