# Energy-Based Test-Time Adaptation for Object Detection

<p align="center">
  <img src="model.png" width="1200">
</p>

## Overview

This repository implements a new framework for robust object detection under domain shifts and image corruptions.

---

## Contents
1. [Installation](#installation)
2. [Dataset Preparation](#dataset-preparation)
3. [Project Structure](#project-structure)
4. [Usage](#usage)
   - [Training the Energy Model](#1-training-the-energy-model)
   - [Running Adaptation](#2-running-adaptation)
   - [Visualization](#3-visualization)
5. [Acknowledgements](#acknowledgements)

---

## Installation

**Requirements:**
- Python 3.8
- PyTorch 1.9.0 (CUDA 11.1)
- Detectron2

**Setup:**
```bash
# Create conda environment
conda create -n ETAOD python=3.8
conda activate ETAOD

# Install PyTorch
pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 \
    -f https://download.pytorch.org/whl/torch_stable.html

# Install other dependencies
pip install -r requirements.txt

# Install Detectron2
cd detectron2
python -m pip install -e .
```

---

## Dataset Preparation

### 1. Download Cityscapes
Download from [Cityscapes official website](https://www.cityscapes-dataset.com/downloads/):
- `leftImg8bit_trainvaltest.zip` (11GB) - Images
- `gtFine_trainvaltest.zip` (241MB) - Annotations

### 2. Generate Corrupted Images
Apply corruptions to the validation set using [imagecorruptions](https://github.com/bethgelab/imagecorruptions):

```bash
python preprocessing/corrupt.py \
    --input datasets/cityscapes/leftImg8bit/val \
    --output datasets/{corruption_type}/leftImg8bit/val \
    --corruption {corruption_type} \
    --severity 5
```

Supported corruptions: `motion_blur`, `defocus_blur`, `snow`, `fog`, `frost`, `brightness`, etc.

### 3. Directory Structure
```
eta-object-detection/
├── datasets/
│   ├── cityscapes/
│   │   ├── leftImg8bit/
│   │   │   ├── train/
│   │   │   └── val/
│   │   └── annotations/
│   │       ├── instancesonly_filtered_gtFine_train.json
│   │       └── instancesonly_filtered_gtFine_val.json
│   ├── motion_blur/
│   │   └── leftImg8bit/val/
│   ├── defocus_blur/
│   │   └── leftImg8bit/val/
│   └── snow/
│       └── leftImg8bit/val/
└── detectron2/
    └── tools/output/res50_fbn_1x/
        └── cityscapes_train_final.pth
```

### 4. Download Pretrained Weights
Download the Cityscapes-trained Faster R-CNN weights from [this link](https://drive.google.com/file/d/1pjnmfRzz9zL_CuT-bXfR5W8J0KGw9Va4/view?usp=sharing) and place in:
```
detectron2/tools/output/res50_fbn_1x/cityscapes_train_final.pth
```
---

## Project Structure

```
eta-object-detection/
├── preprocessing/
│   ├── cityscapes_to_coco.py    # Convert Cityscapes → COCO format
│   └── corrupt.py                # Apply corruptions to images
├── test/
│   ├── run_detectron.py          # Run baseline detection
│   ├── run_energy_model_on_test_image.py  # Test energy model
│   └── visualize_adaptation.py   # Visualize adaptation results
├── train_energy_model.py         # Train ROI energy model
├── adaptation.py                 # Run BN + energy adaptation
└── models/                       # Saved energy models
```

---

## Usage

### 1. Training the Energy Model

The energy model learns to predict detection quality by training on paired clean/corrupted images.

**Edit Configuration:**
```python
# In train_energy_model.py, set:
CORRUPTION = "motion_blur"  # or "defocus_blur", "snow", "fog", etc.
CLEAN_DIR = "datasets/cityscapes/leftImg8bit/train"
BLUR_DIR = f"datasets/{CORRUPTION}/leftImg8bit/train"
```

**Run Training:**
```bash
python train_energy_model.py
```

**Key Parameters:**
- `num_epochs`: Training epochs (default: 2)
- `learning_rate`: Adam LR (default: 5e-4)
- `temperature`: Gibbs transform temperature for target energy
  - Motion blur: 200
  - Other corruptions: 200
- `batch_accum`: Gradient accumulation steps (default: 32)

**Output:**
- Model checkpoint: `models/{corruption}_roi_energy_model_epoch{N}.pth`
- TensorBoard logs: `runs/energy_model/`

**Monitor Training:**
```bash
tensorboard --logdir=runs/energy_model
```

---

### 2. Running Adaptation

Perform test-time adaptation using the trained energy model.

**Edit Configuration:**
```python
# In adaptation.py, set:
CORRUPTION = "motion_blur"
config = {
    'image_dir': f'datasets/{CORRUPTION}/leftImg8bit/val',
    'energy_model_path': f'models/{CORRUPTION}_roi_energy_model_epoch2.pth',
    'adaptation_lr': 2.0,  # See tuning guide below
    'iterations_per_image': 1,
}
```

**Run Adaptation:**
```bash
python adaptation.py
```

**Output:**
- Console: Per-image and global mAP metrics
- TensorBoard logs: `runs/bn_energy_adaptation_{timestamp}/`

**Monitor Results:**
```bash
tensorboard --logdir=runs/bn_energy_adaptation_{timestamp}
```

---

### 3. Visualization

Visualize detection results before and after adaptation:

```bash
python test/visualize_adaptation.py
```

This generates side-by-side comparison images showing:
- Original corrupted image
- Detections before adaptation
- Detections after adaptation

---

## Acknowledgements

This work builds upon several excellent open-source projects:

- **[Detectron2](https://github.com/facebookresearch/detectron2)** - Facebook AI Research's object detection framework
- **[AMROD](https://github.com/ShileiCao/AMROD)** - Adaptive Multi-Resolution Object Detection baseline
- **[Cityscapes Dataset](https://www.cityscapes-dataset.com/)** - Urban street scene dataset

## Contact

For questions or issues, please open a GitHub issue or contact [your email].