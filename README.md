# HAM10000 Skin Lesion Classification

![Python](https://img.shields.io/badge/Python-3.11-blue) ![PyTorch](https://img.shields.io/badge/PyTorch-1.15-red) ![Timm](https://img.shields.io/badge/timm-0.9-green)

This repository contains a **two-phase deep learning pipeline** for classifying skin lesion images from the [HAM10000 dataset](https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000). The project uses **Supervised Contrastive Learning** for feature representation and a fine-tuned classifier with **Focal Loss** for handling class imbalance.

---

## Table of Contents
- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Pipeline](#pipeline)
- [Requirements](#requirements)
- [Usage](#usage)
- [Results](#results)
- [License](#license)

---

## Project Overview
Skin cancer detection is a critical problem in medical imaging. This project implements a **deep learning approach** to classify skin lesions into seven categories:  

- akiec  
- bcc  
- bkl  
- df  
- mel  
- nv  
- vasc  

The pipeline is divided into **two phases**:

1. **Phase 1: Supervised Contrastive Pretraining**
   - Input images → Data augmentation → Encoder (EfficientNetB0) → Projection head (128-D features)  
   - Loss: **Supervised Contrastive Loss**  
   - Purpose: Learn robust and discriminative feature representations

2. **Phase 2: Fine-tuning Classifier**
   - Frozen/partially frozen encoder → MLP classifier → Prediction  
   - Loss: **Focal Loss** (γ=3) to handle class imbalance  
   - Optimizer: Adam with low learning rate for fine-tuning

---

## Dataset
- **HAM10000 Dataset**: A large collection of multi-source dermatoscopic images of common pigmented skin lesions.  
- Contains images and metadata CSV file.  

**Folder structure:**
```bash
HAM10000/
├── HAM10000_images_part_1/
├── HAM10000_images_part_2/
└── HAM10000_metadata.csv
```

---

## Pipeline Overview
```text
Phase 1:
image -> augment -> EfficientNet encoder -> projection (128-D) -> SupCon Loss -> 1792-D embeddings

Phase 2:
encoder (frozen/fine-tuned) -> MLP classifier -> Focal Loss -> predictions
```

## Requirements
```bash
torch
torchvision
timm
albumentations
opencv-python
scikit-learn
pandas
numpy
```

## Install all dependencies via:
```bash
pip install -r requirements.txt
```


## Usage
# 1. Clone repository:
```bash
git clone https://github.com/username/ham10000-classification.git
cd ham10000-classification
```
# 2. Prepare dataset in HAM10000/ folder.
# 3. Run Phase 1 (Encoder Pretraining):
```bash
# Execute your phase 1 code in train_encoder.py or notebook
python train_encoder.py
```
# 4. Run Phase 2 (Classifier Fine-tuning):
```bash
# Execute classifier training
python train_classifier.py
```
# 5. Evaluate model:
```bash
# Use your test/validation loader for predictions
python evaluate.py
```

## Results

**Validation Metrics:**

| Class  | Precision | Recall | F1-score |
|--------|-----------|--------|----------|
| bkl    | 0.71      | 0.82   | 0.76     |
| nv     | 0.97      | 0.89   | 0.93     |
| df     | 0.54      | 0.65   | 0.59     |
| mel    | 0.68      | 0.74   | 0.71     |
| vasc   | 0.95      | 0.75   | 0.84     |
| bcc    | 0.68      | 0.93   | 0.79     |
| akiec  | 0.62      | 0.72   | 0.67     |

**Overall Accuracy:** 86%  
**Macro F1-score:** 0.75






## Author

**Md Nahidul Islam**  
**GitHub:** [leon-dream1](https://github.com/leon-dream1)







