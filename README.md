# GSA-YOLO: A High-Efficiency Framework via Structured Sparsity and Adaptive Knowledge Distillation for Real-Time X-ray Security Inspection

[![Pytorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![YOLOv8](https://img.shields.io/badge/YOLO-v8.3-green.svg)](https://github.com/ultralytics/ultralytics)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Official implementation of the paper **"GSA-YOLO: A High-Efficiency Framework via Structured Sparsity and Adaptive Knowledge Distillation for Real-Time X-ray Security Inspection"**.

---

## ✨ Highlights
- 🚀 **High Efficiency**: Achieves **189.62 FPS** on NVIDIA RTX 3090.
- 📉 **Model Slimming**: Reduces GFLOPs from **8.7G to 8.0G** while improving accuracy.
- 🎯 **Robust Performance**: Improves mAP50:95 by **2.4%** (HiXray) and **1.8%** (PIDray) over the YOLOv8n baseline.
- 🛠️ **Three Core Modules**: 
  - **Group Lasso (GL)** for feature refinement in the Neck.
  - **Sparse Structure Selection (SSS)** for hard pruning in the Head.
  - **Adaptive Knowledge Distillation (Ada-KD)** for comprehensive accuracy recovery.

---

## 🏗️ Framework Overview

GSA-YOLO is built upon the YOLOv8n architecture and strategically integrates structured sparsity and knowledge transfer.

![Framework](figures/baseline.png) 
*Figure 1: The detailed GSA-YOLO framework integrating GL, SSS, and Ada-KD modules.*

---

## 📊 Main Results

### Performance on HiXray and PIDray Datasets

| Model | GFLOPs | FPS (Inf) | mAP50 (HiXray) | mAP50:95 (HiXray) | mAP50:95 (PIDray) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| YOLOv8n (Baseline) | 8.7G | 170.58 | 0.806 | 0.507 | 0.661 |
| **GSA-YOLO (Ours)** | **8.0G** | **189.62** | **0.827** | **0.531** | **0.679** |

### Visual Comparisons

![Visual Comparison](figures/case_study.png)
*Figure 2: Comparative inference results on challenging X-ray scenarios (HiXray & PIDray).*

---

## ⚙️ Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/KJHkong/GSA-YOLO.git
   cd GSA-YOLO

🚀 Quick Start
1. Training with Sparsity (GL & SSS)
To train the model with Group Lasso and Sparse Structure Selection:

python train.py --cfg gsa-yolo.yaml --data hixray.yaml --sparsity --beta 1e-4 --gamma 1e-3

2. Adaptive Knowledge Distillation
After pruning, use Ada-KD to recover accuracy:

python train_distill.py --teacher yolov8m.pt --student pruned_model.pt --lambda0 4 --theta 15<img width="818" height="701" alt="3 Framework" src="https://github.com/user-attachments/assets/bc342432-0a07-419e-9e92-bb6e182aa01a" />
