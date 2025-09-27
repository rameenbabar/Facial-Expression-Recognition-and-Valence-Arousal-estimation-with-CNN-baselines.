# Facial Affect Recognition with CNN Baselines

This repository contains the implementation of **CNN baselines for facial affect analysis** 
The task involves **facial expression recognition (8 classes)** and **continuous affect prediction (valence & arousal in [-1,1])**, using multiple CNN backbones.

---

## Overview

Facial affect recognition combines **categorical classification** (neutral, happy, sad, surprise, fear, disgust, anger, contempt) with **continuous prediction** of **valence** (pleasant/unpleasant) and **arousal** (calm/excited).  

We implement and compare **CNN baselines**: 
- EfficientNet-B0  
- MobileNetV3-Large  
- DenseNet121  
- VGG16  

Each model is extended into a **multi-task architecture**:
- **Classification head** → predicts the 8 discrete emotions.  
- **Regression head** → predicts valence & arousal values.  

Training is performed on **Google Colab with GPU** (recommended, since training is very slow on CPU).

---

## 📂 Dataset

The dataset includes: 
- Cropped 224×224 RGB face images.  
- Expression labels (0–7).  
- Valence & arousal annotations in [-1, +1].  
- Facial landmarks. 

Annotations are stored as `.npy` files (`_exp.npy`, `_val.npy`, `_aro.npy`).  

---

## Evaluation Metrics

**Classification:**  
- Accuracy  
- F1-Score (Macro)  
- Cohen’s Kappa  
- Krippendorff’s Alpha  
- AUC (ROC)  
- AUC-PR  

**Regression (Valence & Arousal):**  
- RMSE  
- Pearson Correlation (CORR)  
- Sign Agreement (SAGR)  
- Concordance Correlation Coefficient (CCC)  

---

## 📒 Notebooks

- `EfficientNet_MobileNet.ipynb` and `DenseNet_VGG16.ipynb`→ Main training & evaluation notebook.  

---

## ⚡ How to Run

1. Clone the repo:
   ```bash
   git clone https://github.com/<your-username>/Facial-Expression-Recognition-and-Valence–Arousal-estimation-with-CNN-baselines.git
   cd Facial-Expression-Recognition-and-Valence–Arousal-estimation-with-CNN-baselines
   ```
