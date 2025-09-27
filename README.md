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

## How to Run

1. Clone the repo:
   ```bash
   git clone https://github.com/<your-username>/Facial-Expression-Recognition-and-Valence–Arousal-estimation-with-CNN-baselines.git
   cd Facial-Expression-Recognition-and-Valence–Arousal-estimation-with-CNN-baselines
   ```
---

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

---

3. Open the notebook in Google Colab, mount your Google Drive, and set `DATA_ROOT` to point to your dataset folder.

## Results

### Categorical Classification

<table>
  <tr>
    <th>Model</th><th>ACC</th><th>F1_macro</th><th>Kappa</th>
    <th>Alpha</th><th>AUC_macro</th><th>AUPR_macro</th>
  </tr>
  <tr>
    <td>EfficientNet-B0</td>
    <td>0.4708</td><td>0.4705</td><td>0.3954</td><td>0.3940</td>
    <td><b>0.8378</b></td><td><b>0.4862</b></td>
  </tr>
  <tr>
    <td>MobileNetV3-Large</td>
    <td><b>0.4858</b></td><td><b>0.4798</b></td><td><b>0.4114</b></td><td><b>0.4097</b></td>
    <td>0.8179</td><td>0.4617</td>
  </tr>
</table>

<p><i>MobileNetV3-Large slightly outperforms EfficientNet-B0 on classification accuracy and F1, 
while EfficientNet-B0 shows stronger AUC and AUPR, suggesting better ranking ability for imbalanced data.</i></p>

---

## Continuous Metrics

<table>
  <tr>
    <th>Model</th><th>RMSE_val</th><th>RMSE_aro</th><th>CORR_val</th><th>CORR_aro</th>
    <th>SAGR_val</th><th>SAGR_aro</th><th>CCC_val</th><th>CCC_aro</th>
  </tr>
  <tr>
    <td>EfficientNet-B0</td>
    <td><b>0.3772</b></td><td>0.3511</td><td><b>0.6121</b></td><td><b>0.4881</b></td>
    <td><b>0.7629</b></td><td>0.7496</td><td><b>0.5783</b></td><td><b>0.4040</b></td>
  </tr>
  <tr>
    <td>MobileNetV3-Large</td>
    <td>0.4027</td><td><b>0.3487</b></td><td>0.5435</td><td>0.3922</td>
    <td>0.7579</td><td><b>0.7913</b></td><td>0.4803</td><td>0.3442</td>
  </tr>
</table>

<p><i>EfficientNet-B0 achieves lower RMSE, higher correlation, and stronger concordance on valence/arousal, 
indicating better regression performance. MobileNet, however, shows stronger SAGR for arousal, 
capturing correct polarity more often.</i></p>

---

## Speed–Accuracy Trade-off

<table>
  <tr>
    <th>Model</th><th>Best ACC</th><th>Training Time (10 epochs)</th>
  </tr>
  <tr>
    <td>EfficientNet-B0</td>
    <td>0.4708 (47%)</td><td>46.5 min</td>
  </tr>
  <tr>
    <td>MobileNetV3-Large</td>
    <td><b>0.4858 (~49%)</b></td><td><b>12.1 min</b></td>
  </tr>
</table>

<p><i>MobileNetV3-Large trains nearly 4× faster than EfficientNet-B0 while delivering slightly 
better classification accuracy, making it a strong candidate for real-time or resource-constrained settings.</i></p>

---


