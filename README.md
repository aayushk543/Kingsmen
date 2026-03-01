# 🛡️ Hackenza 2026: An Autonomous Pipeline for Robust Vision Adaptation

**Team Kingsmen:** Aayush Kushwaha, Anshul D Jain, Vaiebhav Shreevarshan R, Divyam Agarwal
**Track:** Applied Machine Learning - The Robust Vision Challenge

## 📖 Abstract
The transition of computer vision models from curated datasets to autonomous deployment introduces a "triad of failure" that standard machine learning pipelines are ill-equipped to handle. This repository contains an end-to-end, self-correcting vision pipeline engineered to achieve high accuracy on clean validation sets despite training on toxic data, while simultaneously adapting to unknown sensor noise and estimating target class distributions on the fly. 

Built strictly from scratch without pre-trained weights or training-time augmentations, this system mimics the messy reality of production-grade AI.

---

## ⚠️ The Triad of Failure (Problem Statement)
This pipeline is explicitly designed to navigate three simultaneous failure vectors:

* **Data Poisoning (The Training Barrier):** 30% of the labels in the training dataset (`source_toxic.pt`) are incorrect. A standard training approach treats this noise as ground truth, forcing the model to learn patterns of error rather than actual features.
* **Population Drift (The Logic Barrier):** Class ratios in the target environment are unknown and heavily skewed. The system actively estimates the target class weights on the fly to avoid biased predictions.
* **Sensor Degradation (The Input Barrier):** The target stream introduces severe covariate shifts (e.g., Impulse Noise). The model must remain functional when image quality drops significantly from the training baseline.

---

## 🏗️ Core Architecture: Custom ResNet-9
To survive these hostile conditions, we engineered a custom ResNet-9 architecture optimized for 28x28 grayscale Fashion-MNIST images. 

* **Optimal Capacity:** Deep enough to learn complex spatial features, but shallow enough to resist memorizing the 30% label noise.
* **Group Normalization (GN):** Standard Batch Normalization fails during Test-Time Adaptation if the target stream is skewed. GN decouples normalization from batch statistics, enabling stable, single-sample inference at test time.
* **Real-Time Adaptation Speed:** Offers the low-latency overhead necessary for online Sharpness-Aware Minimization (SAM) updates without the massive computational weight of deeper models like ResNet-50.

---

## ⚙️ The 3-Phase Operational Pipeline

### Phase 1: Robust Decontamination (Training)
We decontaminate the training process using an **Active-Passive Loss (APL)** framework rather than standard Categorical Cross-Entropy, which overfits to noisy labels.
* **Active Loss (Generalized Cross Entropy - GCE):** Provides the necessary gradient scaling to drive rapid convergence on the structural features of clothing items.
* **Passive Loss (Normalized Cross Entropy):** Constrains global risk, ensuring the model cannot be dragged toward poisoned labels.

### Phase 2: Distribution Reconnaissance
To combat Population Drift (Label Shift), we implement a **Black Box Shift Estimation (BBSE-Soft)** algorithm.
* Generates a soft confusion matrix using continuous softmax probabilities over the clean `val_sanity.pt` anchor.
* Calculates the target marginal over the completely unlabelled `target_static.pt` stream.
* Extracts the target population weights using **Tikhonov Regularization (Ridge Regression)** to guarantee matrix invertibility and numerical stability.

### Phase 3: Test-Time Adaptation (Alignment)
The pipeline combats continuous sensor degradation in real-time using **Sharpness-Aware and Reliable (SAR)** entropy minimization.
* **Label-Shift-Aware Posterior Correction:** Model predictions are dynamically scaled using the BBSE-derived weights to ensure updates respect the detected population drift.
* **Reliable Sample Selection:** A dynamic Shannon entropy filter discards unrecoverable, noise-induced inputs, preserving decontaminated geometric features.
* **Sharpness-Aware Minimization (SAM):** Optimizes frozen Conv2d features and active GN affine parameters toward mathematically flat regions of the loss landscape, guaranteeing generalization to novel, unseen corruptions.

---

## 🚀 Execution & Compliance
* **Strict Adherence:** Zero pre-trained weights (random initialization), zero external clean datasets, and zero training-time augmentations (e.g., AugMix).
* **Framework:** Python 3.10+ & PyTorch.
* **Hardware:** Optimized for dual NVIDIA T4 GPUs via Kaggle.

---

## 📚 References
* [1] Z. Zhang and M. Sabuncu, "Generalized Cross Entropy Loss for Training Deep Neural Networks with Noisy Labels," *NeurIPS*, 2018.
* [2] Z. Lipton et al., "Detecting and Correcting for Label Shift with Black Box Predictors," *ICML*, 2018.
* [3] S. Niu et al., "Towards Stable Test-Time Adaptation in Dynamic Wild World," *ICLR*, 2023.
