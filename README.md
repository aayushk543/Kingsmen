# Kingsmen
# Robust Vision Challenge  
### Adaptation Under Distributional Shift and Label Noise

This repository implements a robust vision pipeline designed to handle:

- ✅ Noisy labels (30% symmetric noise)
- ✅ Label shift (changing class priors)
- ✅ Covariate shift (sensor corruption)
- ✅ Test-time adaptation
- ✅ Generalization to unseen corruptions

The solution follows a **three-phase engineering protocol**:  
Robust Training → Label Shift Estimation → Test-Time Adaptation.

---

# 🚀 Project Overview

In real-world deployment scenarios (e.g., manufacturing or edge vision systems), models must handle:

1. **Data Poisoning** – Incorrect labels in training data  
2. **Sensor Degradation** – Corrupted input images  
3. **Population Drift** – Shift in class distribution  

This repository implements a robust pipeline that addresses all three challenges without human intervention.

---

# 🏗️ Methodology

## Phase 1 — Robust Training (Decontamination)

Standard Cross Entropy overfits noisy labels.  
We instead use:

### 🔹 Generalized Cross Entropy (GCE)
Balances between CE and MAE for noise robustness.

\[
L_q = \frac{1 - p_y^q}{q}
\]

**Additional Techniques**
- Label smoothing
- Standard data augmentation (no corruption-specific tuning)
- Early stopping using clean validation set

---

## Phase 2 — Label Shift Estimation

Target class priors differ from source.

We estimate target class weights using:

### 🔹 Confusion Matrix Method

1. Train model on source data  
2. Compute confusion matrix on clean validation set  
3. Estimate predicted class distribution on target  
4. Solve:

\[
C w = \hat{p}_t
\]

Where:
- `C` = normalized confusion matrix  
- `w` = estimated class prior weights  

These weights are used to correct logits at inference:

\[
\text{logits} = \text{logits} + \log(w)
\]

---

## Phase 3 — Test-Time Adaptation (Alignment)

Target images contain unknown corruption.

We adapt the model by:

### 🔹 BatchNorm Statistics Adaptation
- Freeze weights
- Update BatchNorm running statistics on target stream

### 🔹 Entropy Minimization
Minimize prediction entropy at test time:

\[
H(p) = -\sum p \log p
\]

This encourages confident predictions under shift.

---
