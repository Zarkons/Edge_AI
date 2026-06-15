# Edge AI Anomaly Detection (Predictive Maintenance)

This sub-project implements an unsupervised **Autoencoder Neural Network** for real-time anomaly detection on embedded devices (MacBook / Raspberry Pi 5). 

Instead of training a model on every possible way a machine can break, this network is trained **only on normal operating conditions**. It learns to compress and reconstruct normal data. When unusual data is encountered, the reconstruction fails, causing the metric error (**Loss**) to spike and trigger an alert.

---

## 🧠 Architectural Overview

