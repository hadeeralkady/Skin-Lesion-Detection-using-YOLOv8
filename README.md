# Is Segmentation Necessary? A Detection-Guided Classification Framework for Skin Lesion Analysis

[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/release/python-3120/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.5.0-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/get-started/locally/)
[![YOLOv8](https://img.shields.io/badge/YOLO-v8-00ffff?logo=ultralytics)](https://github.com/ultralytics/ultralytics)
[![Streamlit](https://img.shields.io/badge/Deployment-Streamlit-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)

---

## 📌 Abstract
Traditional skin lesion analysis pipelines are often hindered by the high computational and annotation costs of pixel-level segmentation. This project investigates a critical research question: **Is explicit segmentation truly necessary for high-accuracy diagnosis?** 

We propose a **segmentation-free, detection-guided framework** that integrates **YOLOv8** for precise lesion localization and **ResNet18** for multi-class categorization (Melanoma, Nevus, Benign Keratosis). By treating localization as an object detection task, we achieve efficient, real-time diagnostic performance without the fragility of traditional segmentation masks.

---

## ⚠️ Medical Disclaimer
This project is intended strictly for **research and educational purposes**. It is not a medical device and must not be used for clinical diagnosis, treatment, or medical decision-making.

---

## 🧠 System Architecture
The framework is designed as a modular hierarchical pipeline that mimics a specialist's focus on pathological regions.

1.  **Stage 1: Localization (YOLOv8):** An anchor-free detector identifies the lesion boundaries and generates coordinates.
2.  **Stage 2: ROI bridge:** The system dynamically crops and resizes the detected region to 224x224 pixels.
3.  **Stage 3: Diagnostic Classification (ResNet18):** A Deep Residual Network performs the final diagnosis on the localized Region of Interest (ROI).

---

## 🚀 Key Research Contributions
*   **Adaptive Small-Object Scaling:** Implemented a **2.0x scaling factor** for small-scale lesions to preserve peripheral skin context, critical for border irregularity analysis.
*   **Automated Label Engineering:** Developed custom scripts to transform **ISIC 2018 Task 1** binary segmentation masks into YOLO-compliant bounding box labels.
*   **Class Imbalance Mitigation:** Utilized a `WeightedRandomSampler` and class-weighted Cross-Entropy loss to address the high frequency of Melanocytic Nevus samples in the HAM10000 dataset.
*   **Real-Time Deployment:** Optimized the pipeline for CPU-based inference with a latency of **~59.0 ms**, making it suitable for point-of-care screening.

---

## 📊 Experimental Results
The framework was evaluated on the **ISIC 2018 (Task 1,3)** benchmark.

| Component | Metric | Value |
| :--- | :--- | :--- |
| **Localization (YOLOv8)** | mAP @ 0.5 | **98.6%** |
| **Localization (YOLOv8)** | Precision | **96.3%** |
| **Diagnostic (ResNet18)** | Validation Accuracy | **91.0%** |
| **Inference Latency** | Time per image (CPU) | **~59.0 ms** |

---

## 🖥️ Streamlit Web Application
We developed an interactive dashboard to bridge the gap between research and clinical utility.
*   **Upload & Analyze:** Drag and drop dermoscopic images for instant analysis.
*   **Threshold Tuning:** Interactively adjust the confidence threshold (0.05 to 0.90) to observe model sensitivity.
*   **Visual Summary:** Real-time display of bounding boxes and class probabilities.

### How to run:
1. Clone the repo:
   ```bash
   git clone https://github.com/hadeeralkady/Skin-Lesion-Detection-using-YOLOv8.git
   cd Skin-Lesion-Detection-using-YOLOv8

## 🧪 Challenges & Observations

* Model performance was sensitive to confidence threshold  
* Best predictions observed at **0.2 – 0.25**  
* Medical images require careful threshold calibration  
* Bounding box localization is harder than classification  
* Classification accuracy depends heavily on data quality and class balance

---

## 🧑‍💻 Technologies Used

* Python 3.10+  
* PyTorch  
* Ultralytics YOLOv8  
* OpenCV  
* Streamlit  
* Matplotlib  

---
### 1- Install dependencies:
   ```bash
   pip install -r requirements.txt

### 2- Launch the app:
   ```bash
   streamlit run streamlit_app.py

## 🧑‍💻 Authors
Hadeer Elkady - hadeeralkady
Hagar Galal - hagarsliem

## 🤝 Acknowledgments

* ISIC Archive  
* Ultralytics YOLO Team  
* Open-source medical AI community  

---

## 📜 License

This repository contains **code only**.  
Dataset license belongs to **ISIC Archive** and must be respected separately.

---

## ⭐ Final Note

This project demonstrates **responsible AI development in healthcare**, focusing on transparency, reproducibility, and ethical considerations.
