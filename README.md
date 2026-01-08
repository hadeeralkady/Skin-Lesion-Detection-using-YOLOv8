## 📌 Project Overview

This project presents an **end-to-end deep learning pipeline** for detecting **skin lesion regions** in dermoscopic images using **YOLOv8**.
The primary objective is **localizing suspicious lesion areas**, not diagnosing diseases.

The project demonstrates:

* Medical image preprocessing
* YOLOv8 fine-tuning
* Model evaluation
* Inference via **Streamlit Web App**
* Ethical AI practices in medical imaging

---

## ⚠️ Medical Disclaimer

> This project is intended **strictly for research and educational purposes**.
> It **must not** be used for medical diagnosis, treatment, or clinical decision-making.

---

## 🧠 Model Architecture

* **Base Model:** YOLOv8 (Ultralytics)
* **Task:** Object Detection
* **Input Size:** 640×640
* **Output:** Bounding boxes around lesion regions

YOLOv8 was chosen for its:

* Real-time performance
* Strong localization accuracy
* Lightweight deployment suitability

---

## 📂 Dataset

### Dataset Source

This project uses dermoscopic images from the:

**ISIC (ISIC2018_Task1-2_Training)– International Skin Imaging Collaboration Archive**
🔗 [https://www.isic-archive.com](https://www.isic-archive.com)

### Dataset Usage

Due to **licensing, privacy, and ethical constraints**, the dataset is **not included** in this repository.

Researchers wishing to reproduce the results must:

1. Download images from the official ISIC website
2. Convert annotations to YOLO format
3. Organize data according to the structure below

```
YOLO_dataset/
├── images/
│   ├── train/
│   ├── val/
│   └── test/
├── labels/
│   ├── train/
│   ├── val/
│   └── test/
└── dataset.yaml
```

---

## Training Details

| Parameter            | Value                                          |
| -------------------- | ---------------------------------------------- |
| Epochs               | 50 (fine-tuned from pretrained YOLOv8 weights) |
| Images               | 1816 training images                           |
| Image Size           | 640                                            |
| Optimizer            | Default YOLOv8 (AdamW – auto-selected)         |
| Confidence Threshold | 0.2 – 0.25                                     |

- Training was performed using CPU
- Model convergence was achieved within 50 epochs
- Small-object sensitivity was improved by dataset preprocessing and bounding box adjustments
- Final model selected based on best validation mAP

### Training Command Example

```bash
yolo detect train \
model=yolov8n.pt \
data=dataset.yaml \
epochs=120 \
imgsz=640
```

To **continue training from an existing model**:

```bash
yolo detect train \
model=path/to/best.pt \
data=dataset.yaml \
epochs=50
```

---

## 📊 Evaluation

Validation was performed using YOLO built-in evaluation tools.

Generated outputs include:

* Precision
* Recall
* mAP@0.5
* Prediction visualizations

Confidence threshold tuning was required due to:

* Small lesion sizes
* Low contrast medical images

---

## 🖥️ Streamlit Web Application

An interactive **Streamlit UI** is provided for inference:

### Features

* Single image prediction
* Folder-based batch inference
* Adjustable confidence threshold

---

## 🧪 Challenges & Observations

* Model performance was sensitive to confidence threshold
* Best predictions observed at **0.2 – 0.25**
* Medical images require careful threshold calibration
* Bounding box localization is harder than classification

---

## 🚀 Future Work

* Increase training epochs with data augmentation
* Experiment with YOLOv8m / YOLOv8l
* Add segmentation-based models
* Integrate Grad-CAM visualizations
* Improve small lesion detection

---

## 🧑‍💻 Technologies Used

* Python 3.10+
* PyTorch
* Ultralytics YOLOv8
* OpenCV
* Streamlit
* Matplotlib

---

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
