# Skin Lesion Detection & Classification (YOLOv8 + ResNet18)

---

## 📌 Project Overview

This project presents an **end-to-end deep learning pipeline** for detecting **skin lesion regions** in dermoscopic images using **YOLOv8**, and classifying the detected lesions into **3 classes (MEL, NV, BKL)** using **ResNet18**.

The primary objective is **localizing suspicious lesion areas** and **classifying them**, not diagnosing diseases.

The project demonstrates:

* Medical image preprocessing
* YOLOv8 fine-tuning (detection)
* ResNet18 training (classification)
* Model evaluation
* Inference via **Streamlit Web App**
* Ethical AI practices in medical imaging

---

## ⚠️ Medical Disclaimer

> This project is intended **strictly for research and educational purposes**.  
> It **must not** be used for medical diagnosis, treatment, or clinical decision-making.

---

## 🧠 Model Architecture

### 1) Detection Model (YOLOv8)
* **Base Model:** YOLOv8 (Ultralytics)
* **Task:** Object Detection
* **Input Size:** 640×640
* **Output:** Bounding boxes around lesion regions

YOLOv8 was chosen for its:

* Real-time performance
* Strong localization accuracy
* Lightweight deployment suitability

### 2) Classification Model (ResNet18)
* **Base Model:** ResNet18 (PyTorch)
* **Task:** Multi-class classification
* **Input Size:** 224×224
* **Output:** 3 classes: MEL, NV, BKL

ResNet18 was chosen for its:

* High accuracy with limited data
* Fast training and inference
* Lightweight deployment suitability

---

## 📂 Dataset

### Dataset Source

This project uses dermoscopic images from the:

**ISIC (ISIC2018_Task1-2_Training)– International Skin Imaging Collaboration Archive**  
🔗 https://www.isic-archive.com

### Dataset Usage

Due to **licensing, privacy, and ethical constraints**, the dataset is **not included** in this repository.

Researchers wishing to reproduce the results must:

1. Download images from the official ISIC website
2. Convert annotations to YOLO format (for detection)
3. Prepare classification labels (MEL, NV, BKL)
4. Organize data according to the structure below

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

## Training Details (Detection)

| Parameter            | Value                                          |
| -------------------- | ---------------------------------------------- |
| Epochs               | 50 (fine-tuned from pretrained YOLOv8 weights) |
| Images               | 1816 training images                           |
| Image Size           | 640                                            |
| Optimizer            | Default YOLOv8 (AdamW – auto-selected)         |
| Confidence Threshold | 0.2 – 0.25                                     |

* Training was performed using CPU  
* Model convergence was achieved within 50 epochs  
* Small-object sensitivity was improved by dataset preprocessing and bounding box adjustments  
* Final model selected based on best validation mAP  

### Training Command Example (YOLOv8)

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

## Training Details (Classification)

| Parameter            | Value                         |
| -------------------- | ----------------------------- |
| Model                | ResNet18                      |
| Classes              | 3 (MEL, NV, BKL)              |
| Input Size           | 224×224                       |
| Batch Size           | 16–32 (depending on GPU)      |
| Optimizer            | Adam / SGD                    |
| Learning Rate        | 1e-3 (typical starting value) |
| Epochs               | 20–30                         |
  
* Final model saved as `resnet18_model.pth`

---

## 📊 Evaluation

### Detection Evaluation
Validation was performed using YOLO built-in evaluation tools.

Generated outputs include:

* Precision
* Recall
* mAP@0.5
* Prediction visualizations

Confidence threshold tuning was required due to:

* Small lesion sizes
* Low contrast medical images

### Classification Evaluation
Classification performance was evaluated using:

* Accuracy
* Precision / Recall
* Confusion Matrix

---

## 🖥️ Streamlit Web Application

An interactive **Streamlit UI** is provided for inference:

### Features

* Single image prediction
* YOLO detection (bounding box)
* ResNet18 classification (class + confidence)
* Adjustable confidence threshold

---

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
