import streamlit as st 
from ultralytics import YOLO
from PIL import Image
import os

import torch
import torchvision
import torch.nn as nn
import torchvision.transforms as transforms

# Page Config
st.set_page_config(
    page_title="Skin Lesion Detection | Research Demo",
    page_icon="🧠",
    layout="wide"
)

# Session State – Show About the Model only once
if "show_intro" not in st.session_state:
    st.session_state.show_intro = True

# =====================================================
# Sidebar – Navigation & Settings
# =====================================================
st.sidebar.markdown("## 🧭 Navigation")

page = st.sidebar.radio(
    "Go to",
    ["Detection", "Metrics"]
)

st.sidebar.divider()

st.sidebar.markdown("### ⚙️ Inference Settings")

conf_threshold = st.sidebar.slider(
    "Confidence Threshold",
    0.05, 0.90, 0.25, 0.05
)

st.sidebar.markdown("""
Lower → more detections  
Higher → fewer, more confident detections
""")

st.sidebar.divider()

st.sidebar.markdown("""⚠️ **Research Use Only**  
Not intended for clinical diagnosis.
""")

# =====================================================
# Load YOLO Model
# =====================================================
@st.cache_resource
def load_yolo_model():
    try:
        model = YOLO("Models/best.pt")
        return model
    except Exception as e:
        st.error(f"Error loading YOLO model: {e}")
        return None

model = load_yolo_model()

if model is None:
    st.stop()

# =====================================================
# ====== NEW: Load ResNet18 Classifier ======
# =====================================================
@st.cache_resource
def load_classifier():
    clf = torchvision.models.resnet18(pretrained=False)
    clf.fc = nn.Linear(512, 3)  # 3 classes: MEL, NV, BKL

    clf.load_state_dict(torch.load(
        "Models/resnet18_model.pth",
        map_location="cpu"
    ))
    clf.eval()
    return clf

classifier = load_classifier()

# =====================================================
# ====== NEW: Transform for classifier ======
# =====================================================
classifier_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

def classify_image(img):
    img_t = classifier_transform(img).unsqueeze(0)
    with torch.no_grad():
        out = classifier(img_t)
        probs = torch.softmax(out, dim=1)
        conf, pred = torch.max(probs, dim=1)
    return pred.item(), conf.item()

# =====================================================
# Header
# =====================================================
st.markdown("<h1>🧠 Skin Lesion Detection</h1>", unsafe_allow_html=True)
st.markdown(
    "<p class='subtitle'>YOLOv8 Fine-Tuned + ResNet18 Classification • Research Demo</p>",
    unsafe_allow_html=True
)
st.divider()

# =====================================================
# About the Model (ONLY SHOW ONCE)
# =====================================================
if st.session_state.show_intro:
    st.markdown("""
    <div class="card">
    <h3 class="section-title">📌 About the Model</h3>

    This application demonstrates a **YOLOv8 object detection model**
    fine-tuned to localize suspected **skin lesion regions**, then a **ResNet18 classifier**
    to predict the lesion type (MEL / NV / BKL).

    ⚠️ **Disclaimer**  
    This system is intended **strictly for research and educational purposes**  
    and **must not** be used for medical diagnosis or clinical decision-making.
    </div>
    """, unsafe_allow_html=True)

    st.session_state.show_intro = False

# =====================================================
# DETECTION PAGE
# =====================================================
if page == "Detection":

    left, right = st.columns([1, 1])

    # -------- Upload + Detection --------
    with left:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("### 📤 Upload Image")
        uploaded_file = st.file_uploader(
            "Dermoscopic image (JPG / PNG)",
            type=["jpg", "jpeg", "png"]
        )

        if uploaded_file:
            image = Image.open(uploaded_file).convert("RGB")
            st.image(image, use_container_width=True)

            if st.button("Run Detection", key="run-left", use_container_width=True):
                with st.spinner("Running YOLO inference..."):
                    results = model(image, conf=conf_threshold)
                result = results[0]
                st.session_state.last_result = result
                st.session_state.last_image = image  # save image for classification

        st.markdown("</div>", unsafe_allow_html=True)

    # -------- Output Section --------
    with right:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("### 🧪 Detection Output")

        if "last_result" in st.session_state:
            result = st.session_state.last_result
            image = st.session_state.last_image

            st.image(result.plot(), use_container_width=True)

            if result.boxes and len(result.boxes) > 0:
                st.markdown("### 📊 Detection Summary")

                # ====== NEW: Classification on full image ======
                class_names = ["MEL", "NV", "BKL"]
                cls_idx, cls_conf = classify_image(image)

                for i, box in enumerate(result.boxes):
                    conf = float(box.conf[0])
                    st.markdown(
                        f"<div class='card'>Lesion {i+1}<br>"
                        f"Detection Confidence: {conf:.2f}<br>"
                        f"Class: {class_names[cls_idx]} (Conf: {cls_conf:.2f})</div>",
                        unsafe_allow_html=True
                    )
            else:
                st.warning("No lesion detected at the selected confidence level.")

        st.markdown("</div>", unsafe_allow_html=True)

# =====================================================
# METRICS PAGE
# =====================================================
elif page == "Metrics":

    st.markdown("## 📊 Model Performance Metrics")

    col1, col2, col3, col4 = st.columns(4)

    metrics = {
        "Precision": "96.3%",
        "Recall": "94.9%",
        "mAP@0.5": "98.6%",
        "mAP@0.5–0.95": "75.4%"
    }

    for col, (k, v) in zip([col1, col2, col3, col4], metrics.items()):
        with col:
            st.markdown(
                f"<div class='card'><h3>{k}</h3><h2>{v}</h2></div>",
                unsafe_allow_html=True
            )

    st.markdown("""
    <div class="card">
    <h3 class="section-title">🧠 Interpretation</h3>
    <p><span style="color:#93c5fd; font-weight:bold;">High precision</span> → <span style="color:#e5e7eb;">low false positives</span></p>
    <p><span style="color:#93c5fd; font-weight:bold;">Strong recall</span> → <span style="color:#e5e7eb;">good lesion coverage</span></p>
    <p><span style="color:#93c5fd; font-weight:bold;">High mAP@0.5</span> → <span style="color:#e5e7eb;">accurate localization</span></p>
    <p><span style="color:#93c5fd; font-weight:bold;">mAP@0.5–0.95</span> → <span style="color:#e5e7eb;">reflects strict IoU evaluation</span></p>
    </div>
    """, unsafe_allow_html=True)

# =====================================================
# Footer
# =====================================================
st.divider()
st.markdown(
    "<p class='footer'>For academic and research use only</p>",
    unsafe_allow_html=True
)