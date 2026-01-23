import streamlit as st
from ultralytics import YOLO
from PIL import Image
import torch
import torchvision
import torch.nn as nn
import torchvision.transforms as transforms
import pandas as pd
import altair as alt
import time

import base64
from io import BytesIO

def image_to_base64(img):
    buffered = BytesIO()
    img.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()

# ===========================
# Page Config
# ===========================
st.set_page_config(
    page_title="Skin Lesion Detection | Research Demo",
    page_icon="🧠",
    layout="wide"
)

# ===========================
# Session State for Splash
# ===========================
if "splash_done" not in st.session_state:
    st.session_state.splash_done = False

# ===========================
# CSS Styling (Auto Theme)
# ===========================
st.markdown(
    """
    <style>
    /* ----------- Auto Theme using prefers-color-scheme ----------- */
    :root {
        --bg: #ffffff;
        --text: #0f172a;
        --card: #f8fafc;
        --border: #e2e8f0;
        --primary: #2563eb;
        --secondary: #10b981;
        --muted: #64748b;
    }

    @media (prefers-color-scheme: dark) {
        :root {
            --bg: #0b1020;
            --text: #e5e7eb;
            --card: #0f172a;
            --border: #1f2937;
            --primary: #2563eb;
            --secondary: #10b981;
            --muted: #94a3b8;
        }
    }

    .stApp {
        background: var(--bg);
        color: var(--text);
    }

    .card {
        background: var(--card);
        border-radius: 16px;
        padding: 18px;
        box-shadow: 0 8px 20px rgba(0,0,0,0.12);
        border: 1px solid var(--border);
    }

    .section-title {
        color: var(--primary);
        font-weight: 700;
    }

    .subtitle {
        color: var(--muted);
        font-size: 16px;
    }

    .footer {
        text-align: center;
        color: var(--muted);
        margin-top: 20px;
    }

    .stButton>button {
        background: var(--primary);
        color: white;
        border-radius: 12px;
        padding: 10px 14px;
        font-weight: 700;
    }
    .stButton>button:hover {
        background: #1d4ed8;
    }

    /* Drag & Drop Style */
    .stFileUploader>div>div {
        border: 2px dashed var(--primary);
        border-radius: 12px;
        padding: 14px;
        background: rgba(37, 99, 235, 0.05);
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ===========================
# Sidebar Navigation
# ===========================
st.sidebar.markdown("## 🧭 Navigation")

page = st.sidebar.radio(
    "Go to",
    ["Home", "Detection", "Metrics", "About"]
)

st.sidebar.divider()

conf_threshold = st.sidebar.slider(
    "Confidence Threshold",
    0.05, 0.90, 0.25, 0.05
)

st.sidebar.markdown("""
Lower → more detections  
Higher → fewer, more confident detections
""")

st.sidebar.divider()
st.sidebar.markdown("⚠️ **Research Use Only**  \nNot intended for clinical diagnosis.")

# ===========================
# Load YOLO Model
# ===========================
@st.cache_resource
def load_yolo_model():
    return YOLO("Models/best.pt")

model = load_yolo_model()

# ===========================
# Load Classifier
# ===========================
@st.cache_resource
def load_classifier():
    clf = torchvision.models.resnet18(pretrained=False)
    clf.fc = nn.Linear(512, 3)
    clf.load_state_dict(torch.load(r"Models/resnet18_model.pth", map_location="cpu"))
    clf.eval()
    return clf

classifier = load_classifier()

classifier_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

def classify_crop(crop_img):
    img_t = classifier_transform(crop_img).unsqueeze(0)
    with torch.no_grad():
        out = classifier(img_t)
        probs = torch.softmax(out, dim=1)
        conf, pred = torch.max(probs, dim=1)
    return pred.item(), conf.item()

class_names = ["MEL", "NV", "BKL"]

# Splash Screen
if not st.session_state.splash_done:
    st.markdown("<h1>🧠 Skin Lesion Detection</h1>", unsafe_allow_html=True)
    st.markdown("<p class='subtitle'>Loading the Research Demo... Please wait</p>", unsafe_allow_html=True)
    st.markdown("<div class='card'><p style='margin:0;'>✨ Welcome! The app will open in 3 seconds...</p></div>", unsafe_allow_html=True)
    time.sleep(3)
    st.session_state.splash_done = True
    st.stop()

# Home Page
if page == "Home":
    st.markdown("<h1>🧠 Skin Lesion Detection</h1>", unsafe_allow_html=True)
    st.markdown("<p class='subtitle'>YOLOv8 + ResNet18 • Research Demo</p>", unsafe_allow_html=True)
    st.divider()

    st.markdown("""
    <div class="card">
        <h3 class="section-title">What this app does</h3>
        <ul>
            <li>Detects lesion regions using YOLOv8</li>
            <li>Classifies lesion type (MEL / NV / BKL) using ResNet18</li>
            <li>Displays results in a clean dashboard</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

# ===========================
# Detection Page
# ===========================
elif page == "Detection":
    st.markdown("<h1>🧪 Detection</h1>", unsafe_allow_html=True)
    st.markdown("<p class='subtitle'>Upload a dermoscopic image and run detection</p>", unsafe_allow_html=True)
    st.divider()

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

    with right:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("### 🧪 Detection Output")

        if "last_result" in st.session_state:
            result = st.session_state.last_result
            image = st.session_state.last_image

            st.image(result.plot(), use_container_width=True)

            if result.boxes and len(result.boxes) > 0:
                st.markdown("### 📊 Detection Summary")

                for i, box in enumerate(result.boxes):
                    conf = float(box.conf[0])
                    x1, y1, x2, y2 = map(int, box.xyxy[0])

                    crop_img = image.crop((x1, y1, x2, y2))
                    cls_idx, cls_conf = classify_crop(crop_img)

                    st.markdown(
                        f"<div class='card'>Lesion {i+1}<br>"
                        f"Detection Confidence: {conf:.2f}<br>"
                        f"Class: {class_names[cls_idx]} (Conf: {cls_conf:.2f})</div>",
                        unsafe_allow_html=True
                    )
            else:
                st.warning("No lesion detected at the selected confidence level.")
        else:
            st.info("Upload an image and click Run Detection.")

        st.markdown("</div>", unsafe_allow_html=True)

# ===========================
# Metrics Page
# ===========================
elif page == "Metrics":
    st.markdown("<h1>📊 Metrics Dashboard</h1>", unsafe_allow_html=True)
    st.markdown("<p class='subtitle'>Model performance visualized</p>", unsafe_allow_html=True)
    st.divider()

    metrics_data = {
        "Metric": ["Precision", "Recall", "mAP@0.5", "mAP@0.5-0.95"],
        "Value": [96.3, 94.9, 98.6, 75.4]
    }
    df_metrics = pd.DataFrame(metrics_data)

    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("### 📈 Performance Chart")
    chart = alt.Chart(df_metrics).mark_bar().encode(
        x='Metric',
        y='Value',
        tooltip=['Metric', 'Value']
    ).properties(width=800, height=400)
    st.altair_chart(chart, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

# ===========================
# About Page
# ===========================
elif page == "About":
    st.markdown("<h1>ℹ️ About</h1>", unsafe_allow_html=True)
    st.markdown("<p class='subtitle'>Research demo built for skin lesion detection</p>", unsafe_allow_html=True)
    st.divider()

    st.markdown("""
    <div class='card'>
        <h3 class='section-title'>Models Used</h3>
        <ul>
            <li><b>YOLOv8</b> for lesion localization</li>
            <li><b>ResNet18</b> for classification (MEL / NV / BKL)</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class='card'>
        <h3 class='section-title'>Disclaimer</h3>
        <p>This application is for <b>research and educational purposes only</b>.</p>
        <p>It should not be used for clinical diagnosis.</p>
    </div>
    """, unsafe_allow_html=True)

st.divider()
st.markdown("<p class='footer'>For academic and research use only</p>", unsafe_allow_html=True)
