import streamlit as st 
from ultralytics import YOLO
from PIL import Image
import os

# =====================================================
# Page Config
# =====================================================
st.set_page_config(
    page_title="Skin Lesion Detection | Research Demo",
    page_icon="🧠",
    layout="wide"
)

# =====================================================
# Session State – Show About the Model only once
# =====================================================
if "show_intro" not in st.session_state:
    st.session_state.show_intro = True

# =====================================================
# Global CSS – Dark Medical Theme
# =====================================================
st.markdown("""
<style>
html, body {
    background-color: #0e1117;
    color: #e5e7eb;
}

.block-container {
    padding-top: 2rem;
}

.stSidebar {
    background-color: #0e1117;
    border-right: 1px solid #1f2937;
}

h1 {
    font-size: 40px;
    text-align: center;
    color: #e5e7eb;
}

.subtitle {
    font-size: 16px;
    color: #9ca3af;
    text-align: center;
    margin-bottom: 20px;
}

.section-title {
    color: #93c5fd;
    margin-bottom: 10px;
}

.card {
    background-color: #111827;
    border: 1px solid #1f2937;
    border-radius: 8px;
    padding: 12px;
    margin-bottom: 10px;
}

/* زرار Run Detection مميز */
.stButton > button[key="run-left"] {
    background-color: #10b981 !important;  /* أخضر مميز */
    color: white !important;
    border-radius: 10px;
    height: 45px;
    border: none;
}
.stButton > button[key="run-left"]:hover {
    background-color: #059669 !important; /* أخضر أغمق عند المرور */
}

.stButton > button {
    background-color: #2563eb;
    color: white;
    border-radius: 10px;
    height: 45px;
    border: none;
}

.stButton > button:hover {
    background-color: #1d4ed8;
}

.footer {
    text-align: center;
    color: #6b7280;
    font-size: 13px;
}
</style>
""", unsafe_allow_html=True)

# =====================================================
# Sidebar – Navigation & Settings
# =====================================================
st.sidebar.markdown("## 🧭 Navigation")

page = st.sidebar.radio(
    "Go to",
    ["Detection", "Metrics", "Example Gallery"]
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
# Load Model
# =====================================================
@st.cache_resource
def load_model():
    try:
        model = YOLO("models/best.pt")
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# Load model once
model = load_model()

# Stop app execution if model failed to load
if model is None:
    st.stop()

# =====================================================
# Header
# =====================================================
st.markdown("<h1>🧠 Skin Lesion Detection</h1>", unsafe_allow_html=True)
st.markdown(
    "<p class='subtitle'>YOLOv8 Fine-Tuned • Research Demonstration</p>",
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
    fine-tuned to localize suspected **skin lesion regions** in dermoscopic images.

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

            # زرار Run Detection تحت الصورة + لون مميز
            if st.button("Run Detection", key="run-left", use_container_width=True):
                with st.spinner("Running YOLO inference..."):
                    results = model(image, conf=conf_threshold)
                result = results[0]
                st.session_state.last_result = result

        st.markdown("</div>", unsafe_allow_html=True)

    # -------- Output Section --------
    with right:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("### 🧪 Detection Output")

        if "last_result" in st.session_state:
            result = st.session_state.last_result
            st.image(result.plot(), use_container_width=True)

            if result.boxes and len(result.boxes) > 0:
                st.markdown("### 📊 Detection Summary")
                for i, box in enumerate(result.boxes):
                    conf = float(box.conf[0])
                    st.markdown(
                        f"<div class='card'>Lesion {i+1}<br>Confidence: {conf:.2f}</div>",
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

    # Interpretation Section مع تمييز العنوان والشرح
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
# EXAMPLE GALLERY PAGE
# =====================================================
elif page == "Example Gallery":

    st.markdown("## 🖼️ Example Gallery")

    example_dir = "examples"

    if not os.path.exists(example_dir):
        st.warning(
            "Example gallery folder not found. "
            "Please create an 'examples' folder and add sample images."
        )
    else:
        images = [
            img for img in os.listdir(example_dir)
            if img.lower().endswith(("jpg", "jpeg", "png"))
        ]

        if len(images) == 0:
            st.info("No example images available.")
        else:
            cols = st.columns(3)
            for idx, img_name in enumerate(images):
                with cols[idx % 3]:
                    img_path = os.path.join(example_dir, img_name)
                    img = Image.open(img_path)
                    st.image(img, caption=img_name, use_container_width=True)

# =====================================================
# Footer
# =====================================================
st.divider()
st.markdown(
    "<p class='footer'>For academic and research use only</p>",
    unsafe_allow_html=True
)