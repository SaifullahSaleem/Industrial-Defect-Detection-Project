# app.py
"""
Visual Defect Detection System

A simple Streamlit web app that loads a pre‑trained YOLO11n classification model
and predicts whether an uploaded casting product image is *Defective* or *Normal*.
"""

import time
import logging
from pathlib import Path

import streamlit as st
import torch
from torchvision import transforms
from PIL import Image

# ---------------------------------------------------------------
# Logging configuration (console output)
# ---------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------
# Model path & class mapping
# ---------------------------------------------------------------
MODEL_PATH = Path(__file__).parent / "complete_defect_classifier.pth"
CLASS_MAP = {0: "Defective", 1: "Normal"}

# ---------------------------------------------------------------
# Load the model once (cached per session)
# ---------------------------------------------------------------
@st.cache_resource
def load_model():
    """Load the complete PyTorch model and select the device."""
    logger.info("Loading model...")
    if not MODEL_PATH.is_file():
        logger.error(f"Model file not found at {MODEL_PATH}")
        st.error("Model file not found. Please ensure 'complete_defect_classifier.pth' is present.")
        st.stop()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device selected: {device}")

    try:
        model = torch.load(MODEL_PATH, map_location=device, weights_only=False)
        model.to(device)
        model.eval()
        logger.info(f"Model loaded successfully on {device}")
    except Exception as e:
        logger.exception("Failed to load model")
        st.error(f"Error loading model: {e}")
        st.stop()

    return model, device

# ---------------------------------------------------------------
# Pre‑processing (must match training)
# ---------------------------------------------------------------
_preprocess = transforms.Compose([
    transforms.Resize((300, 300)),
    transforms.ToTensor(),
])

def predict_image(pil_image: Image.Image, model: torch.nn.Module, device: torch.device):
    """Run inference on a single image.

    Returns:
        label (str), confidence (float), latency_ms (float)
    """
    # Convert to RGB and apply transforms
    image = pil_image.convert("RGB")
    tensor = _preprocess(image).unsqueeze(0).to(device)  # Shape: [1, C, H, W]

    start = time.time()
    with torch.no_grad():
        outputs = model(tensor)
        logits = outputs[1] if isinstance(outputs, tuple) else outputs
        probs = torch.softmax(logits, dim=1)
        pred_idx = torch.argmax(probs, dim=1).item()
        confidence = probs[0, pred_idx].item()
    latency = (time.time() - start) * 1000  # ms

    label = CLASS_MAP.get(pred_idx, "Unknown")
    logger.info(f"Prediction={label} | Confidence={confidence:.4f} | Latency={latency:.2f}ms")
    return label, confidence, latency

# ---------------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------------
st.set_page_config(page_title="Visual Defect Detection System", page_icon=":mag:")

st.title("Visual Defect Detection System")
st.write(
    "Upload an image of a manufactured casting product to determine whether it is **Defective** or **Normal**."
)

uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    try:
        img = Image.open(uploaded_file)
        st.image(img, caption="Uploaded Image", use_column_width=True)
    except Exception:
        st.error("The uploaded file could not be read as an image.")
        st.stop()

    if st.button("Analyze Image"):
        model, device = load_model()
        try:
            label, confidence, latency = predict_image(img, model, device)
        except Exception as e:
            logger.exception("Inference error")
            st.error(f"Error during inference: {e}")
            st.stop()

        st.subheader("Result")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Prediction", label)
        with col2:
            st.metric("Confidence", f"{confidence * 100:.2f}%")
        st.metric("Inference Time (ms)", f"{latency:.2f}")
        st.metric("Device", str(device))

        if label == "Normal":
            st.success("The product appears normal.")
        else:
            st.error("Defect detected! Please review the product.")
else:
    st.info("Awaiting image upload.")
