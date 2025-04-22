import streamlit as st
import torch
import numpy as np
import cv2
from PIL import Image

# Load your model
from mod import DenseNet  # or however you import your model
from func import load_and_preprocess_image,generate_gradcam

# Load label mapping
import json

# --- Step 1: Load the saved chest condition map ---
# 1. this will help us decode model output class indices into readable labels
# 2. must match the mapping used during training
with open("chest_condition_map.json", "r") as f:
    chest_condition_map = json.load(f)

# --- Step 2: Load the trained model ---
# 1. we cache the model so it doesn't reload on every rerun
# 2. uses the exact architecture and loads trained weights from .pth file
@st.cache_resource
def load_model():
    model = DenseNet(growth_rate=12, block_config=(6, 12, 24, 16))  # or your custom config
    model.load_state_dict(torch.load(
        "/Users/debabratapanda/PycharmProjects/Capstone_project/model_epoch_20.pth",
        map_location="mps"
    ))
    model.eval()
    return model

model = load_model()

# Your existing preprocessing function
import tempfile

# --- Step 3: Preprocess Uploaded Image ---
# 1. save uploaded file temporarily
# 2. run the real preprocessing pipeline (CLAHE, normalization, resizing)
# 3. return a PyTorch tensor of shape [1, 1, H, W]
def preprocess_uploaded_image(uploaded_file, target_size=(224, 224)):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_path = tmp_file.name

    final_image = load_and_preprocess_image(tmp_path, target_size=target_size, save_flag=False)

    if final_image is None:
        raise ValueError("Preprocessing failed!")

    final_image = final_image[np.newaxis, np.newaxis, :, :]  # [1, 1, H, W]
    return torch.tensor(final_image, dtype=torch.float32)

# --- Streamlit UI ---
st.title("🩻 X-ray Diagnosis Assistant")

# File uploader component
uploaded_file = st.file_uploader("Upload a Chest X-ray or Bone Scan", type=["png", "jpg", "jpeg"])

# Setup device for Apple Silicon (MPS) or CPU fallback
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
model.to(device)
st.write(f"device is:{device}")

# -- Upload and Show Image --
if uploaded_file is not None:
    st.image(uploaded_file, caption="📤 Uploaded Image", use_container_width=True)

    # -- Preprocess and move to device --
    img_tensor = preprocess_uploaded_image(uploaded_file).to(device)

    # ✅ Detach and enable gradient tracking for Grad-CAM
    img_tensor = img_tensor.detach().clone().requires_grad_(True)

    # -- Grad-CAM: run before model to register hooks
    heatmap_fracture = generate_gradcam(model, img_tensor, head="fracture",
                                        target_layer_name="blocks.3.layers.8.conv2", device=device)
    heatmap_chest = generate_gradcam(model, img_tensor, head="chest", target_layer_name="blocks.3.layers.8.conv2",
                                     device=device)

    # -- If needed, you can re-run for predictions (with no_grad for safety) --
    with torch.no_grad():
        fracture_pred, chest_pred = model(img_tensor)

        # Fracture logic
        fracture_prob = torch.sigmoid(fracture_pred).item()
        fracture_label = "Fracture" if fracture_prob > 0.6 else "Not Fracture"

        # Chest logic
        chest_probs = torch.softmax(chest_pred, dim=1)[0].cpu().numpy()
        chest_idx = np.argmax(chest_probs)
        chest_confidence = chest_probs[chest_idx] * 100
        chest_label = [k for k, v in chest_condition_map.items() if v == chest_idx][0]

    # -- Display Results --
    st.markdown("## 🔍 Prediction Results")
    if chest_confidence > 90:
        st.write(f"**🦴 Fracture Detection:** `{fracture_label}` ({fracture_prob:.2%} confidence)")
        st.markdown("## 🎯 Grad-CAM Visualizations")
        st.image(heatmap_fracture, caption="🦴 Fracture Detection Grad-CAM", use_container_width=True)
    if chest_confidence < 90:
        st.write(f"**🫁 Chest Condition:** `{chest_label}` ({chest_confidence:.2f}% confidence)")
        st.markdown("## 🎯 Grad-CAM Visualizations")
        st.image(heatmap_chest, caption="🫁 Chest Condition Grad-CAM", use_container_width=True)

