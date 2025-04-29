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
@st.cache_resource
def load_model():
    model = DenseNet(growth_rate=12, block_config=(6, 12, 24, 16))
    model.load_state_dict(
        torch.load("/Users/debabratapanda/PycharmProjects/Capstone_project/model_epoch_8.pth", weights_only=True)   # ⬅️ your fresh file
    )
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

    # -- predictions --
    with torch.no_grad():
        # --------------------  Forward pass  --------------------
        fracture_logits, chest_logits = model(img_tensor)

        # ====================== FRACTURE =========================
        frac_probs = torch.softmax(fracture_logits, dim=1)[0]  # [p_no, p_yes]
        frac_idx = int(frac_probs.argmax())  # 0 / 1
        frac_conf = frac_probs[frac_idx].item()  # 0–1
        frac_label = "Fracture" if frac_idx == 1 else "Not Fracture"

        # ====================== CHEST  ===========================
        chest_probs = torch.softmax(chest_logits, dim=1)[0].cpu().numpy()

        # ---- Top-3 (high→low) ----------------------------------
        top3_idx = chest_probs.argsort()[-3:][::-1]
        idx2label = {v: k for k, v in chest_condition_map.items()}
        top3 = [(idx2label[i], chest_probs[i] * 100) for i in top3_idx]

        # ---- Best class & confidence ----------------------------
        chest_label, chest_conf = top3[0]  # chest_conf already in %

    # --------------------- Display -------------------------------
    st.markdown("## 🔍 Prediction Results")

    if chest_conf > 90:  # fracture dominates
        st.write(f"**🦴 Fracture Detection:** `{frac_label}` "
                 f"({frac_conf:.2%} confidence)")

    else:  # show chest details
        st.write(f"**🫁 Chest Condition (Top-1):** "
                 f"`{chest_label}`  ({chest_conf:.2f}% confidence)")
        # Top-3 list
        st.write("**Top-3 Chest Probabilities:**")
        for lbl, conf in top3:
            st.write(f"- `{lbl}` — {conf:.2f}%")
