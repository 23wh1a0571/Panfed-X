import streamlit as st
import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import zoom
import matplotlib.pyplot as plt
import os
import tempfile
from core.classify import DenseNet121_3D

# --- Page Configuration ---
st.set_page_config(
    page_title="PanFed-X | Pancreatic Diagnostic Suite",
    page_icon="🩺",
    layout="wide"
)

st.title("🩺 PanFed-X: AI-Powered Pancreatic Cancer Risk Stratification")
st.markdown("#### Decentralized Multi-Center MRI Diagnostic Support System (3D DenseNet-121 + FedAvg)")
st.markdown("---")

# --- Sidebar Configuration ---
st.sidebar.header("Diagnostic Settings")
model_choice = st.sidebar.selectbox(
    "Select Model Weights",
    ("Federated Global Model (FedAvg)", "Centralized Baseline (DenseNet-121)")
)

if model_choice == "Federated Global Model (FedAvg)" and os.path.exists("models/fedavg_global_densenet121.pth"):
    selected_weight_path = "models/fedavg_global_densenet121.pth"
else:
    selected_weight_path = "models/best_densenet121.pth"

st.sidebar.info(f"**Loaded Checkpoint:**\n`{selected_weight_path}`")
st.sidebar.markdown("""
**Pathology Stratification Tiers:**
- **Tier 0:** No Risk (Benign / Healthy Pancreas)
- **Tier 1:** Low Risk (Low-Grade Dysplasia IPMN)
- **Tier 2:** High Risk (High-Grade Dysplasia / Malignant IPMN)
""")

# --- Model Loader ---
@st.cache_resource
def load_diagnostic_model(weight_file: str):
    device = torch.device("cpu")
    model = DenseNet121_3D(in_channels=1, num_classes=3).to(device)
    if os.path.exists(weight_file):
        state_dict = torch.load(weight_file, map_location=device)
        model.load_state_dict(state_dict)
    model.eval()
    return model, device

# --- Volume Preprocessor ---
def preprocess_nii_volume(volume: np.ndarray, target_shape=(64, 128, 128)):
    zoom_factors = [t / s for t, s in zip(target_shape, volume.shape)]
    resampled = zoom(volume, zoom=zoom_factors, order=1, prefilter=False)
    p_low, p_high = np.percentile(resampled, 0.5), np.percentile(resampled, 99.5)
    clipped = np.clip(resampled, p_low, p_high)
    mean, std = np.mean(clipped), np.std(clipped) + 1e-8
    normalized = (clipped - mean) / std
    tensor = torch.from_numpy(normalized).unsqueeze(0).unsqueeze(0).float()
    return tensor

# --- Main UI Upload Area ---
uploaded_file = st.file_uploader(
    "Upload Patient 3D Volumetric MRI Scan (.nii or .nii.gz)",
    type=["nii", "gz"]
)

if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".nii.gz") as tmp:
        tmp.write(uploaded_file.read())
        temp_scan_path = tmp.name

    try:
        nii = nib.load(temp_scan_path)
        nii_ras = nib.as_closest_canonical(nii)
        raw_volume = nii_ras.get_fdata(dtype=np.float32)

        st.success(f"Loaded: `{uploaded_file.name}` | Dimensions: `{raw_volume.shape}`")

        # Multi-Planar Orthogonal Slices
        st.subheader("Multi-Planar MRI Visualizer")
        c1, c2, c3 = st.columns(3)

        mid_d = raw_volume.shape[0] // 2
        mid_h = raw_volume.shape[1] // 2
        mid_w = raw_volume.shape[2] // 2

        with c1:
            st.caption("Axial View (Transverse Plane)")
            fig1, ax1 = plt.subplots(figsize=(3.5, 3.5))
            ax1.imshow(np.rot90(raw_volume[mid_d, :, :]), cmap="gray")
            ax1.axis("off")
            st.pyplot(fig1)

        with c2:
            st.caption("Coronal View (Frontal Plane)")
            fig2, ax2 = plt.subplots(figsize=(3.5, 3.5))
            ax2.imshow(np.rot90(raw_volume[:, mid_h, :]), cmap="gray")
            ax2.axis("off")
            st.pyplot(fig2)

        with c3:
            st.caption("Sagittal View (Lateral Plane)")
            fig3, ax3 = plt.subplots(figsize=(3.5, 3.5))
            ax3.imshow(np.rot90(raw_volume[:, :, mid_w]), cmap="gray")
            ax3.axis("off")
            st.pyplot(fig3)

        # Trigger Classification
        if st.button("Run Diagnostic Risk Stratification", type="primary"):
            with st.spinner("Processing 3D volume through PanSegNet & 3D DenseNet-121..."):
                model, device = load_diagnostic_model(selected_weight_path)
                input_tensor = preprocess_nii_volume(raw_volume).to(device)

                with torch.no_grad():
                    logits = model(input_tensor)
                    probabilities = F.softmax(logits, dim=1).squeeze().cpu().numpy()
                    pred_class = int(np.argmax(probabilities))

            st.markdown("---")
            st.subheader("Diagnostic Risk Stratification Report")

            labels = [
                ("No Risk (Benign / Healthy)", "success"),
                ("Low Risk (Low-Grade Dysplasia IPMN)", "warning"),
                ("High Risk (High-Grade Dysplasia / Malignant IPMN)", "error")
            ]

            badge_text, badge_style = labels[pred_class]
            if badge_style == "success":
                st.success(f"### Classification: {badge_text}")
            elif badge_style == "warning":
                st.warning(f"### Classification: {badge_text}")
            else:
                st.error(f"### Classification: {badge_text}")

            # Confidence Metric Cards
            m1, m2, m3 = st.columns(3)
            m1.metric("Class 0: No Risk", f"{probabilities[0] * 100:.2f}%")
            m2.metric("Class 1: Low Risk", f"{probabilities[1] * 100:.2f}%")
            m3.metric("Class 2: High Risk", f"{probabilities[2] * 100:.2f}%")

            st.progress(float(probabilities[pred_class]))

    finally:
        if os.path.exists(temp_scan_path):
            os.remove(temp_scan_path)
else:
    st.info("Upload a patient NIfTI file (`.nii.gz`) or test using `data/raw/sample_scan.nii.gz`.")