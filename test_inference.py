import os
import glob
import numpy as np
import nibabel as nib
import torch
import torch.nn.functional as F
from scipy.ndimage import zoom
from core.classify import DenseNet121_3D

def load_and_preprocess_scan(file_path: str, target_shape=(64, 128, 128)):
    """Loads a 3D NIfTI MRI volume, resamples, and normalizes it."""
    nii = nib.load(file_path)
    nii_ras = nib.as_closest_canonical(nii)
    volume = nii_ras.get_fdata(dtype=np.float32)

    # 3D spline interpolation to match network input shape [64, 128, 128]
    zoom_factors = [t / s for t, s in zip(target_shape, volume.shape)]
    resampled = zoom(volume, zoom=zoom_factors, order=1, prefilter=False)

    # Intensity clipping and Z-score normalization
    p_low, p_high = np.percentile(resampled, 0.5), np.percentile(resampled, 99.5)
    clipped = np.clip(resampled, p_low, p_high)
    mean, std = np.mean(clipped), np.std(clipped) + 1e-8
    normalized = (clipped - mean) / std

    # Shape: [1, 1, Depth, Height, Width]
    tensor = torch.from_numpy(normalized).unsqueeze(0).unsqueeze(0).float()
    return tensor, volume.shape

def run_diagnostic_prediction(scan_path: str, model_path: str = "models/best_densenet121.pth"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n=======================================================")
    print(f" PanFed-X Diagnostic Risk Stratification Engine")
    print(f"=======================================================")
    print(f"Target Scan : {scan_path}")
    print(f"Model File  : {model_path}")
    print(f"Execution   : {device}\n")

    if not os.path.exists(scan_path):
        print(f"[Error] Scan not found at {scan_path}")
        return

    # 1. Initialize architecture and load weights
    model = DenseNet121_3D(in_channels=1, num_classes=3).to(device)
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    # 2. Preprocess scan
    print("[1/2] Loading and standardizing 3D volumetric MRI scan...")
    input_tensor, orig_shape = load_and_preprocess_scan(scan_path)
    input_tensor = input_tensor.to(device)
    print(f"      Original Dimensions: {orig_shape} -> Resampled to: (64, 128, 128)")

    # 3. Model forward pass
    print("[2/2] Running inference through 3D DenseNet-121...")
    with torch.no_grad():
        logits = model(input_tensor)
        probabilities = F.softmax(logits, dim=1).squeeze().cpu().numpy()
        predicted_class = int(torch.argmax(logits, dim=1).item())

    # Risk mapping labels
    risk_labels = {
        0: "No Risk (Benign / Healthy)",
        1: "Low Risk (Low-Grade Dysplasia IPMN)",
        2: "High Risk (High-Grade Dysplasia / Malignant)"
    }

    print("\n---------------- Diagnostic Summary -------------------")
    print(f" Predicted Stratification: {risk_labels[predicted_class]}")
    print(f" Risk Class Index         : {predicted_class}")
    print(f" Confidence Probabilities:")
    print(f"   * No Risk   (Class 0)  : {probabilities[0] * 100:.2f}%")
    print(f"   * Low Risk  (Class 1)  : {probabilities[1] * 100:.2f}%")
    print(f"   * High Risk (Class 2)  : {probabilities[2] * 100:.2f}%")
    print("-------------------------------------------------------\n")

if __name__ == "__main__":
    # Test on the sample scan or the first extracted T1 scan
    sample_file = "data/raw/sample_scan.nii.gz"
    if not os.path.exists(sample_file):
        matched_scans = glob.glob("data/raw/**/*.nii.gz", recursive=True)
        sample_file = matched_scans[0] if matched_scans else "data/raw/t1/t1/AHN05.nii.gz"

    run_diagnostic_prediction(scan_path=sample_file)