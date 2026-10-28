import os
import tkinter as tk
from tkinter import filedialog
from typing import Tuple, Optional
import numpy as np
import nibabel as nib
from scipy.ndimage import zoom
import torch


class MRIPreprocessor:
    def __init__(self, target_spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0)):
        self.target_spacing = target_spacing

    def load_nifti(self, file_path: str):
        nii = nib.load(file_path)
        nii_ras = nib.as_closest_canonical(nii)
        data = nii_ras.get_fdata(dtype=np.float32)
        affine = nii_ras.affine
        original_spacing = nii_ras.header.get_zooms()[:3]
        return data, affine, original_spacing

    def resample_isotropic(self, volume: np.ndarray, original_spacing: Tuple[float, ...]):
        zoom_factors = [
            orig_s / target_s
            for orig_s, target_s in zip(original_spacing, self.target_spacing)
        ]
        return zoom(volume, zoom=zoom_factors, order=3, prefilter=True)

    def normalize_intensity(self, volume: np.ndarray):
        p_low = np.percentile(volume, 0.5)
        p_high = np.percentile(volume, 99.5)
        clipped = np.clip(volume, p_low, p_high)
        foreground = clipped[clipped > 0]
        if len(foreground) > 0:
            mean = np.mean(foreground)
            std = np.std(foreground) + 1e-8
            normalized = (clipped - mean) / std
        else:
            normalized = (clipped - np.mean(clipped)) / (np.std(clipped) + 1e-8)
        return normalized.astype(np.float32)

    def process_file(self, input_path: str, output_path: Optional[str] = None):
        print(f"\n[Module 1] Processing: {input_path}")
        data, affine, original_spacing = self.load_nifti(input_path)
        print(f"  -> Original shape: {data.shape}, Spacing: {original_spacing}")

        resampled = self.resample_isotropic(data, original_spacing)
        print(f"  -> Resampled shape: {resampled.shape}")

        normalized = self.normalize_intensity(resampled)

        tensor = torch.from_numpy(normalized).unsqueeze(0).unsqueeze(0)
        print(f"  -> Tensor shape [B, C, D, H, W]: {tuple(tensor.shape)}")

        if output_path:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            np.save(output_path, normalized)
            print(f"  -> Saved processed volume to: {output_path}")

        return tensor


def select_file_via_dialog():
    """Opens a native Windows file selection dialog."""
    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)

    file_path = filedialog.askopenfilename(
        title="Select an MRI NIfTI File",
        filetypes=[
            ("NIfTI Medical Image", "*.nii;*.nii.gz"),
            ("All Files", "*.*")
        ]
    )
    return file_path


if __name__ == "__main__":
    print("[Action Required] Opening file selector window... Please choose your MRI file.")
    selected_path = select_file_via_dialog()

    if selected_path:
        filename = os.path.basename(selected_path)
        base_name = filename.replace(".nii.gz", "").replace(".nii", "")
        save_dest = os.path.join("data", "processed", f"{base_name}_norm.npy")

        preprocessor = MRIPreprocessor(target_spacing=(1.0, 1.0, 1.0))
        tensor = preprocessor.process_file(selected_path, output_path=save_dest)
        print("\nModule 1 finished successfully!")
    else:
        print("[Cancelled] No file was selected.")