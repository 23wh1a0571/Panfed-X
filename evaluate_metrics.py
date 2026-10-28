import os
import glob
import pandas as pd
import numpy as np
import nibabel as nib
import torch
from torch.utils.data import Dataset, DataLoader
from scipy.ndimage import zoom
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from core.classify import DenseNet121_3D


class IPMNEvalDataset(Dataset):
    def __init__(self, csv_path: str, scans_dir: str, target_shape=(64, 128, 128)):
        self.target_shape = target_shape
        df = pd.read_csv(csv_path)
        patient_col, label_col = df.columns[0], df.columns[1]

        self.samples = []
        for _, row in df.iterrows():
            file_name = str(row[patient_col]).strip()
            if pd.isna(row[label_col]):
                continue
            try:
                label = int(row[label_col])
            except ValueError:
                continue

            full_path = os.path.join(scans_dir, file_name)
            if not os.path.exists(full_path):
                matched = glob.glob(os.path.join(scans_dir, "**", file_name), recursive=True)
                if matched:
                    full_path = matched[0]
                else:
                    continue
            self.samples.append((full_path, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        file_path, label = self.samples[idx]
        nii = nib.load(file_path)
        nii_ras = nib.as_closest_canonical(nii)
        volume = nii_ras.get_fdata(dtype=np.float32)

        zoom_factors = [t / s for t, s in zip(self.target_shape, volume.shape)]
        resampled = zoom(volume, zoom=zoom_factors, order=1, prefilter=False)

        p_low, p_high = np.percentile(resampled, 0.5), np.percentile(resampled, 99.5)
        clipped = np.clip(resampled, p_low, p_high)
        mean, std = np.mean(clipped), np.std(clipped) + 1e-8
        normalized = (clipped - mean) / std

        tensor = torch.from_numpy(normalized).unsqueeze(0).float()
        return tensor, torch.tensor(label, dtype=torch.long)


def generate_evaluation_report():
    device = torch.device("cpu")
    print("=======================================================")
    print(" PanFed-X: Evaluation & Confusion Matrix Generator")
    print("=======================================================")

    csv_path = "data/raw/IPMN_labels_t1_total.csv"
    scans_dir = "data/raw/t1"

    dataset = IPMNEvalDataset(csv_path=csv_path, scans_dir=scans_dir)
    print(f"[Dataset] Evaluating on holdout validation cohort ({len(dataset)} total scans available)...")

    # Evaluate on a hold-out test slice (last 68 scans matching central test node)
    test_indices = list(range(len(dataset) - 68, len(dataset)))
    test_subset = torch.utils.data.Subset(dataset, test_indices)
    test_loader = DataLoader(test_subset, batch_size=2, shuffle=False)

    # Load Model Checkpoint (prefer federated model if present, else best centralized)
    model_path = "models/fedavg_global_densenet121.pth"
    if not os.path.exists(model_path):
        model_path = "models/best_densenet121.pth"

    print(f"[Model] Loading weights from: {model_path}")
    model = DenseNet121_3D(in_channels=1, num_classes=3).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    all_preds = []
    all_targets = []

    print("[Inference] Generating diagnostic predictions across test cohort...")
    with torch.no_grad():
        for inputs, targets in test_loader:
            outputs = model(inputs)
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())

    class_names = ["No Risk (0)", "Low Risk (1)", "High Risk (2)"]

    # 1. Classification Metrics Report
    print("\n---------------- Detailed Classification Report ----------------")
    print(classification_report(all_targets, all_preds, target_names=class_names, zero_division=0))
    print("----------------------------------------------------------------\n")

    # 2. Confusion Matrix Plot
    cm = confusion_matrix(all_targets, all_preds)

    plt.figure(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names,
                yticklabels=class_names,
                cbar=False, annot_kws={"size": 13, "weight": "bold"})
    plt.title("PanFed-X Multi-Class Confusion Matrix", fontsize=13, fontweight='bold', pad=15)
    plt.xlabel("Predicted Risk Stratification", fontsize=11, labelpad=10)
    plt.ylabel("Ground Truth Pathology Label", fontsize=11, labelpad=10)
    plt.tight_layout()

    cm_filename = "confusion_matrix.png"
    plt.savefig(cm_filename, dpi=300)
    print(f"[Success] Confusion matrix plot saved to: {cm_filename}")
    plt.show()


if __name__ == "__main__":
    generate_evaluation_report()