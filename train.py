import os
import glob
import pandas as pd
import numpy as np
import nibabel as nib
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from scipy.ndimage import zoom
from core.classify import DenseNet121_3D
from tqdm import tqdm


# ==========================================
# 1. Volumetric IPMN Dataset
# ==========================================
class IPMNDataset(Dataset):
    def __init__(self, csv_path: str, scans_dir: str, target_shape=(64, 128, 128)):
        self.target_shape = target_shape
        df = pd.read_csv(csv_path)

        # Standardize column names
        patient_col = df.columns[0]
        label_col = df.columns[1]

        self.samples = []
        for _, row in df.iterrows():
            file_name = str(row[patient_col]).strip()
            if pd.isna(row[label_col]):
                continue
            try:
                label = int(row[label_col])
            except ValueError:
                continue

            # Handle possible nested file paths
            full_path = os.path.join(scans_dir, file_name)
            if not os.path.exists(full_path):
                # Search recursively inside scans_dir
                matched = glob.glob(os.path.join(scans_dir, "**", file_name), recursive=True)
                if matched:
                    full_path = matched[0]
                else:
                    continue

            self.samples.append((full_path, label))

        print(f"[Dataset] Successfully matched {len(self.samples)} valid scans from {csv_path}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        file_path, label = self.samples[idx]

        # 1. Load scan
        nii = nib.load(file_path)
        nii_ras = nib.as_closest_canonical(nii)
        volume = nii_ras.get_fdata(dtype=np.float32)

        # 2. Resample to uniform shape [64, 128, 128]
        zoom_factors = [t / s for t, s in zip(self.target_shape, volume.shape)]
        resampled = zoom(volume, zoom=zoom_factors, order=1, prefilter=False)

        # 3. Z-score intensity normalize
        p_low, p_high = np.percentile(resampled, 0.5), np.percentile(resampled, 99.5)
        clipped = np.clip(resampled, p_low, p_high)
        mean, std = np.mean(clipped), np.std(clipped) + 1e-8
        normalized = (clipped - mean) / std

        # 4. Shape to [Channel, D, H, W]
        tensor = torch.from_numpy(normalized).unsqueeze(0).float()
        return tensor, torch.tensor(label, dtype=torch.long)


# ==========================================
# 2. Training Loop
# ==========================================
def train_densenet():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Device] Using: {device}")

    csv_path = "data/raw/IPMN_labels_t1_total.csv"
    scans_dir = "data/raw/t1"
    os.makedirs("models", exist_ok=True)

    dataset = IPMNDataset(csv_path=csv_path, scans_dir=scans_dir)
    if len(dataset) == 0:
        print("[Error] No scans found. Please verify the folder path.")
        return

    # Train / Validation Split (80/20)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_set, val_set = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_set, batch_size=2, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_set, batch_size=2, shuffle=False, num_workers=0)

    # Initialize DenseNet-121
    model = DenseNet121_3D(in_channels=1, num_classes=3).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)

    epochs = 15
    best_val_acc = 0.0
    save_path = "models/best_densenet121.pth"

    print(f"\n[Training] Starting 3D DenseNet-121 training for {epochs} epochs...")

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        correct, total = 0, 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs} [Train]")
        for inputs, targets in pbar:
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            preds = torch.argmax(outputs, dim=1)
            correct += (preds == targets).sum().item()
            total += targets.size(0)

            pbar.set_postfix({"loss": f"{loss.item():.4f}", "acc": f"{(correct / total) * 100:.1f}%"})

        epoch_loss = running_loss / total
        epoch_acc = (correct / total) * 100

        # Validation
        model.eval()
        val_correct, val_total = 0, 0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                preds = torch.argmax(outputs, dim=1)
                val_correct += (preds == targets).sum().item()
                val_total += targets.size(0)

        val_acc = (val_correct / val_total) * 100 if val_total > 0 else 0.0
        print(f"Epoch {epoch} Result -> Train Loss: {epoch_loss:.4f} | Train Acc: {epoch_acc:.1f}% | Val Acc: {val_acc:.1f}%")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), save_path)
            print(f"  -> Saved new best checkpoint to {save_path} (Val Acc: {val_acc:.1f}%)")

    print("\n[Done] Training complete. Weights saved to models/best_densenet121.pth")


if __name__ == "__main__":
    train_densenet()