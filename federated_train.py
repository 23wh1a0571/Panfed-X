import os
import copy
import glob
import pandas as pd
import numpy as np
import nibabel as nib
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from scipy.ndimage import zoom
import matplotlib.pyplot as plt
from core.classify import DenseNet121_3D


# ==========================================
# 1. Volumetric Dataset
# ==========================================
class IPMNDataset(Dataset):
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


# ==========================================
# 2. FedAvg Aggregator Function
# ==========================================
def federated_averaging(global_model, client_models, client_sample_counts):
    """
    Performs Federated Averaging (FedAvg):
    Updates global_model weights as the weighted average of client weights.
    """
    total_samples = sum(client_sample_counts)
    global_dict = global_model.state_dict()

    for key in global_dict.keys():
        weighted_sum = torch.zeros_like(global_dict[key], dtype=torch.float32)
        for i, client_model in enumerate(client_models):
            weight = client_sample_counts[i] / total_samples
            weighted_sum += client_model.state_dict()[key].to(dtype=torch.float32) * weight
        global_dict[key] = weighted_sum.to(global_dict[key].dtype)

    global_model.load_state_dict(global_dict)
    return global_model


# ==========================================
# 3. Federated Simulation Protocol
# ==========================================
def run_federated_learning():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=======================================================")
    print(" PanFed-X: Multi-Center Federated Learning Simulation")
    print("=======================================================")
    print(f"Device: {device}\n")

    csv_path = "data/raw/IPMN_labels_t1_total.csv"
    scans_dir = "data/raw/t1"

    dataset = IPMNDataset(csv_path=csv_path, scans_dir=scans_dir)
    total_samples = len(dataset)
    print(f"[Dataset] Total Scans available: {total_samples}")

    # Hold out 20% for Centralized Global Validation
    test_size = int(0.2 * total_samples)
    federated_pool_size = total_samples - test_size
    fed_pool, global_val_set = random_split(dataset, [federated_pool_size, test_size])

    # Partition federated pool across 3 Hospital Centers (Simulating Non-IID / Multi-site)
    num_clients = 3
    client_names = ["Hospital Alpha (Site 1)", "Hospital Beta (Site 2)", "Hospital Gamma (Site 3)"]
    
    # Split among clients
    base_split = federated_pool_size // num_clients
    rem = federated_pool_size % num_clients
    client_lengths = [base_split + (1 if i < rem else 0) for i in range(num_clients)]
    client_subsets = random_split(fed_pool, client_lengths)

    client_loaders = [
        DataLoader(sub, batch_size=2, shuffle=True, num_workers=0)
        for sub in client_subsets
    ]
    global_val_loader = DataLoader(global_val_set, batch_size=2, shuffle=False, num_workers=0)

    for i, name in enumerate(client_names):
        print(f"  -> {name}: {len(client_subsets[i])} scans (private localized partition)")

    print(f"  -> Central Evaluation Node: {len(global_val_set)} scans\n")

    # Initialize Global Model
    global_model = DenseNet121_3D(in_channels=1, num_classes=3).to(device)
    # Seed with trained weights if available
    init_ckpt = "models/best_densenet121.pth"
    if os.path.exists(init_ckpt):
        print(f"[Init] Initializing Global Model from checkpoint: {init_ckpt}")
        global_model.load_state_dict(torch.load(init_ckpt, map_location=device))

    criterion = nn.CrossEntropyLoss()
    rounds = 3  # Federated communication rounds
    local_epochs = 1  # Local epochs per hospital before aggregation

    round_history = []
    print("\n[Federated Training] Beginning FedAvg Communication Rounds...")

    for r in range(1, rounds + 1):
        print(f"\n--- [Federated Round {r}/{rounds}] ---")
        client_models = []
        client_sample_counts = []

        # Local training loop at each hospital node
        for client_idx in range(num_clients):
            print(f"  Training {client_names[client_idx]} locally for {local_epochs} epoch(s)...")
            client_model = copy.deepcopy(global_model).to(device)
            client_model.train()
            optimizer = torch.optim.AdamW(client_model.parameters(), lr=5e-5, weight_decay=1e-4)

            loader = client_loaders[client_idx]
            for ep in range(local_epochs):
                for inputs, targets in loader:
                    inputs, targets = inputs.to(device), targets.to(device)
                    optimizer.zero_grad()
                    loss = criterion(client_model(inputs), targets)
                    loss.backward()
                    optimizer.step()

            client_models.append(client_model)
            client_sample_counts.append(len(client_subsets[client_idx]))

        # Central Aggregation (FedAvg)
        print("  Aggregating local model weights via FedAvg...")
        global_model = federated_averaging(global_model, client_models, client_sample_counts)

        # Global Validation
        global_model.eval()
        correct, total, total_loss = 0, 0, 0.0
        with torch.no_grad():
            for inputs, targets in global_val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = global_model(inputs)
                loss = criterion(outputs, targets)
                total_loss += loss.item() * targets.size(0)
                preds = torch.argmax(outputs, dim=1)
                correct += (preds == targets).sum().item()
                total += targets.size(0)

        r_loss = total_loss / total
        r_acc = (correct / total) * 100
        round_history.append({"round": r, "loss": r_loss, "val_acc": r_acc})
        print(f"  [Round {r} Global Evaluation] Loss: {r_loss:.4f} | Accuracy: {r_acc:.2f}%")

    # Save FedAvg Global Checkpoint
    os.makedirs("models", exist_ok=True)
    fed_save_path = "models/fedavg_global_densenet121.pth"
    torch.save(global_model.state_dict(), fed_save_path)
    print(f"\n[Saved] Global Federated model checkpoint saved to: {fed_save_path}")

    # Plot Federated Progression
    rounds_x = [h["round"] for h in round_history]
    losses_y = [h["loss"] for h in round_history]
    accs_y = [h["val_acc"] for h in round_history]

    fig, ax1 = plt.subplots(figsize=(7, 4.5))
    ax1.plot(rounds_x, losses_y, 'b-o', label='Global Validation Loss')
    ax1.set_xlabel('Federated Round', fontsize=11)
    ax1.set_ylabel('Loss', color='b', fontsize=11)
    ax1.tick_params(axis='y', labelcolor='b')

    ax2 = ax1.twinx()
    ax2.plot(rounds_x, accs_y, 'g-s', label='Global Validation Acc (%)')
    ax2.set_ylabel('Accuracy (%)', color='g', fontsize=11)
    ax2.tick_params(axis='y', labelcolor='g')

    plt.title('PanFed-X Multi-Center FedAvg Convergence', fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig('federated_convergence.png', dpi=300)
    print("[Success] Federated round plot saved as 'federated_convergence.png'")


if __name__ == "__main__":
    run_federated_learning()