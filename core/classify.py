import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple, Optional


# ==========================================
# 1. 3D DenseNet-121 Core Components
# ==========================================
class _DenseLayer3D(nn.Module):
    def __init__(self, in_features: int, growth_rate: int, bn_size: int, drop_rate: float):
        super().__init__()
        self.drop_rate = drop_rate

        self.norm1 = nn.BatchNorm3d(in_features)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv3d(in_features, bn_size * growth_rate, kernel_size=1, stride=1, bias=False)

        self.norm2 = nn.BatchNorm3d(bn_size * growth_rate)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(bn_size * growth_rate, growth_rate, kernel_size=3, stride=1, padding=1, bias=False)

    def forward(self, prev_features: torch.Tensor) -> torch.Tensor:
        out = self.conv1(self.relu1(self.norm1(prev_features)))
        out = self.conv2(self.relu2(self.norm2(out)))
        if self.drop_rate > 0:
            out = F.dropout3d(out, p=self.drop_rate, training=self.training)
        return torch.cat([prev_features, out], 1)


class _DenseBlock3D(nn.Module):
    def __init__(self, num_layers: int, in_features: int, growth_rate: int, bn_size: int, drop_rate: float):
        super().__init__()
        layers = []
        for i in range(num_layers):
            layer = _DenseLayer3D(
                in_features=in_features + i * growth_rate,
                growth_rate=growth_rate,
                bn_size=bn_size,
                drop_rate=drop_rate
            )
            layers.append(layer)
        self.block = nn.Sequential(*layers)

    def forward(self, init_features: torch.Tensor) -> torch.Tensor:
        return self.block(init_features)


class _Transition3D(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.norm = nn.BatchNorm3d(in_features)
        self.relu = nn.ReLU(inplace=True)
        self.conv = nn.Conv3d(in_features, out_features, kernel_size=1, stride=1, bias=False)
        self.pool = nn.AvgPool3d(kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pool(self.conv(self.relu(self.norm(x))))


class DenseNet121_3D(nn.Module):
    """
    Volumetric 3D DenseNet-121 Architecture for Pancreatic IPMN Risk Prediction.
    Block configuration: (6, 12, 24, 16) layers per dense block.
    """
    def __init__(
        self,
        in_channels: int = 1,
        growth_rate: int = 32,
        block_config: Tuple[int, int, int, int] = (6, 12, 24, 16),
        num_init_features: int = 64,
        bn_size: int = 4,
        drop_rate: float = 0.2,
        num_classes: int = 3
    ):
        super().__init__()

        # Initial volumetric convolution stem
        self.features = nn.Sequential(
            nn.Conv3d(in_channels, num_init_features, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm3d(num_init_features),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
        )

        # Dense blocks with transition layers
        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            block = _DenseBlock3D(
                num_layers=num_layers,
                in_features=num_features,
                growth_rate=growth_rate,
                bn_size=bn_size,
                drop_rate=drop_rate
            )
            self.features.add_module(f'denseblock{i + 1}', block)
            num_features = num_features + num_layers * growth_rate

            if i != len(block_config) - 1:
                trans = _Transition3D(in_features=num_features, out_features=num_features // 2)
                self.features.add_module(f'transition{i + 1}', trans)
                num_features = num_features // 2

        # Final normalization and classification head
        self.features.add_module('norm_final', nn.BatchNorm3d(num_features))
        self.relu_final = nn.ReLU(inplace=True)
        self.global_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(num_features, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.features(x)
        out = self.relu_final(features)
        out = self.global_pool(out)
        out = torch.flatten(out, 1)
        logits = self.classifier(out)
        return logits


# ==========================================
# 2. Risk Classifier Engine
# ==========================================
class PancreaticRiskClassifier:
    CLASS_LABELS = {
        0: "No Risk",
        1: "Low Risk",
        2: "High Risk"
    }

    def __init__(self, model_weights_path: Optional[str] = None, device: str = "cpu"):
        self.device = torch.device(device)
        self.model = DenseNet121_3D(in_channels=1, num_classes=3).to(self.device)
        self.model.eval()

        if model_weights_path and os.path.exists(model_weights_path):
            self.model.load_state_dict(torch.load(model_weights_path, map_location=self.device))
            print(f"[Module 3] Loaded DenseNet-121 weights from {model_weights_path}")
        else:
            print("[Module 3] Initialized 3D DenseNet-121 (inference mode ready).")

    @torch.no_grad()
    def predict(self, roi_numpy: np.ndarray) -> Dict:
        """
        Takes cropped pancreas 3D ROI array of shape [D, H, W] or tensor [1, 1, D, H, W]
        and predicts the risk category and probability distribution.
        """
        if isinstance(roi_numpy, np.ndarray):
            tensor = torch.from_numpy(roi_numpy).float()
            if tensor.ndim == 3:
                tensor = tensor.unsqueeze(0).unsqueeze(0)  # Shape: [1, 1, D, H, W]
        else:
            tensor = roi_numpy

        tensor = tensor.to(self.device)
        logits = self.model(tensor)
        probabilities = F.softmax(logits, dim=1).squeeze(0).cpu().numpy()

        predicted_class_idx = int(np.argmax(probabilities))
        predicted_label = self.CLASS_LABELS[predicted_class_idx]

        results = {
            "predicted_class": predicted_label,
            "confidence": float(probabilities[predicted_class_idx]),
            "probabilities": {
                self.CLASS_LABELS[i]: round(float(prob) * 100, 2)
                for i, prob in enumerate(probabilities)
            }
        }
        return results


# ==========================================
# 3. Quick Test Execution
# ==========================================
if __name__ == "__main__":
    print("[Module 3] Initializing and testing 3D DenseNet-121 classifier...")
    classifier = PancreaticRiskClassifier(device="cpu")

    # Synthetic cropped ROI mimicking Module 2 output [Depth: 64, Height: 128, Width: 128]
    dummy_roi = np.random.randn(64, 128, 128).astype(np.float32)

    result = classifier.predict(dummy_roi)

    print("\n----- Diagnostic Output -----")
    print(f"Risk Assessment: {result['predicted_class']}")
    print(f"Confidence:      {result['confidence'] * 100:.2f}%")
    print(f"Distribution:    {result['probabilities']}")
    print("------------------------------")
    print("[Module 3] Completed successfully!")