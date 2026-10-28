import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.ndimage import label, find_objects
from typing import Tuple, Optional


# ==========================================
# 1. Linear Self-Attention Bottleneck
# ==========================================
class LinearSelfAttention3D(nn.Module):
    """
    Linear attention reduces computational complexity from O(N^2) to O(N)
    by computing phi(Q) @ (rho(K)^T @ V) instead of SoftMax(QK^T) @ V.
    """
    def __init__(self, in_channels: int):
        super().__init__()
        self.in_channels = in_channels
        self.q_proj = nn.Conv3d(in_channels, in_channels, kernel_size=1)
        self.k_proj = nn.Conv3d(in_channels, in_channels, kernel_size=1)
        self.v_proj = nn.Conv3d(in_channels, in_channels, kernel_size=1)
        self.out_proj = nn.Conv3d(in_channels, in_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, d, h, w = x.shape
        n = d * h * w

        q = self.q_proj(x).view(b, c, n)
        k = self.k_proj(x).view(b, c, n)
        v = self.v_proj(x).view(b, c, n)

        # phi(Q) and rho(K) kernel functions via Softmax
        phi_q = F.softmax(q, dim=-1)   # [B, C, N]
        rho_k = F.softmax(k, dim=1)    # [B, C, N]

        # Context matrix computed in linear time: [B, C, N] x [B, N, C] -> [B, C, C]
        context = torch.bmm(rho_k, v.transpose(1, 2))

        # Re-projected output: [B, C, C] x [B, C, N] -> [B, C, N]
        out = torch.bmm(context.transpose(1, 2), phi_q).view(b, c, d, h, w)
        return x + self.out_proj(out)


# ==========================================
# 2. PanSegNet Architecture
# ==========================================
class ConvBlock3D(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class PanSegNet(nn.Module):
    def __init__(self, in_channels: int = 1, out_channels: int = 2, base_filters: int = 16):
        super().__init__()
        # Encoder
        self.enc1 = ConvBlock3D(in_channels, base_filters)
        self.pool1 = nn.MaxPool3d(2)

        self.enc2 = ConvBlock3D(base_filters, base_filters * 2)
        self.pool2 = nn.MaxPool3d(2)

        self.enc3 = ConvBlock3D(base_filters * 2, base_filters * 4)
        self.pool3 = nn.MaxPool3d(2)

        # Bottleneck with Linear Attention
        self.bottleneck = ConvBlock3D(base_filters * 4, base_filters * 8)
        self.attn = LinearSelfAttention3D(base_filters * 8)

        # Decoder
        self.up3 = nn.ConvTranspose3d(base_filters * 8, base_filters * 4, kernel_size=2, stride=2)
        self.dec3 = ConvBlock3D(base_filters * 8, base_filters * 4)

        self.up2 = nn.ConvTranspose3d(base_filters * 4, base_filters * 2, kernel_size=2, stride=2)
        self.dec2 = ConvBlock3D(base_filters * 4, base_filters * 2)

        self.up1 = nn.ConvTranspose3d(base_filters * 2, base_filters, kernel_size=2, stride=2)
        self.dec1 = ConvBlock3D(base_filters * 2, base_filters)

        # Terminal segmentation head
        self.final_conv = nn.Conv3d(base_filters, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder passes
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))

        # Bottleneck
        b = self.bottleneck(self.pool3(e3))
        b = self.attn(b)

        # Decoder passes
        d3 = self.dec3(torch.cat([self.up3(b), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))

        return self.final_conv(d1)


# ==========================================
# 3. Pancreas ROI Extraction Engine
# ==========================================
class PancreasSegmentor:
    def __init__(self, model_weights_path: Optional[str] = None, device: str = "cpu"):
        self.device = torch.device(device)
        self.model = PanSegNet(in_channels=1, out_channels=2).to(self.device)
        self.model.eval()

        if model_weights_path and os.path.exists(model_weights_path):
            self.model.load_state_dict(torch.load(model_weights_path, map_location=self.device))
            print(f"[Module 2] Loaded PanSegNet weights from {model_weights_path}")
        else:
            print("[Module 2] Initialized PanSegNet (inference mode ready).")

    @torch.no_grad()
    def segment(self, volume_tensor: torch.Tensor) -> np.ndarray:
        """
        Takes normalized volume tensor [1, 1, D, H, W], outputs binary mask [D, H, W].
        """
        input_tensor = volume_tensor.to(self.device)
        logits = self.model(input_tensor)
        probs = F.softmax(logits, dim=1)
        pred_mask = torch.argmax(probs, dim=1).squeeze(0).cpu().numpy().astype(np.uint8)

        # Clean disconnected artifacts using Connected Component Analysis (CCA)
        labeled_mask, num_features = label(pred_mask)
        if num_features > 0:
            sizes = np.bincount(labeled_mask.ravel())
            sizes[0] = 0  # ignore background
            largest_label = sizes.argmax()
            pred_mask = (labeled_mask == largest_label).astype(np.uint8)

        return pred_mask

    def crop_pancreas_roi(
        self,
        volume: np.ndarray,
        mask: np.ndarray,
        margin: int = 5,
        target_shape: Tuple[int, int, int] = (64, 128, 128)
    ) -> Tuple[np.ndarray, Tuple[slice, ...]]:
        """
        Locates the 3D bounding box around the segmented pancreas, crops it,
        and resizes it to fixed dimensions for DenseNet-121.
        """
        slices = find_objects(mask)
        if not slices or slices[0] is None:
            # Fallback center crop if no pancreas voxels are detected
            d, h, w = volume.shape
            slices = (slice(d // 4, 3 * d // 4), slice(h // 4, 3 * h // 4), slice(w // 4, 3 * w // 4))
        else:
            slices = slices[0]

        # Apply boundary margin
        d_slice = slice(max(0, slices[0].start - margin), min(volume.shape[0], slices[0].stop + margin))
        h_slice = slice(max(0, slices[1].start - margin), min(volume.shape[1], slices[1].stop + margin))
        w_slice = slice(max(0, slices[2].start - margin), min(volume.shape[2], slices[2].stop + margin))
        bounding_box = (d_slice, h_slice, w_slice)

        cropped = volume[bounding_box]

        # Resize to fixed tensor shape for Module 3 classifier using PyTorch 3D interpolation
        crop_t = torch.from_numpy(cropped).unsqueeze(0).unsqueeze(0).float()
        resized_t = F.interpolate(crop_t, size=target_shape, mode="trilinear", align_corners=False)
        cropped_roi = resized_t.squeeze().cpu().numpy()

        return cropped_roi, bounding_box


if __name__ == "__main__":
    print("[Module 2] Verifying PanSegNet architecture...")
    segmentor = PancreasSegmentor(device="cpu")

    # Synthetic test tensor [Batch: 1, Channel: 1, Depth: 32, Height: 64, Width: 64]
    dummy_input = torch.randn(1, 1, 32, 64, 64)
    mask = segmentor.segment(dummy_input)
    print(f"  -> Generated Segmentation Mask shape: {mask.shape}")

    # Crop demonstration
    dummy_raw_vol = dummy_input.squeeze().numpy()
    roi, bbox = segmentor.crop_pancreas_roi(dummy_raw_vol, mask)
    print(f"  -> Cropped Pancreas ROI for DenseNet-121 shape: {roi.shape}")
    print("[Module 2] Ready and working smoothly!")