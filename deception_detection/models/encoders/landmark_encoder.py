import torch
import torch.nn as nn


class LandmarkEncoder(nn.Module):
    """
    Encodes facial landmark coordinates (with delta/delta-delta) into a learned representation.

    Input:  (B, T_l, n_landmarks * coords, 3)   e.g. (B, T_l, 136, 3)
    Output: (B, T_l, d_out)

    Architecture:
    1. Per-frame MLP: abstracts raw coordinates into facial behavior features.
    2. Temporal 1D conv: captures movement patterns across frames.
    """

    def __init__(self, n_landmarks: int = 68, coords: int = 2, d_out: int = 256):
        super().__init__()
        d_in = n_landmarks * coords * 3  # 68 * 2 * 3 = 408

        self.frame_mlp = nn.Sequential(
            nn.Linear(d_in, 512),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(512, d_out),
            nn.GELU(),
        )

        self.temporal_conv = nn.Sequential(
            nn.Conv1d(d_out, d_out, kernel_size=5, padding=2),
            nn.BatchNorm1d(d_out),
            nn.GELU(),
            nn.Conv1d(d_out, d_out, kernel_size=5, padding=2),
            nn.BatchNorm1d(d_out),
            nn.GELU(),
        )

        self.residual = nn.Identity()

    def forward(self, landmarks: torch.Tensor, padding_mask=None) -> torch.Tensor:
        """
        Args:
            landmarks:    (B, T_l, n_landmarks * coords, 3)
            padding_mask: (B, T_l) BoolTensor, True = padded (optional)
        Returns:
            (B, T_l, d_out)
        """
        B, T, S, C = landmarks.shape
        x = landmarks.reshape(B, T, S * C)      # (B, T, 408)
        x = self.frame_mlp(x)                   # (B, T, d_out)

        residual = self.residual(x)

        x = x.permute(0, 2, 1)                  # (B, d_out, T)
        x = self.temporal_conv(x)
        x = x.permute(0, 2, 1)                  # (B, T, d_out)

        return x + residual
