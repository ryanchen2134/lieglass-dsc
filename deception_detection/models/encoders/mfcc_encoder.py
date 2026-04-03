import torch
import torch.nn as nn


class MFCCEncoder(nn.Module):
    """
    Encodes MFCC features (with delta/delta-delta) into a learned representation.

    Input:  (B, T_m, n_coefficients, 3)
    Output: (B, T_m, d_out)

    Architecture: Linear projection → 1D conv stack (local temporal patterns) → residual.
    """

    def __init__(self, n_coefficients: int = 13, d_out: int = 256):
        super().__init__()
        d_in = n_coefficients * 3  # 39

        self.input_proj = nn.Linear(d_in, d_out)

        # 1D conv stack: kernel_size=5 covers ~50ms at 10ms stride
        self.conv_stack = nn.Sequential(
            nn.Conv1d(d_out, d_out, kernel_size=5, padding=2),
            nn.BatchNorm1d(d_out),
            nn.GELU(),
            nn.Conv1d(d_out, d_out, kernel_size=5, padding=2),
            nn.BatchNorm1d(d_out),
            nn.GELU(),
            nn.Conv1d(d_out, d_out, kernel_size=3, padding=1),
            nn.BatchNorm1d(d_out),
            nn.GELU(),
        )
        self.residual = nn.Identity()

    def forward(self, mfcc: torch.Tensor, padding_mask=None) -> torch.Tensor:
        """
        Args:
            mfcc:         (B, T_m, n_coefficients, 3)
            padding_mask: (B, T_m) BoolTensor, True = padded (optional, unused in conv)
        Returns:
            (B, T_m, d_out)
        """
        B, T, D, C = mfcc.shape
        x = mfcc.reshape(B, T, D * C)          # (B, T, 39)
        x = self.input_proj(x)                  # (B, T, d_out)

        residual = self.residual(x)

        x = x.permute(0, 2, 1)                 # (B, d_out, T)
        x = self.conv_stack(x)
        x = x.permute(0, 2, 1)                 # (B, T, d_out)

        return x + residual
