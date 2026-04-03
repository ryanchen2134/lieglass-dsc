import torch
import torch.nn as nn


class ContinuousPositionalEncoding(nn.Module):
    """
    Sinusoidal positional encodings from real-valued timestamps (in seconds).

    PE(t, 2i)   = sin(t / omega_i)
    PE(t, 2i+1) = cos(t / omega_i)

    where omega_i = 10000^(2i / d_model).

    Added to embeddings (not concatenated), so d_pe must equal d_model.
    """

    def __init__(self, d_model: int):
        super().__init__()
        self.d_model = d_model
        inv_freq = 1.0 / (10000.0 ** (torch.arange(0, d_model, 2).float() / d_model))
        self.register_buffer("inv_freq", inv_freq)  # (d_model // 2,)

    def forward(self, timestamps: torch.Tensor) -> torch.Tensor:
        """
        Args:
            timestamps: FloatTensor of any shape (...), values in seconds.
        Returns:
            FloatTensor (..., d_model)
        """
        t = timestamps.unsqueeze(-1)                  # (..., 1)
        sinusoid = t * self.inv_freq                  # (..., d_model // 2)
        pe = torch.cat([sinusoid.sin(), sinusoid.cos()], dim=-1)  # (..., d_model)
        return pe
