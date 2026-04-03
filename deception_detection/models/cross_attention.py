import torch
import torch.nn as nn
import torch.nn.functional as F
from .positional_encoding import ContinuousPositionalEncoding


class ModalityAligner(nn.Module):
    """
    Cross-attention module that aligns a modality sequence (MFCC or landmarks)
    to text-token resolution.

    Text embeddings → QUERIES.
    Modality embeddings → KEYS and VALUES.
    Output: (B, n, d_kv) — one aligned vector per text token.

    Continuous sinusoidal PE based on real timestamps is added to both
    queries and keys so attention learns temporal correspondence.
    """

    def __init__(self, d_query: int, d_kv: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_kv % n_heads == 0, f"d_kv ({d_kv}) must be divisible by n_heads ({n_heads})"
        self.n_heads = n_heads
        self.d_head = d_kv // n_heads
        self.scale = self.d_head ** -0.5

        self.q_proj = nn.Linear(d_query, d_kv)
        self.k_proj = nn.Linear(d_kv, d_kv)
        self.v_proj = nn.Linear(d_kv, d_kv)
        self.out_proj = nn.Linear(d_kv, d_kv)

        # Separate PE for query (text space) and key (modality space)
        self.pe_query = ContinuousPositionalEncoding(d_query)
        self.pe_key = ContinuousPositionalEncoding(d_kv)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        text_emb: torch.Tensor,
        text_timestamps: torch.Tensor,
        modality_emb: torch.Tensor,
        modality_timestamps: torch.Tensor,
        modality_mask=None,
        text_padding_mask=None,
        modality_padding_mask=None,
    ) -> torch.Tensor:
        """
        Args:
            text_emb:             (B, n, d_query)
            text_timestamps:      (B, n, 2) — [t_start, t_end] per token
            modality_emb:         (B, T, d_kv)
            modality_timestamps:  (B, T) — center time per frame
            modality_mask:        (B, T) BoolTensor, True = VALID frame (optional)
            text_padding_mask:    (B, n) BoolTensor, True = PAD (optional)
            modality_padding_mask:(B, T) BoolTensor, True = PAD (optional)
        Returns:
            (B, n, d_kv)
        """
        B, n, _ = text_emb.shape
        T = modality_emb.shape[1]

        # Add continuous PE
        text_mid = (text_timestamps[..., 0] + text_timestamps[..., 1]) / 2.0  # (B, n)
        text_with_pe = text_emb + self.pe_query(text_mid)           # (B, n, d_query)
        mod_with_pe = modality_emb + self.pe_key(modality_timestamps) # (B, T, d_kv)

        # Project to Q, K, V
        Q = self.q_proj(text_with_pe)    # (B, n, d_kv)
        K = self.k_proj(mod_with_pe)     # (B, T, d_kv)
        V = self.v_proj(modality_emb)    # (B, T, d_kv) — V from raw (no PE)

        # Reshape for multi-head attention
        Q = Q.view(B, n, self.n_heads, self.d_head).transpose(1, 2)  # (B, h, n, d_head)
        K = K.view(B, T, self.n_heads, self.d_head).transpose(1, 2)  # (B, h, T, d_head)
        V = V.view(B, T, self.n_heads, self.d_head).transpose(1, 2)  # (B, h, T, d_head)

        # Scaled dot-product attention
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale  # (B, h, n, T)

        # Combined key mask: pad positions OR blacked-out frames
        key_mask = torch.zeros(B, T, dtype=torch.bool, device=attn_scores.device)
        if modality_padding_mask is not None:
            key_mask = key_mask | modality_padding_mask
        if modality_mask is not None:
            key_mask = key_mask | (~modality_mask)   # modality_mask True=valid → invert

        if key_mask.any():
            attn_scores = attn_scores.masked_fill(key_mask[:, None, None, :], float("-inf"))

        attn_weights = F.softmax(attn_scores, dim=-1)  # (B, h, n, T)
        # Handle all-masked rows (softmax of all -inf → nan): replace with uniform
        attn_weights = torch.nan_to_num(attn_weights, nan=0.0)
        attn_weights = self.dropout(attn_weights)

        out = torch.matmul(attn_weights, V)               # (B, h, n, d_head)
        out = out.transpose(1, 2).contiguous().view(B, n, -1)  # (B, n, d_kv)
        return self.out_proj(out)
