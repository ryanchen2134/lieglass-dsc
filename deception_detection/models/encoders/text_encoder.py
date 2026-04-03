import torch
import torch.nn as nn


class TextEncoder(nn.Module):
    """
    Frozen HuggingFace transformer producing per-token contextual embeddings.

    Uses AutoModel.last_hidden_state for token-level outputs.
    All parameters are frozen — no gradients flow through this module.
    """

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        super().__init__()
        from transformers import AutoModel
        self.model = AutoModel.from_pretrained(model_name)
        for p in self.model.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def forward(self, token_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            token_ids:      LongTensor (B, n)
            attention_mask: BoolTensor (B, n) — True where valid (not padded)
        Returns:
            FloatTensor (B, n, 384) — per-token embeddings
        """
        outputs = self.model(
            input_ids=token_ids,
            attention_mask=attention_mask.long(),
        )
        return outputs.last_hidden_state
