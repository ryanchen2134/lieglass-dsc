import torch
import torch.nn as nn
from .encoders.text_encoder import TextEncoder
from .encoders.mfcc_encoder import MFCCEncoder
from .encoders.landmark_encoder import LandmarkEncoder
from .cross_attention import ModalityAligner
from .classifier import BiDirTransformerClassifier


class MultimodalDeceptionModel(nn.Module):
    """
    Full multimodal deception detection model.

    Pipeline:
    1. Encode each modality independently.
    2. Align MFCC and landmarks to text-token resolution via cross-attention.
    3. Concatenate aligned representations along the feature dimension.
    4. Classify with bidirectional transformer + [CLS] pooling + dense head.
    """

    def __init__(self, config):
        super().__init__()
        self.config = config

        # Encoders — text, mfcc, and landmark encoders are all frozen
        self.text_encoder = TextEncoder(config.text_model_name)

        self.mfcc_encoder = MFCCEncoder(
            n_coefficients=config.n_mfcc_coefficients,
            d_out=config.d_mfcc,
        )
        for p in self.mfcc_encoder.parameters():
            p.requires_grad = False

        self.landmark_encoder = LandmarkEncoder(
            n_landmarks=config.n_landmarks,
            coords=config.landmark_coords,
            d_out=config.d_landmark,
        )
        for p in self.landmark_encoder.parameters():
            p.requires_grad = False

        # Frozen encoders with BatchNorm must stay in eval mode permanently so
        # their BN uses running stats (not batch stats) consistently during
        # both training and validation — eliminating the train/eval discrepancy.
        self.mfcc_encoder.eval()
        self.landmark_encoder.eval()

        # Cross-attention aligners
        self.mfcc_aligner = ModalityAligner(
            d_query=config.d_text,
            d_kv=config.d_mfcc,
            n_heads=config.cross_attn_heads,
            dropout=config.cross_attn_dropout,
        )
        self.landmark_aligner = ModalityAligner(
            d_query=config.d_text,
            d_kv=config.d_landmark,
            n_heads=config.cross_attn_heads,
            dropout=config.cross_attn_dropout,
        )

        # Classifier
        self.classifier = BiDirTransformerClassifier(
            d_fused=config.d_text + config.d_mfcc + config.d_landmark,
            n_layers=config.n_clf_layers,
            n_heads=config.clf_heads,
            ff_mult=config.clf_ff_mult,
            dropout=config.dropout,
        )

    def train(self, mode: bool = True):
        super().train(mode)
        # Keep frozen encoders in eval mode regardless of outer training mode.
        self.mfcc_encoder.eval()
        self.landmark_encoder.eval()
        return self

    def forward(self, batch: dict) -> torch.Tensor:
        """
        Args:
            batch: dict from collate_fn.
        Returns:
            logits: (B,)
        """
        # 1. Text encoding (frozen)
        text_emb = self.text_encoder(
            batch["text_token_ids"],
            attention_mask=~batch["text_padding_mask"],  # HF expects True=valid
        )  # (B, n, d_text)

        # 2. MFCC encoding
        mfcc_emb = self.mfcc_encoder(
            batch["mfcc"],
            padding_mask=batch["mfcc_padding_mask"],
        )  # (B, T_m, d_mfcc)

        # 3. Landmark encoding
        land_emb = self.landmark_encoder(
            batch["landmarks"],
            padding_mask=batch["landmark_padding_mask"],
        )  # (B, T_l, d_landmark)

        # 4. Cross-attend MFCC → text resolution
        mfcc_aligned = self.mfcc_aligner(
            text_emb=text_emb,
            text_timestamps=batch["text_timestamps"],
            modality_emb=mfcc_emb,
            modality_timestamps=batch["mfcc_timestamps"],
            modality_mask=None,
            text_padding_mask=batch["text_padding_mask"],
            modality_padding_mask=batch["mfcc_padding_mask"],
        )  # (B, n, d_mfcc)

        # 5. Cross-attend landmarks → text resolution
        land_aligned = self.landmark_aligner(
            text_emb=text_emb,
            text_timestamps=batch["text_timestamps"],
            modality_emb=land_emb,
            modality_timestamps=batch["landmark_timestamps"],
            modality_mask=batch["frame_mask"],
            text_padding_mask=batch["text_padding_mask"],
            modality_padding_mask=batch["landmark_padding_mask"],
        )  # (B, n, d_landmark)

        # 6. Concatenate (not add) along feature dim
        fused = torch.cat([text_emb, mfcc_aligned, land_aligned], dim=-1)
        # (B, n, d_text + d_mfcc + d_landmark = 896)

        # 7. Classify
        return self.classifier(fused, padding_mask=batch["text_padding_mask"])  # (B,)
