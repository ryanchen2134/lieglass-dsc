"""Smoke test for the bidirectional multi-stage UT-Adapter fusion model.

Run from repo root:
    python scripts/verify_fusion.py
"""
from __future__ import annotations

import torch

from deception_detection.config import ModelConfig
from deception_detection.models.fusion_model import FusionModel


def main() -> None:
    config = ModelConfig()
    config.device = "cpu"
    model = FusionModel(config).to("cpu").eval()

    B = 2
    T = 16000 * 3                                            # 3 seconds
    N = 32

    waveform = torch.randn(B, T)
    waveform_mask = torch.ones(B, T, dtype=torch.bool)
    waveform_mask[1, T // 2 :] = False

    frames = torch.rand(B, N, 1, 224, 224)
    frame_mask = torch.ones(B, N, dtype=torch.bool)
    frame_mask[1, -8:] = False

    batch = {
        "waveform": waveform,
        "waveform_mask": waveform_mask,
        "frames": frames,
        "frame_mask": frame_mask,
    }

    # ---- 1. Forward shape check ----
    logits = model(batch)
    assert logits.shape == (B,), f"expected ({B},), got {logits.shape}"
    print(f"OK  logits shape={tuple(logits.shape)}  values={logits.tolist()}")

    # ---- 2. Multi-stage outputs ----
    audio_stages, a_mask = model.audio_model.forward_multistage(waveform, waveform_mask)
    print(f"OK  audio stages: {len(audio_stages)} of {tuple(audio_stages[0].shape)};  a_mask {tuple(a_mask.shape)}")
    assert len(audio_stages) == len(config.audio_fusion_layers)
    assert all(s.shape == audio_stages[0].shape for s in audio_stages)

    # Re-normalize visual input the same way the model does internally.
    img_mean = model._img_mean
    img_std = model._img_std
    frames_norm = (frames.float() - img_mean) / img_std
    visual_stages, v_mask = model.visual_model.forward_multistage(frames_norm, frame_mask)
    print(f"OK  visual stages: {len(visual_stages)} of {tuple(visual_stages[0].shape)};  v_mask {tuple(v_mask.shape)}")
    assert len(visual_stages) == len(config.visual_fusion_layers)

    # ---- 3. Trainable param breakdown ----
    def count(prefix: str) -> tuple[int, int]:
        params = [(n, p) for n, p in model.named_parameters() if n.startswith(prefix)]
        trainable = sum(p.numel() for _, p in params if p.requires_grad)
        total = sum(p.numel() for _, p in params)
        return trainable, total

    print("\nTrainable / total params by submodule:")
    for prefix in ("audio_model", "visual_model", "fusion", "head"):
        tr, tot = count(prefix)
        print(f"  {prefix:13s}  {tr:>11,d} / {tot:>11,d}  ({100*tr/max(tot,1):5.1f}% trainable)")
    grand_tr = sum(p.numel() for p in model.parameters() if p.requires_grad)
    grand_tot = sum(p.numel() for p in model.parameters())
    print(f"  {'TOTAL':13s}  {grand_tr:>11,d} / {grand_tot:>11,d}")

    # ---- 4. Backward pass; verify frozen vs. trainable ----
    model.train()
    logits = model(batch)
    logits.sum().backward()

    layer0 = model.audio_model.w2v2.encoder.layers[0]
    if hasattr(layer0, "u_attn"):
        # Wrapped layer: q_proj must be frozen; adapter must have grad.
        q_grad = layer0.attention.q_proj.weight.grad
        u_grad = layer0.u_attn.l1.weight.grad
        assert q_grad is None, "frozen Wav2Vec2 q_proj should have no grad"
        assert u_grad is not None, "UT-Adapter l1 should receive gradient"
        print("\nOK  frozen attention has no grad; UT-Adapter has grad.")
    else:
        print("\n(no UT-Adapters wrapped — config.use_ut_adapters is False)")


if __name__ == "__main__":
    main()
