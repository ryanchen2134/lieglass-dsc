"""
Real-time inference processor for the GUI.

Loads a FusionModel that mirrors the architecture in ``deception_detection``
and the per-run configuration written next to each checkpoint
(``checkpoints/<run>/config.json``). The processor consumes face crops from
``frame_input_queue`` and audio samples from ``audio_input_queue``, maintains
sliding windows over both modalities, and pushes deception scores onto
``result_queue``.
"""

from __future__ import annotations

import json
import queue
import threading
import time
from collections import deque
from pathlib import Path

import numpy as np
import torch

from Deploy.config import ModelConfig
from Deploy.fusion_model import FusionModel


# Sliding-window length over face frames. Independent of ``config.max_frames``
# (which is the architecture's safety cap); 64 frames ≈ 2 s of reaction time.
WINDOW_FRAMES = 64
AUDIO_SR = 16000
WINDOW_AUDIO_SECONDS = 2


def _resolve_config_path(checkpoint_path: Path, explicit: str | Path | None) -> Path | None:
    if explicit is not None:
        p = Path(explicit)
        if p.is_file():
            return p
    sibling = checkpoint_path.parent / "config.json"
    if sibling.is_file():
        return sibling
    return None


def load_config_for_checkpoint(checkpoint_path: str | Path,
                               config_path: str | Path | None = None) -> ModelConfig:
    """Build a ``ModelConfig`` for the run that produced ``checkpoint_path``.

    Looks for ``config.json`` next to the checkpoint (or at ``config_path`` if
    given). Falls back to ``ModelConfig()`` defaults if no JSON is present.
    """
    ckpt = Path(checkpoint_path)
    cfg_path = _resolve_config_path(ckpt, config_path)
    if cfg_path is None:
        print(f"[inference] No config.json near {ckpt}; using default ModelConfig.")
        return ModelConfig()
    print(f"[inference] Loading config from {cfg_path}")
    return ModelConfig.from_json(cfg_path)


class LocalInferenceProcessor(threading.Thread):
    def __init__(self, checkpoint_path: str | Path, result_queue: queue.Queue,
                 config_path: str | Path | None = None,
                 window_frames: int = WINDOW_FRAMES):
        super().__init__(daemon=True)
        self.result_queue = result_queue
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 1. Resolve config from the checkpoint's run directory.
        self.config = load_config_for_checkpoint(checkpoint_path, config_path)

        # 2. Build model and load weights (state dict was saved on a possibly
        #    different device — let torch.load remap it).
        self.model = FusionModel(self.config)
        state_dict = torch.load(str(checkpoint_path), map_location="cpu")
        # Some training scripts wrap the dict under a key like "model" or "state_dict".
        if isinstance(state_dict, dict) and "model" in state_dict and isinstance(state_dict["model"], dict):
            state_dict = state_dict["model"]
        elif isinstance(state_dict, dict) and "state_dict" in state_dict and isinstance(state_dict["state_dict"], dict):
            state_dict = state_dict["state_dict"]
        missing, unexpected = self.model.load_state_dict(state_dict, strict=False)
        if missing:
            print(f"[inference] Missing keys: {len(missing)} (showing 5): {missing[:5]}")
        if unexpected:
            print(f"[inference] Unexpected keys: {len(unexpected)} (showing 5): {unexpected[:5]}")
        self.model.to(self.device)
        self.model.eval()

        # 3. Sliding windows. Cap window at config.max_frames to satisfy the
        #    visual model's positional embedding bound.
        cap = int(getattr(self.config, "max_frames", window_frames) or window_frames)
        self.window_frames = max(1, min(window_frames, cap))
        self.frame_buffer = deque(maxlen=self.window_frames)
        self.audio_buffer = deque(maxlen=AUDIO_SR * WINDOW_AUDIO_SECONDS)

        self.frame_input_queue = queue.Queue(maxsize=128)
        self.audio_input_queue = queue.Queue(maxsize=1000)
        self.stop_event = threading.Event()

    def run(self):
        with torch.no_grad():
            while not self.stop_event.is_set():
                while not self.frame_input_queue.empty():
                    self.frame_buffer.append(self.frame_input_queue.get_nowait())
                while not self.audio_input_queue.empty():
                    self.audio_buffer.append(self.audio_input_queue.get_nowait())

                if len(self.frame_buffer) >= self.window_frames:
                    if len(self.audio_buffer) < AUDIO_SR:
                        audio_array = np.zeros(AUDIO_SR, dtype=np.float32)
                    else:
                        audio_array = np.array(list(self.audio_buffer), dtype=np.float32)

                    waveform = torch.from_numpy(audio_array).unsqueeze(0).to(self.device)

                    # Frames: list of (H, W) float32 arrays in [0, 1]; stack to
                    # (N, H, W) -> (1, N, 1, H, W). The visual encoder expands
                    # 1->3 channels internally to feed RGB-pretrained backbones.
                    frames_list = [torch.from_numpy(f) for f in self.frame_buffer]
                    frames_tensor = torch.stack(frames_list).unsqueeze(0).unsqueeze(2).to(self.device)

                    batch = {
                        "waveform": waveform,
                        "frames": frames_tensor,
                        "waveform_mask": None,
                        "frame_mask": None,
                    }

                    logits = self.model(batch)
                    score = torch.sigmoid(logits).item()
                    self.result_queue.put({"score": score})
                    time.sleep(0.01)
                else:
                    time.sleep(0.1)
