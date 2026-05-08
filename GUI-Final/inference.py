"""
Real-time inference processor for the GUI.

Loads a FusionModel that mirrors the architecture in ``deception_detection``
and the per-run configuration written next to each checkpoint
(``checkpoints/<run>/config.json``). The processor consumes face crops from
``frame_input_queue`` and audio samples from ``audio_input_queue``, maintains
sliding windows over both modalities, and pushes calibrated, smoothed
deception scores (with a confidence interval) onto ``result_queue``.

Score post-processing pipeline:
  raw model logit  ->  divide by TEMPERATURE
                   ->  smoother (EMA or 1-D Kalman, switchable)
                   ->  CI in logit space (Kalman P, or rolling std fallback)
                   ->  sigmoid -> point score and CI bounds in [0, 1]
"""

from __future__ import annotations

import json
import math
import queue
import threading
import time
from collections import deque
from pathlib import Path

import numpy as np
import torch

from Deploy.config import ModelConfig
from Deploy.fusion_model import FusionModel


# ---------------------------------------------------------------------------
# Tunables — adjust freely; sensible defaults below.
# ---------------------------------------------------------------------------

# --- Sliding windows --------------------------------------------------------
WINDOW_FRAMES = 64                     # Face frames per inference (~2 s @ 32 fps)
AUDIO_SR = 16000                       # Wav2Vec2 sample rate
WINDOW_AUDIO_SECONDS = 2               # Audio context length (seconds)

# --- Inference cadence ------------------------------------------------------
# Re-infer only after this many *new* frames have arrived. Adjacent windows
# share most frames, so emitting every tick wastes GPU and produces highly
# correlated, jittery readouts. 8 ≈ 4 emissions/s @ 32 fps.
INFERENCE_STRIDE_FRAMES = 8

# --- Calibration ------------------------------------------------------------
# Temperature scaling on the raw logit. Fit on held-out validation logits
# (NLL minimization). T > 1 softens overconfident predictions; T = 1 is a
# no-op. Quick fit: ``T = argmin BCE(sigmoid(logits / T), labels)``.
TEMPERATURE = 1.0

# --- Smoothing --------------------------------------------------------------
# "ema"     — exponential moving average on the (temperature-scaled) logit.
# "kalman"  — 1-D random-walk Kalman filter on the logit (experimental).
# "none"    — pass-through (raw logit).
SMOOTHER = "ema"

# EMA weight on the new measurement (alpha=1 disables smoothing).
EMA_ALPHA = 0.25

# Kalman parameters (state = scalar logit; constant-position random walk).
#   x_t = x_{t-1} + w,    w ~ N(0, KALMAN_PROCESS_VAR)
#   z_t = x_t      + v,   v ~ N(0, KALMAN_MEAS_VAR)
# Larger Q tracks faster, larger R smooths more.
KALMAN_PROCESS_VAR = 0.05
KALMAN_MEAS_VAR = 0.5
KALMAN_INIT_VAR = 4.0                  # P_0 (large -> first measurement dominates)

# --- Confidence interval ----------------------------------------------------
CI_Z = 1.96                            # 1.96 = 95% CI; 1.0 = ±1σ (~68%)
# Window length used to estimate σ from raw logits when SMOOTHER != "kalman".
CI_FALLBACK_WINDOW = 32


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


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


# ---------------------------------------------------------------------------
# Smoother (returns posterior mean and std of the filtered logit)
# ---------------------------------------------------------------------------


class LogitSmoother:
    """Switchable EMA / Kalman smoother on a scalar logit stream.

    ``update(z)`` returns ``(mean, std)`` where ``mean`` is the filtered logit
    and ``std`` is the estimated 1-σ uncertainty (in logit space).
    """

    def __init__(self,
                 mode: str = SMOOTHER,
                 ema_alpha: float = EMA_ALPHA,
                 kalman_q: float = KALMAN_PROCESS_VAR,
                 kalman_r: float = KALMAN_MEAS_VAR,
                 kalman_p0: float = KALMAN_INIT_VAR,
                 ci_window: int = CI_FALLBACK_WINDOW):
        if mode not in ("ema", "kalman", "none"):
            raise ValueError(f"Unknown SMOOTHER: {mode!r}")
        self.mode = mode
        self.ema_alpha = float(ema_alpha)
        self.q = float(kalman_q)
        self.r = float(kalman_r)
        self.p = float(kalman_p0)
        self.x = 0.0
        self.initialized = False
        self.recent: deque[float] = deque(maxlen=ci_window)

    def reset(self) -> None:
        self.x = 0.0
        self.p = KALMAN_INIT_VAR
        self.initialized = False
        self.recent.clear()

    def update(self, z: float) -> tuple[float, float]:
        self.recent.append(z)

        if self.mode == "kalman":
            if not self.initialized:
                self.x = z
                self.initialized = True
            p_pred = self.p + self.q
            k = p_pred / (p_pred + self.r)
            self.x = self.x + k * (z - self.x)
            self.p = (1.0 - k) * p_pred
            return self.x, math.sqrt(max(self.p, 1e-9))

        if self.mode == "ema":
            if not self.initialized:
                self.x = z
                self.initialized = True
            else:
                self.x = self.ema_alpha * z + (1.0 - self.ema_alpha) * self.x
        else:  # "none"
            self.x = z

        if len(self.recent) >= 2:
            sigma = float(np.std(np.array(self.recent), ddof=1))
        else:
            sigma = math.sqrt(KALMAN_INIT_VAR)
        return self.x, sigma


# ---------------------------------------------------------------------------
# Inference processor
# ---------------------------------------------------------------------------


class LocalInferenceProcessor(threading.Thread):
    def __init__(self, checkpoint_path: str | Path, result_queue: queue.Queue,
                 config_path: str | Path | None = None,
                 window_frames: int = WINDOW_FRAMES,
                 stride_frames: int = INFERENCE_STRIDE_FRAMES,
                 temperature: float = TEMPERATURE,
                 smoother_mode: str = SMOOTHER,
                 ema_alpha: float = EMA_ALPHA,
                 kalman_q: float = KALMAN_PROCESS_VAR,
                 kalman_r: float = KALMAN_MEAS_VAR,
                 kalman_p0: float = KALMAN_INIT_VAR,
                 ci_z: float = CI_Z):
        super().__init__(daemon=True)
        self.result_queue = result_queue
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 1. Resolve config from the checkpoint's run directory.
        self.config = load_config_for_checkpoint(checkpoint_path, config_path)

        # 2. Build model and load weights.
        self.model = FusionModel(self.config)
        state_dict = torch.load(str(checkpoint_path), map_location="cpu")
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

        # 3. Sliding windows (cap at config.max_frames for the visual PE bound).
        cap = int(getattr(self.config, "max_frames", window_frames) or window_frames)
        self.window_frames = max(1, min(window_frames, cap))
        self.stride_frames = max(1, int(stride_frames))
        self.frame_buffer = deque(maxlen=self.window_frames)
        self.audio_buffer = deque(maxlen=AUDIO_SR * WINDOW_AUDIO_SECONDS)

        self.frame_input_queue = queue.Queue(maxsize=128)
        self.audio_input_queue = queue.Queue(maxsize=1000)
        self.stop_event = threading.Event()

        # 4. Score post-processing.
        self.temperature = float(temperature)
        self.ci_z = float(ci_z)
        self.smoother = LogitSmoother(
            mode=smoother_mode,
            ema_alpha=ema_alpha,
            kalman_q=kalman_q,
            kalman_r=kalman_r,
            kalman_p0=kalman_p0,
        )
        self._frames_since_last_infer = 0

    def run(self):
        with torch.no_grad():
            while not self.stop_event.is_set():
                # Drain inputs
                new_frames = 0
                while not self.frame_input_queue.empty():
                    self.frame_buffer.append(self.frame_input_queue.get_nowait())
                    new_frames += 1
                while not self.audio_input_queue.empty():
                    self.audio_buffer.append(self.audio_input_queue.get_nowait())
                self._frames_since_last_infer += new_frames

                ready = (len(self.frame_buffer) >= self.window_frames
                         and self._frames_since_last_infer >= self.stride_frames)
                if not ready:
                    time.sleep(0.05)
                    continue

                self._frames_since_last_infer = 0

                if len(self.audio_buffer) < AUDIO_SR:
                    audio_array = np.zeros(AUDIO_SR, dtype=np.float32)
                else:
                    audio_array = np.array(list(self.audio_buffer), dtype=np.float32)
                waveform = torch.from_numpy(audio_array).unsqueeze(0).to(self.device)

                frames_list = [torch.from_numpy(f) for f in self.frame_buffer]
                frames_tensor = torch.stack(frames_list).unsqueeze(0).unsqueeze(2).to(self.device)

                batch = {
                    "waveform": waveform,
                    "frames": frames_tensor,
                    "waveform_mask": None,
                    "frame_mask": None,
                }

                raw_logit = float(self.model(batch).item())
                scaled = raw_logit / max(self.temperature, 1e-6)
                mean_logit, sigma_logit = self.smoother.update(scaled)

                lo_logit = mean_logit - self.ci_z * sigma_logit
                hi_logit = mean_logit + self.ci_z * sigma_logit
                score = _sigmoid(mean_logit)
                score_lo = _sigmoid(lo_logit)
                score_hi = _sigmoid(hi_logit)

                self.result_queue.put({
                    "score": score,
                    "score_lo": score_lo,
                    "score_hi": score_hi,
                    "logit": mean_logit,
                    "logit_std": sigma_logit,
                    "raw_logit": raw_logit,
                })
