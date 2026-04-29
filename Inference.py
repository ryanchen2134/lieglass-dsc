import torch
import cv2
import numpy as np
import torchaudio
import argparse
from pathlib import Path

# Import your existing project components
from deception_detection.config import ModelConfig
from deception_detection.models.fusion_model import FusionModel

def preprocess_video(video_path, n_frames=64):
    """Extracts and resizes frames from video."""
    cap = cv2.VideoCapture(str(video_path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if total <= 0:
        raise ValueError(f"Could not open video or video is empty: {video_path}")

    indices = np.linspace(0, total - 1, n_frames, dtype=int)
    frames = []
    
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            frame = np.zeros((224, 224, 3), dtype=np.uint8)
        else:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (224, 224))
        frames.append(frame)
    
    cap.release()
    # (N, H, W, 3) -> (N, 3, H, W) is usually handled by your model's forward or config
    return torch.from_numpy(np.stack(frames)).permute(0, 3, 1, 2).float() / 255.0

def preprocess_audio(audio_path, target_sr=16000):
    """Loads and resamples audio."""
    waveform, sr = torchaudio.load(str(audio_path))
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if sr != target_sr:
        waveform = torchaudio.functional.resample(waveform, sr, target_sr)
    return waveform.squeeze(0) # (L,)

@torch.inference_mode()
def run_inference(model, video_path, audio_path, device):
    model.eval()
    
    # 1. Prepare Tensors
    frames = preprocess_video(video_path).unsqueeze(0).to(device) # Add batch dim (1, N, 3, H, W)
    audio = preprocess_audio(audio_path).unsqueeze(0).to(device)  # Add batch dim (1, L)
    
    # 2. Build the batch dictionary the model expects
    # Note: Adjust keys based on your FusionModel forward() signature
    batch = {
        "frames": frames,
        "waveform": audio,
        "label": torch.tensor([0.0]).to(device) # Dummy label for loss calculation if needed
    }

    # 3. Forward Pass
    logits = model(batch)
    prob = torch.sigmoid(logits).item()
    
    return prob

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=str, required=True, help="Path to input .mp4")
    parser.add_argument("--audio", type=str, required=True, help="Path to input .wav")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to .pt model weights")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = ModelConfig()
    
    # Load Model
    model = FusionModel(config).to(device)
    print(f"Loading weights from {args.checkpoint}...")
    state_dict = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state_dict)

    # Inference
    percentage = run_inference(model, args.video, args.audio, device)
    
    print("-" * 30)
    print(f"Prediction: {percentage * 100:.2f}% Deceptive")
    print("-" * 30)

if __name__ == "__main__":
    main()