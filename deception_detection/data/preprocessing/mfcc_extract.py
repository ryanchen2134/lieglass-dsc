"""
Step 3: MFCC Extraction with Spectral Subtraction, Delta/Delta-Delta, and CMVN.

GPU-accelerated via torchaudio (STFT, mel filterbank, DCT, deltas all on CUDA).
Falls back to CPU multiprocessing when --device cpu --workers N is used.

For each audio.wav:
1. Load at 16kHz mono.
2. Spectral subtraction: estimate noise from first 0.5s, subtract, clip negatives.
3. Compute 13 MFCCs via mel filterbank + DCT (n_fft=400/25ms, hop=160/10ms).
4. Compute delta (order=1) and delta-delta (order=2) via torchaudio.
5. Stack → (T_m, 13, 3).
6. CMVN: per-utterance normalization.
7. Compute timestamps.
8. Save mfcc.pt: {mfcc: FloatTensor(T_m, 13, 3), timestamps: FloatTensor(T_m,)}
"""

import torch
import torchaudio
import torchaudio.functional as F_audio
from pathlib import Path

N_MFCC = 13
N_MELS = 128
SR = 16000
N_FFT = 400       # 25ms at 16kHz
HOP_LENGTH = 160  # 10ms at 16kHz
N_NOISE_FRAMES = int(0.5 * SR / HOP_LENGTH)  # frames in first 0.5s

# Per-process cache: device str → (MelScale transform, DCT matrix)
_transform_cache: dict = {}


def _get_transforms(device: torch.device):
    key = str(device)
    if key not in _transform_cache:
        mel_scale = torchaudio.transforms.MelScale(
            n_mels=N_MELS, sample_rate=SR, n_stft=N_FFT // 2 + 1
        ).to(device)
        dct_mat = F_audio.create_dct(N_MFCC, N_MELS, norm="ortho").T.to(device)  # (N_MFCC, N_MELS)
        _transform_cache[key] = (mel_scale, dct_mat)
    return _transform_cache[key]


def extract_mfcc(audio_path: Path, output_dir: Path, device: torch.device) -> bool:
    """
    Extract MFCC features from audio.wav and save mfcc.pt.
    All heavy computation runs on `device` (CUDA or CPU).
    Returns True on success.
    """
    out_pt = output_dir / "mfcc.pt"
    if out_pt.exists():
        return True

    if not audio_path.exists():
        raise FileNotFoundError(f"audio.wav not found: {audio_path}")

    # Load audio
    waveform, sr_orig = torchaudio.load(str(audio_path))
    if sr_orig != SR:
        waveform = torchaudio.functional.resample(waveform, sr_orig, SR)
    waveform = waveform.mean(0)  # mono (T,)

    if waveform.numel() == 0:
        torch.save(
            {
                "mfcc": torch.zeros(1, N_MFCC, 3, dtype=torch.float32),
                "timestamps": torch.zeros(1, dtype=torch.float32),
            },
            str(out_pt),
        )
        return True

    waveform = waveform.to(device)

    # STFT → power spectrum  (freq, T_m)
    window = torch.hann_window(N_FFT, device=device)
    stft = torch.stft(
        waveform, n_fft=N_FFT, hop_length=HOP_LENGTH,
        win_length=N_FFT, window=window,
        center=True, return_complex=True,
    )
    power = stft.abs().pow(2)  # (freq, T_m)

    # Spectral subtraction: subtract first-0.5s noise estimate
    n_noise = min(N_NOISE_FRAMES, power.shape[1])
    noise_est = power[:, :n_noise].mean(dim=1, keepdim=True)
    power_clean = torch.clamp(power - noise_est, min=1e-10)

    # power_to_db: 10 * log10(power)
    db_spec = 10.0 * torch.log10(power_clean)  # (freq, T_m)

    # Mel filterbank + DCT → MFCCs
    mel_scale, dct_mat = _get_transforms(device)
    mel_db = mel_scale(db_spec)                      # (N_MELS, T_m)
    mfcc = torch.matmul(dct_mat, mel_db)             # (N_MFCC, T_m)

    # Delta and delta-delta
    delta1 = F_audio.compute_deltas(mfcc)            # (N_MFCC, T_m)
    delta2 = F_audio.compute_deltas(delta1)          # (N_MFCC, T_m)

    # Stack → (N_MFCC, T_m, 3) → (T_m, N_MFCC, 3)
    mfcc_stacked = torch.stack([mfcc, delta1, delta2], dim=-1).permute(1, 0, 2)

    T_m = mfcc_stacked.shape[0]

    # CMVN: per-utterance mean/std
    flat = mfcc_stacked.reshape(T_m, -1)
    mean = flat.mean(0, keepdim=True)
    std = flat.std(0, keepdim=True) + 1e-8
    mfcc_norm = ((flat - mean) / std).reshape(T_m, N_MFCC, 3).cpu()

    # Timestamps: center of each frame
    timestamps = (
        torch.arange(T_m, dtype=torch.float32) * (HOP_LENGTH / SR)
        + (N_FFT / 2 / SR)
    )

    torch.save({"mfcc": mfcc_norm, "timestamps": timestamps}, str(out_pt))
    return True


# ── Multiprocessing worker (CPU mode) ────────────────────────────────────────

def _worker_task(args: tuple) -> tuple:
    import traceback
    sample_id, audio_path_str, output_dir_str = args
    try:
        extract_mfcc(Path(audio_path_str), Path(output_dir_str), torch.device("cpu"))
        return sample_id, True, ""
    except Exception:
        return sample_id, False, traceback.format_exc()


def main():
    import argparse
    import csv
    import multiprocessing
    import traceback
    from tqdm import tqdm

    PROJECT_ROOT = Path(__file__).resolve().parents[3]

    parser = argparse.ArgumentParser(description="Step 3: MFCC extraction → mfcc.pt")
    parser.add_argument("--manifest", default="Data/manifest_dolos.csv")
    parser.add_argument("--feature_dir", default="features")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--workers", type=int, default=1,
                        help="CPU worker processes (only used when --device cpu)")
    parser.add_argument("--sample_id", default=None, help="Process a single sample only")
    args = parser.parse_args()

    manifest_path = PROJECT_ROOT / args.manifest
    feature_dir = PROJECT_ROOT / args.feature_dir
    device = torch.device(args.device)

    with open(manifest_path, newline="") as f:
        samples = list(csv.DictReader(f))

    if args.sample_id:
        samples = [s for s in samples if s["sample_id"] == args.sample_id]
        if not samples:
            print(f"Sample not found: {args.sample_id}")
            return

    # Filter already-done
    pending = [
        s for s in samples
        if not (feature_dir / s["sample_id"] / "mfcc.pt").exists()
    ]
    print(f"Device: {device} | Workers: {args.workers if device.type == 'cpu' else 'N/A (GPU)'}")
    print(f"Samples: {len(samples)} total, {len(samples) - len(pending)} already done, {len(pending)} pending")

    if not pending:
        print("Nothing to do.")
        return

    errors = []

    if device.type == "cuda" or args.workers <= 1:
        # Sequential — GPU inference or single CPU worker
        for row in tqdm(pending, desc="mfcc", unit="sample"):
            sample_id = row["sample_id"]
            sample_dir = feature_dir / sample_id
            try:
                extract_mfcc(sample_dir / "audio.wav", sample_dir, device)
            except Exception as e:
                errors.append((sample_id, str(e)))
                tqdm.write(f"  [ERROR] {sample_id}: {e}")
                tqdm.write(traceback.format_exc())
    else:
        # CPU multiprocessing
        tasks = [
            (s["sample_id"],
             str(feature_dir / s["sample_id"] / "audio.wav"),
             str(feature_dir / s["sample_id"]))
            for s in pending
        ]
        ctx = multiprocessing.get_context("spawn")
        with ctx.Pool(args.workers) as pool:
            for sample_id, ok, tb in tqdm(
                pool.imap_unordered(_worker_task, tasks),
                total=len(tasks), desc="mfcc", unit="sample"
            ):
                if not ok:
                    errors.append((sample_id, tb.splitlines()[-1]))
                    tqdm.write(f"  [ERROR] {sample_id}")
                    tqdm.write(tb)

    print(f"\nDone. {len(pending) - len(errors)}/{len(pending)} succeeded.")
    if errors:
        for sid, err in errors:
            print(f"  {sid}: {err}")


if __name__ == "__main__":
    main()
