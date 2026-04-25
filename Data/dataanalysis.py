"""
DOLOS Dataset Analysis Script
==============================
Generates and caches the following plots:
  1. 2x2 Mel Spectrogram sample grid (2 truth, 2 deception)
  2. Average Mel Spectrogram per class (truth vs deception)
  3. Pitch (F0) contour distribution by class — violin plot
  4. Average frame-difference motion curve by class

All results are cached in `plot_cache/` as .pkl files.
Re-running skips computation if cache exists.

Requirements:
    pip install opencv-python librosa matplotlib seaborn numpy tqdm scipy pandas
    brew install ffmpeg  (macOS)

Folder structure expected (run from project root):
    Data/
    ├── DOLOS/                  <-- original mp4s
    └── DOLOS Resized/          <-- face-cropped mp4s (for motion analysis)
"""

import os
import pickle
import warnings
import subprocess
import tempfile
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import librosa
import librosa.display
import cv2
import pandas as pd
from tqdm import tqdm
from scipy.interpolate import interp1d

warnings.filterwarnings("ignore")

# ── Configuration ─────────────────────────────────────────────────────────────

DATA_DIR     = "Data/DOLOS"          # original clips
RESIZED_DIR  = "Data/DOLOS Resized"  # face-cropped clips
CACHE_DIR    = "plot_cache"
PLOTS_DIR    = "plots"

SR           = 16000    # audio sample rate
N_MELS       = 64       # mel bands
HOP_LENGTH   = 256
N_FFT        = 1024
MAX_CLIPS    = None     # set e.g. 200 to test on a subset; None = all

SEED         = 42
rng          = np.random.default_rng(SEED)

PALETTE       = {"truth": "#4C9BE8", "deception": "#E8714C"}
LABEL_DISPLAY = {"truth": "Truth", "deception": "Deception"}

sns.set_theme(style="darkgrid", palette="muted", font_scale=1.1)
os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

# ── Helpers ───────────────────────────────────────────────────────────────────

def cache_path(name):
    return os.path.join(CACHE_DIR, f"{name}.pkl")

def save_cache(name, obj):
    with open(cache_path(name), "wb") as f:
        pickle.dump(obj, f)
    print(f"  ✓ Cached → {cache_path(name)}")

def load_cache(name):
    p = cache_path(name)
    if os.path.exists(p):
        with open(p, "rb") as f:
            return pickle.load(f)
    return None

def get_label(filename):
    """Return 'truth', 'deception', or 'unknown' based on filename."""
    name = filename.lower().replace(" ", "")
    if "_truth" in name or "_true" in name:
        return "truth"
    elif "_lie" in name or "_deception" in name:
        return "deception"
    else:
        print(f"  ⚠ Unknown label for: {filename}")
        return "unknown"

def list_clips(directory):
    """Return sorted list of (filepath, label) for all .mp4 files, skipping unknowns."""
    clips = []
    for fname in sorted(os.listdir(directory)):
        if fname.lower().endswith(".mp4"):
            label = get_label(fname)
            if label != "unknown":
                clips.append((os.path.join(directory, fname), label))
    return clips

def extract_audio_array(filepath, sr=SR):
    """Extract mono audio via ffmpeg → temp WAV → librosa. Handles corrupt files."""
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_path = tmp.name
    try:
        result = subprocess.run(
            ["ffmpeg", "-y", "-i", filepath, "-ac", "1", "-ar", str(sr),
             "-loglevel", "error", tmp_path],
            check=False,
            capture_output=True
        )
        if result.returncode != 0 or not os.path.exists(tmp_path):
            raise ValueError(f"ffmpeg failed on {os.path.basename(filepath)}")
        if os.path.getsize(tmp_path) == 0:
            raise ValueError(f"Empty WAV output for {os.path.basename(filepath)}")
        y, _ = librosa.load(tmp_path, sr=sr, mono=True)
        return y
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)

def compute_mel(y, sr=SR):
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=N_MELS,
                                        hop_length=HOP_LENGTH, n_fft=N_FFT)
    return librosa.power_to_db(S, ref=np.max)

def compute_f0(y, sr=SR):
    f0, voiced, _ = librosa.pyin(y, fmin=librosa.note_to_hz("C2"),
                                  fmax=librosa.note_to_hz("C7"), sr=sr)
    f0_voiced = f0[voiced] if voiced is not None else np.array([])
    return float(np.nanmean(f0_voiced)) if len(f0_voiced) > 0 else 0.0

def compute_frame_motion(filepath):
    cap = cv2.VideoCapture(filepath)
    diffs = []
    prev = None
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
        if prev is not None:
            diffs.append(np.mean(np.abs(gray - prev)))
        prev = gray
    cap.release()
    return np.array(diffs)

def interpolate_to_length(arr, n=100):
    x_old = np.linspace(0, 1, len(arr))
    x_new = np.linspace(0, 1, n)
    return interp1d(x_old, arr, kind="linear")(x_new)

# ── Step 1: Collect clip list ─────────────────────────────────────────────────

print("\n━━ Loading clip list ━━")
clips = list_clips(DATA_DIR)
if MAX_CLIPS:
    clips = clips[:MAX_CLIPS]

truth_clips     = [(p, l) for p, l in clips if l == "truth"]
deception_clips = [(p, l) for p, l in clips if l == "deception"]

print(f"  Total clips : {len(clips)}")
print(f"  Truth       : {len(truth_clips)}")
print(f"  Deception   : {len(deception_clips)}")

assert len(truth_clips) > 0,     "❌ No truth clips found — check DATA_DIR and filenames"
assert len(deception_clips) > 0, "❌ No deception clips found — check DATA_DIR and filenames"

# ── Step 2: Audio features ────────────────────────────────────────────────────

print("\n━━ Computing audio features (mel + F0) ━━")
audio_data = load_cache("audio_features")

if audio_data is None:
    records = []
    skipped = 0
    for fpath, label in tqdm(clips, desc="Audio"):
        try:
            y   = extract_audio_array(fpath)
            mel = compute_mel(y)
            f0  = compute_f0(y)
            records.append({"filepath": fpath, "label": label, "mel": mel, "f0_mean": f0})
        except Exception as e:
            skipped += 1
            print(f"  ⚠ Skipped {os.path.basename(fpath)}: {type(e).__name__}: {e}")
    print(f"\n  Processed : {len(records)} clips")
    print(f"  Skipped   : {skipped} clips (corrupt/unreadable)")
    audio_data = records
    save_cache("audio_features", audio_data)
else:
    print("  ✓ Loaded from cache")

truth_audio     = [r for r in audio_data if r["label"] == "truth"]
deception_audio = [r for r in audio_data if r["label"] == "deception"]
print(f"  Usable — Truth: {len(truth_audio)}, Deception: {len(deception_audio)}")

assert len(truth_audio) >= 2,     "❌ Need at least 2 usable truth clips for plots"
assert len(deception_audio) >= 2, "❌ Need at least 2 usable deception clips for plots"

# ── Step 3: Motion features ───────────────────────────────────────────────────

print("\n━━ Computing frame motion (DOLOS Resized) ━━")
motion_data = load_cache("motion_features")

if motion_data is None:
    resized_clips = list_clips(RESIZED_DIR)
    if MAX_CLIPS:
        resized_clips = resized_clips[:MAX_CLIPS]
    records = []
    skipped = 0
    for fpath, label in tqdm(resized_clips, desc="Motion"):
        try:
            diffs = compute_frame_motion(fpath)
            if len(diffs) < 2:
                raise ValueError("Too few frames")
            records.append({"filepath": fpath, "label": label, "diffs": diffs})
        except Exception as e:
            skipped += 1
            print(f"  ⚠ Skipped {os.path.basename(fpath)}: {e}")
    print(f"\n  Processed : {len(records)} clips")
    print(f"  Skipped   : {skipped} clips")
    motion_data = records
    save_cache("motion_features", motion_data)
else:
    print("  ✓ Loaded from cache")

truth_motion     = [r for r in motion_data if r["label"] == "truth"]
deception_motion = [r for r in motion_data if r["label"] == "deception"]

# ── Plot 1: 2x2 Mel Spectrogram Sample Grid ───────────────────────────────────

print("\n━━ Plot 1: 2x2 Spectrogram sample grid ━━")

fig, axes = plt.subplots(2, 2, figsize=(12, 7))
fig.suptitle("Mel Spectrogram Samples — DOLOS Dataset",
             fontsize=15, fontweight="bold", y=1.01)

s_t = rng.choice(len(truth_audio),     size=2, replace=False)
s_d = rng.choice(len(deception_audio), size=2, replace=False)
samples = [
    (truth_audio[s_t[0]],     "truth",      axes[0, 0]),
    (truth_audio[s_t[1]],     "truth",      axes[0, 1]),
    (deception_audio[s_d[0]], "deception",  axes[1, 0]),
    (deception_audio[s_d[1]], "deception",  axes[1, 1]),
]

for record, label, ax in samples:
    librosa.display.specshow(record["mel"], sr=SR, hop_length=HOP_LENGTH,
                             x_axis="time", y_axis="mel", ax=ax, cmap="magma")
    fname = os.path.basename(record["filepath"])
    ax.set_title(f"{LABEL_DISPLAY[label]}\n{fname[:40]}",
                 color=PALETTE[label], fontsize=9, fontweight="bold")
    ax.set_xlabel("Time (s)", fontsize=8)
    ax.set_ylabel("Mel freq", fontsize=8)

plt.tight_layout()
out1 = os.path.join(PLOTS_DIR, "1_spectrogram_sample_grid.png")
plt.savefig(out1, dpi=150, bbox_inches="tight")
plt.close()
print(f"  ✓ Saved → {out1}")

# ── Plot 2: Average Mel Spectrogram per Class ─────────────────────────────────

print("\n━━ Plot 2: Average mel spectrogram per class ━━")

def pad_or_trim_mels(mel_list, target_frames):
    out = []
    for m in mel_list:
        if m.shape[1] >= target_frames:
            out.append(m[:, :target_frames])
        else:
            pad = np.full((m.shape[0], target_frames - m.shape[1]), m.min())
            out.append(np.hstack([m, pad]))
    return np.stack(out)

all_mels  = [r["mel"] for r in audio_data]
target    = int(np.median([m.shape[1] for m in all_mels]))
t_stack   = pad_or_trim_mels([r["mel"] for r in truth_audio],     target)
d_stack   = pad_or_trim_mels([r["mel"] for r in deception_audio], target)
avg_truth = t_stack.mean(axis=0)
avg_dec   = d_stack.mean(axis=0)
vmin      = min(avg_truth.min(), avg_dec.min())
vmax      = max(avg_truth.max(), avg_dec.max())

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Average Mel Spectrogram — Truth vs Deception",
             fontsize=14, fontweight="bold")

for ax, avg, label in zip(axes, [avg_truth, avg_dec], ["truth", "deception"]):
    img = librosa.display.specshow(avg, sr=SR, hop_length=HOP_LENGTH,
                                   x_axis="time", y_axis="mel", ax=ax,
                                   cmap="magma", vmin=vmin, vmax=vmax)
    ax.set_title(LABEL_DISPLAY[label], color=PALETTE[label],
                 fontsize=13, fontweight="bold")
    ax.set_xlabel("Normalised time (frames)", fontsize=9)
    ax.set_ylabel("Mel frequency", fontsize=9)
    fig.colorbar(img, ax=ax, format="%+2.0f dB")

plt.tight_layout()
out2 = os.path.join(PLOTS_DIR, "2_avg_mel_spectrogram_per_class.png")
plt.savefig(out2, dpi=150, bbox_inches="tight")
plt.close()
print(f"  ✓ Saved → {out2}")

# ── Plot 3: Pitch (F0) Distribution by Class ──────────────────────────────────

print("\n━━ Plot 3: Pitch (F0) distribution by class ━━")

f0_rows = [{"Label": LABEL_DISPLAY[r["label"]], "Mean F0 (Hz)": r["f0_mean"]}
           for r in audio_data if r["f0_mean"] > 0]
f0_df = pd.DataFrame(f0_rows)

fig, ax = plt.subplots(figsize=(8, 6))
sns.violinplot(data=f0_df, x="Label", y="Mean F0 (Hz)",
               palette={"Truth": PALETTE["truth"], "Deception": PALETTE["deception"]},
               inner="box", cut=0, ax=ax)
ax.set_title("Pitch (F0) Distribution by Class", fontsize=14, fontweight="bold")
ax.set_xlabel("")
ax.set_ylabel("Mean Fundamental Frequency (Hz)", fontsize=11)

for i, lbl in enumerate(["Truth", "Deception"]):
    subset = f0_df[f0_df["Label"] == lbl]["Mean F0 (Hz)"]
    if len(subset) > 0:
        med = subset.median()
        ax.text(i, med + 5, f"Md={med:.0f}Hz", ha="center",
                fontsize=9, color="white", fontweight="bold")

sns.despine()
plt.tight_layout()
out3 = os.path.join(PLOTS_DIR, "3_f0_distribution_by_class.png")
plt.savefig(out3, dpi=150, bbox_inches="tight")
plt.close()
print(f"  ✓ Saved → {out3}")

# ── Plot 4: Average Frame-Motion Curve by Class ───────────────────────────────

print("\n━━ Plot 4: Average frame-motion curve by class ━━")

N_POINTS = 100

def mean_motion_curve(motion_records):
    curves = []
    for r in motion_records:
        d = r["diffs"]
        if len(d) > 1:
            curves.append(interpolate_to_length(d, N_POINTS))
    return np.array(curves)

t_curves = mean_motion_curve(truth_motion)
d_curves = mean_motion_curve(deception_motion)
x = np.linspace(0, 100, N_POINTS)

fig, ax = plt.subplots(figsize=(11, 5))
for curves, label in [(t_curves, "truth"), (d_curves, "deception")]:
    if len(curves) == 0:
        print(f"  ⚠ No motion data for {label}, skipping")
        continue
    mean = curves.mean(axis=0)
    std  = curves.std(axis=0)
    ax.plot(x, mean, color=PALETTE[label], label=LABEL_DISPLAY[label], linewidth=2.5)
    ax.fill_between(x, mean - std, mean + std, color=PALETTE[label], alpha=0.18)

ax.set_title("Average Facial Motion Over Clip Timeline — Truth vs Deception",
             fontsize=13, fontweight="bold")
ax.set_xlabel("Clip progress (%)", fontsize=11)
ax.set_ylabel("Mean frame difference (pixel intensity)", fontsize=11)
ax.legend(fontsize=11)
sns.despine()
plt.tight_layout()
out4 = os.path.join(PLOTS_DIR, "4_avg_frame_motion_by_class.png")
plt.savefig(out4, dpi=150, bbox_inches="tight")
plt.close()
print(f"  ✓ Saved → {out4}")

# ── Done ──────────────────────────────────────────────────────────────────────

print(f"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
✓ All plots saved to: {PLOTS_DIR}/
  1_spectrogram_sample_grid.png
  2_avg_mel_spectrogram_per_class.png
  3_f0_distribution_by_class.png
  4_avg_frame_motion_by_class.png

Cache stored in: {CACHE_DIR}/
  audio_features.pkl
  motion_features.pkl

Re-running skips computation and loads from cache.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
""")