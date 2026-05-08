import argparse
import math
import os
import queue
import threading
import time
from collections import deque
from pathlib import Path

import numpy as np
import ttkbootstrap as ttk
import tkinter as tk
from PIL import Image, ImageTk  # pip install pillow
from dotenv import load_dotenv

import matplotlib
matplotlib.use("TkAgg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

import xreal_capture as xreal
import transcribe
from inference import LocalInferenceProcessor

# ---------------------------------------------------------------------------
# INITIALIZATION
# ---------------------------------------------------------------------------
os.chdir(os.path.dirname(os.path.abspath(__file__)))
load_dotenv()

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CHECKPOINTS_DIR = REPO_ROOT / "checkpoints"


# ---------------------------------------------------------------------------
# Tunables — adjust freely; sensible defaults below.
# ---------------------------------------------------------------------------

# Decision threshold on the FUSED probability of "truthful" (sigmoid space).
# 0.5 is the naive cut; tune off-line on validation (e.g. Youden's J).
DECISION_THRESHOLD = 0.5

# Log-odds (logit-space) fusion weights. The combined logit is
#   z = AV_FUSION_WEIGHT * av_logit + LLM_FUSION_WEIGHT * llm_logit + FUSION_BIAS
# This is more principled than averaging probabilities and lets each modality
# vote with an unbounded magnitude.
AV_FUSION_WEIGHT = 1.0
LLM_FUSION_WEIGHT = 0.6
FUSION_BIAS = 0.0

# Map the running LLM grade to a logit. Grade += 2 on TRUTH, -=1 on MINOR,
# -=3 on MAJOR. Grade=10 -> logit≈+2 (≈88% truth), grade=-10 -> logit≈-2.
LLM_GRADE_TO_LOGIT_SCALE = 0.2

# --- Live HUD probability graph -------------------------------------------
GRAPH_HISTORY_POINTS = 240             # ~1 minute @ 4 Hz emission
GRAPH_REDRAW_INTERVAL_MS = 200         # 5 Hz redraw
GRAPH_FIG_HEIGHT = 2.0                 # inches
GRAPH_DPI = 80
GRAPH_LINE_COLOR = "#00FF88"
GRAPH_CI_COLOR = "#5599FF"             # faint upper/lower CI lines
GRAPH_THRESHOLD_COLOR = "#666666"

# UI refresh tick (independent of inference cadence and graph redraw).
UI_TICK_MS = 33


# ---------------------------------------------------------------------------
# Checkpoint resolution
# ---------------------------------------------------------------------------


def _resolve_checkpoint(run: str | None, fold: int | None,
                        checkpoint: str | None,
                        checkpoints_dir: Path) -> tuple[Path, Path | None]:
    """Resolve (checkpoint_pt, config_json) from CLI args.

    Precedence:
      1. ``--checkpoint`` (explicit .pt path)
      2. ``--run`` (folder name under checkpoints/) + optional ``--fold``
      3. Latest run under ``checkpoints/`` (sorted name desc) + fold 0
    """
    if checkpoint:
        ckpt = Path(checkpoint).expanduser().resolve()
        if not ckpt.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt}")
        cfg = ckpt.parent / "config.json"
        return ckpt, cfg if cfg.is_file() else None

    if run:
        run_dir = Path(run).expanduser()
        if not run_dir.is_absolute():
            run_dir = (checkpoints_dir / run_dir).resolve()
    else:
        if not checkpoints_dir.is_dir():
            raise FileNotFoundError(f"No checkpoints directory: {checkpoints_dir}")
        candidates = sorted([p for p in checkpoints_dir.iterdir() if p.is_dir()])
        if not candidates:
            raise FileNotFoundError(f"No runs found under {checkpoints_dir}")
        run_dir = candidates[-1]
        print(f"[gui] Auto-selected latest run: {run_dir.name}")

    if not run_dir.is_dir():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")

    fold_idx = 0 if fold is None else int(fold)
    ckpt = run_dir / f"fold_{fold_idx}_best.pt"
    if not ckpt.is_file():
        last = run_dir / "last.pt"
        if last.is_file():
            ckpt = last
        else:
            folds = sorted(run_dir.glob("fold_*_best.pt"))
            if not folds:
                raise FileNotFoundError(f"No checkpoint .pt found in {run_dir}")
            ckpt = folds[0]
    cfg = run_dir / "config.json"
    return ckpt, cfg if cfg.is_file() else None


parser = argparse.ArgumentParser(description="LieGlass GUI")
parser.add_argument("--run", type=str, default=None,
                    help="Run name under checkpoints/ (e.g. 20260429_055701) or absolute path.")
parser.add_argument("--fold", type=int, default=None,
                    help="Fold index to load (default: 0). Falls back to last.pt.")
parser.add_argument("--checkpoint", type=str, default=None,
                    help="Explicit path to a .pt file (overrides --run/--fold).")
parser.add_argument("--checkpoints-dir", type=str, default=str(DEFAULT_CHECKPOINTS_DIR),
                    help=f"Directory containing run folders (default: {DEFAULT_CHECKPOINTS_DIR}).")
args, _ = parser.parse_known_args()

ckpt_path, cfg_path = _resolve_checkpoint(
    run=args.run,
    fold=args.fold,
    checkpoint=args.checkpoint,
    checkpoints_dir=Path(args.checkpoints_dir).expanduser().resolve(),
)
print(f"[gui] Using checkpoint: {ckpt_path}")
print(f"[gui] Using config:     {cfg_path if cfg_path else '(default ModelConfig)'}")

llm_queue = queue.Queue()
truth_score_queue = queue.Queue()
face_queue = queue.Queue(maxsize=1)

# ---------------------------------------------------------------------------
# Subsystems
# ---------------------------------------------------------------------------
inference_engine = LocalInferenceProcessor(
    checkpoint_path=ckpt_path,
    result_queue=truth_score_queue,
    config_path=cfg_path,
)
inference_engine.start()

capture_thread = threading.Thread(
    target=xreal.main,
    args=(inference_engine, face_queue),
    daemon=True,
)
capture_thread.start()

audio_engine = transcribe.XRealAudioProcessor(
    api_key=os.getenv("ANTHROPIC_API_KEY"), results_queue=llm_queue,
)
audio_engine.start()


# ---------------------------------------------------------------------------
# Globals
# ---------------------------------------------------------------------------
current_llm_grade = 0
current_av_score = 0.5
current_av_logit = 0.0
current_av_lo = 0.5
current_av_hi = 0.5

score_history: deque[float] = deque(maxlen=GRAPH_HISTORY_POINTS)
score_lo_history: deque[float] = deque(maxlen=GRAPH_HISTORY_POINTS)
score_hi_history: deque[float] = deque(maxlen=GRAPH_HISTORY_POINTS)
_last_graph_draw_ms = 0


def clamp(val, bounds):
    return max(min(val, bounds[1]), bounds[0])


def get_score_color(val):
    r = clamp(int(2 * (1.0 - val) * 255), (0, 255))
    g = clamp(int(2 * val * 255), (0, 255))
    return f"#{r:02x}{g:02x}00"


def _sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


# ---------------------------------------------------------------------------
# Window setup
# ---------------------------------------------------------------------------
root = ttk.Window(themename='darkly')
root.geometry('1920x1080')
root.state("zoomed")
root.overrideredirect(True)
root.configure(background='black')

style = ttk.Style()
style.configure('TFrame', background='black')
style.configure('TLabel', background='black')

# 1. TOP CONTAINER (Percentage(+CI) | LLM grade | T/F | Face)
top_frame = ttk.Frame(root)
top_frame.pack(fill="x", padx=40, pady=(20, 10))

# Far right: face crop preview
face_label = ttk.Label(top_frame)
face_label.pack(side="right", padx=(20, 0))

# Right-ish: TRUE/FALSE
truth_grade = ttk.Label(top_frame, text="WAIT", font=("Source Code Pro", 130), foreground="#5599FF")
truth_grade.pack(side="right", padx=(20, 40))

# Left: AV percentage with stacked CI label.
score_frame = ttk.Frame(top_frame)
score_frame.pack(side="left")
truth_score = ttk.Label(score_frame, text="--%", font=("Source Code Pro", 130), foreground="white")
truth_score.pack(anchor="w")
truth_score_ci = ttk.Label(score_frame, text="±--%", font=("Source Code Pro", 28), foreground="#888888")
truth_score_ci.pack(anchor="w")

# Center: LLM grade
llm_grade = ttk.Label(top_frame, text="0", font=("Source Code Pro", 130), foreground="white")
llm_grade.pack(side="left", expand=True)

# 2. Light gray separator
separator = ttk.Frame(root, height=4, style='secondary.TFrame')
separator.pack(fill="x", padx=40, pady=5)

# 3. Live probability graph
graph_frame = ttk.Frame(root)
graph_frame.pack(fill="x", padx=40, pady=(5, 5))

_fig = Figure(figsize=(18, GRAPH_FIG_HEIGHT), dpi=GRAPH_DPI, facecolor="black")
_ax = _fig.add_subplot(111)
_ax.set_facecolor("black")
_ax.set_ylim(0.0, 1.0)
_ax.set_xlim(0, GRAPH_HISTORY_POINTS - 1)
_ax.tick_params(colors="white", labelsize=8)
for _spine in _ax.spines.values():
    _spine.set_color("white")
_ax.grid(True, color="#222222", linewidth=0.5)

# Persistent line artists (updated in-place each redraw).
(_ci_lo_line,) = _ax.plot([], [], color=GRAPH_CI_COLOR, linewidth=1.0, alpha=0.45)
(_ci_hi_line,) = _ax.plot([], [], color=GRAPH_CI_COLOR, linewidth=1.0, alpha=0.45)
(_score_line,) = _ax.plot([], [], color=GRAPH_LINE_COLOR, linewidth=2.0)
_ax.axhline(DECISION_THRESHOLD, color=GRAPH_THRESHOLD_COLOR, linestyle="--", linewidth=1.0)
_fig.subplots_adjust(left=0.04, right=0.995, top=0.95, bottom=0.18)

_canvas = FigureCanvasTkAgg(_fig, master=graph_frame)
_canvas.get_tk_widget().configure(background="black", highlightthickness=0)
_canvas.get_tk_widget().pack(fill="x")
_canvas.draw()

# 4. Feedback
feedback_frame = ttk.Frame(root)
feedback_frame.pack(fill=tk.BOTH, expand=True, padx=40)

feedback = ttk.Label(
    feedback_frame, text="Starting...",
    font=("Source Code Pro", 50), anchor="nw", justify="left",
    foreground="#00FF00", wraplength=1700,
)
feedback.pack(fill="both", pady=20)


# ---------------------------------------------------------------------------
# Refresh logic
# ---------------------------------------------------------------------------


def _redraw_graph():
    n = len(score_history)
    if n == 0:
        return
    xs = np.arange(n)
    _score_line.set_data(xs, np.fromiter(score_history, dtype=float, count=n))
    _ci_lo_line.set_data(xs, np.fromiter(score_lo_history, dtype=float, count=n))
    _ci_hi_line.set_data(xs, np.fromiter(score_hi_history, dtype=float, count=n))
    _ax.set_xlim(0, max(GRAPH_HISTORY_POINTS - 1, n - 1))
    _canvas.draw_idle()


def update_ui():
    global current_llm_grade, current_av_score, current_av_logit
    global current_av_lo, current_av_hi, _last_graph_draw_ms

    # 1. Face crop preview
    try:
        face_img_np = face_queue.get_nowait()
        img = Image.fromarray(face_img_np)
        img = img.resize((260, 260), Image.Resampling.LANCZOS)
        img_tk = ImageTk.PhotoImage(image=img)
        face_label.configure(image=img_tk)
        face_label.image = img_tk
    except queue.Empty:
        pass

    # 2. Drain AV inference outputs (point estimate + CI in p-space).
    drained_any = False
    try:
        while True:
            data = truth_score_queue.get_nowait()
            current_av_score = float(data["score"])
            current_av_logit = float(data.get("logit", 0.0))
            current_av_lo = float(data.get("score_lo", current_av_score))
            current_av_hi = float(data.get("score_hi", current_av_score))
            score_history.append(current_av_score)
            score_lo_history.append(current_av_lo)
            score_hi_history.append(current_av_hi)
            drained_any = True
    except queue.Empty:
        pass

    if drained_any:
        color = get_score_color(current_av_score)
        truth_score.configure(text=f"{int(current_av_score * 100):02d}%", foreground=color)
        half_pct = max(0, int(round((current_av_hi - current_av_lo) * 100 / 2)))
        truth_score_ci.configure(text=f"±{half_pct}%")

    # Graph redraw at fixed cadence (decoupled from UI tick).
    now_ms = int(time.monotonic() * 1000)
    if now_ms - _last_graph_draw_ms >= GRAPH_REDRAW_INTERVAL_MS and len(score_history) > 0:
        _redraw_graph()
        _last_graph_draw_ms = now_ms

    # 3. Poll LLM
    try:
        data = llm_queue.get_nowait()
        inconsistencies = data.get("inconsistencies", [])

        if inconsistencies:
            latest = inconsistencies[-1]
            severity = latest.get('severity', 'MINOR').upper()

            if "MAJOR" in severity:
                current_llm_grade -= 3
                feedback.configure(foreground="#ff0000")
            elif "MINOR" in severity:
                current_llm_grade -= 1
                feedback.configure(foreground="#ffaa00")
            elif "TRUTH" in severity:
                current_llm_grade += 2
                feedback.configure(foreground="#00ff00")

            grade_color = get_score_color(clamp((current_llm_grade + 10) / 20, (0, 1)))
            llm_grade.configure(text=str(current_llm_grade), foreground=grade_color)

            feedback.configure(
                text=f"[{severity}] {latest.get('description')}\n\nSUGGESTION: {data.get('suggested_question')}"
            )
        else:
            feedback.configure(
                text=f"[SUGGESTION] {data.get('suggested_question')}", foreground="#007DFF",
            )
    except queue.Empty:
        pass

    # 4. Log-odds fusion of AV + LLM, then decision threshold.
    llm_logit = current_llm_grade * LLM_GRADE_TO_LOGIT_SCALE
    combined_logit = (
        AV_FUSION_WEIGHT * current_av_logit
        + LLM_FUSION_WEIGHT * llm_logit
        + FUSION_BIAS
    )
    combined_prob = _sigmoid(combined_logit)

    if combined_prob > DECISION_THRESHOLD:
        truth_grade.configure(text="TRUE", foreground="#00FF00")
    else:
        truth_grade.configure(text="FALSE", foreground="#FF0000")

    root.after(UI_TICK_MS, update_ui)


root.bind("<Configure>", lambda e: feedback.configure(wraplength=root.winfo_width() - 80))
root.after(1000, update_ui)
root.mainloop()
