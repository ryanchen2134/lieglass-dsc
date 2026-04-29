import ttkbootstrap as ttk
import tkinter as tk
from PIL import Image, ImageTk # You'll need: pip install pillow
import os
import queue
import threading
import time
import numpy as np
from dotenv import load_dotenv

import xreal_capture as xreal
import transcribe
from inference import LocalInferenceProcessor

# --- INITIALIZATION ---
os.chdir(os.path.dirname(os.path.abspath(__file__)))
load_dotenv()

llm_queue = queue.Queue()
truth_score_queue = queue.Queue()
face_queue = queue.Queue(maxsize=1) # Queue for the face preview

# --- START SUBSYSTEMS ---
inference_engine = LocalInferenceProcessor("Deploy/fold_1_best.pt", truth_score_queue)
inference_engine.start()

# Pass the face_queue to xreal.main so it can send the crops back
capture_thread = threading.Thread(
    target=xreal.main, 
    args=(inference_engine, face_queue), 
    daemon=True
)
capture_thread.start()

audio_engine = transcribe.XRealAudioProcessor(api_key=os.getenv("ANTHROPIC_API_KEY"), results_queue=llm_queue)
audio_engine.start()

# --- GLOBALS ---
current_llm_grade = 0
current_av_score = 0.5

def clamp(val, bounds):
    return max(min(val, bounds[1]), bounds[0])

def get_score_color(val):
    r = clamp(int(2 * (1.0 - val) * 255), (0, 255))
    g = clamp(int(2 * val * 255), (0, 255))
    return f"#{r:02x}{g:02x}00"

# --- WINDOW SETUP ---
root = ttk.Window(themename='darkly')
root.geometry('1920x1080')
root.state("zoomed")
root.overrideredirect(True)
root.configure(background='black') # Forces total background black

# --- STYLE ---
style = ttk.Style()
style.configure('TFrame', background='black')
style.configure('TLabel', background='black')

# 1. TOP CONTAINER (Percentage | Grade | Face | T/F)
# 1. TOP CONTAINER (Percentage | Grade | T/F | Face)
top_frame = ttk.Frame(root)
top_frame.pack(fill="x", padx=40, pady=(20, 10))

# Far Right: Face Window (Packed first to take the corner)
face_label = ttk.Label(top_frame)
face_label.pack(side="right", padx=(20, 0))

# Right-ish: T/F Grade
truth_grade = ttk.Label(top_frame, text="WAIT", font=("Source Code Pro", 130), foreground="#5599FF")
truth_grade.pack(side="right", padx=(20, 40)) # Extra padding between text and face

# Left: AV Percentage
truth_score = ttk.Label(top_frame, text="--%", font=("Source Code Pro", 130), foreground="white")
truth_score.pack(side="left")

# Center: LLM Grade
llm_grade = ttk.Label(top_frame, text="0", font=("Source Code Pro", 130), foreground="white")
llm_grade.pack(side="left", expand=True)

# 2. THE LIGHT GRAY BAR
separator = ttk.Frame(root, height=4, style='secondary.TFrame') # Uses bootstrap secondary (gray)
separator.pack(fill="x", padx=40, pady=5)

# 3. FEEDBACK AREA
feedback_frame = ttk.Frame(root)
feedback_frame.pack(fill=tk.BOTH, expand=True, padx=40)

feedback = ttk.Label(feedback_frame, text="Starting...", 
                     font=("Source Code Pro", 60), anchor="nw", justify="left", 
                     foreground="#00FF00", wraplength=1700)
feedback.pack(fill="both", pady=20)

# --- REFRESH LOGIC ---

def update_ui():
    global current_llm_grade, current_av_score
    
    # 1. Update Face Window (Increased size to match text)
    try:
        face_img_np = face_queue.get_nowait()
        img = Image.fromarray(face_img_np)
        # 260px is closer to the visual height of 130pt text on 1080p
        img = img.resize((260, 260), Image.Resampling.LANCZOS)
        img_tk = ImageTk.PhotoImage(image=img)
        face_label.configure(image=img_tk)
        face_label.image = img_tk 
    except queue.Empty:
        pass

    # 2. Poll AV Model
    try:
        while True:
            data = truth_score_queue.get_nowait()
            current_av_score = data['score']
            color = get_score_color(current_av_score)
            truth_score.configure(text=f"{int(current_av_score * 100):02d}%", foreground=color)
    except queue.Empty:
        pass

    # 3. Poll LLM & Update Grade
    try:
        data = llm_queue.get_nowait()
        inconsistencies = data.get("inconsistencies", [])
        
        if inconsistencies:
            latest = inconsistencies[-1]
            severity = latest.get('severity', 'MINOR').upper()
            
            # --- ARITHMETIC REPLACEMENT ---
            if "MAJOR" in severity:
                current_llm_grade -= 3
                feedback.configure(foreground="#ff0000")
            elif "MINOR" in severity:
                current_llm_grade -= 1
                feedback.configure(foreground="#ffaa00")
            elif "TRUTH" in severity:
                current_llm_grade += 2
                feedback.configure(foreground="#00ff00")
            
            # Update Score Label with Color
            grade_color = get_score_color(clamp((current_llm_grade + 10) / 20, (0, 1)))
            llm_grade.configure(text=str(current_llm_grade), foreground=grade_color)
            
            # Update Feedback Text
            feedback.configure(text=f"[{severity}] {latest.get('description')}\n\nSUGGESTION: {data.get('suggested_question')}")
            
        else:
            feedback.configure(text=f"[SUGGESTION] {data.get('suggested_question')}", foreground="#007DFF")
            
    except queue.Empty:
        pass

    # 4. Overall Evaluation
    llm_normalized = clamp((current_llm_grade + 10) / 20, (0, 1))
    combined_prob = (current_av_score * 0.6) + (llm_normalized * 0.4)
    
    if combined_prob > 0.5:
        truth_grade.configure(text="TRUE", foreground="#00FF00")
    else:
        truth_grade.configure(text="FALSE", foreground="#FF0000")

    root.after(33, update_ui)

root.bind("<Configure>", lambda e: feedback.configure(wraplength=root.winfo_width() - 80))
root.after(1000, update_ui)
root.mainloop()