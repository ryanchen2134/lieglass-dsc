# --- --- --- --- ---
# Imports
# --- --- --- --- ---
from ultralytics import YOLO
import supervision as sv
import numpy as np
import os
import cv2
import sys
from collections import defaultdict  # NEW IMPORT
from tkinter import filedialog
from moviepy import VideoFileClip  # MoviePy v2.0+

# --- --- --- --- ---
# Initializing model / directories
# --- --- --- --- ---
os.chdir(os.path.dirname(os.path.abspath(__file__)))
model = YOLO('yolov8n-face.pt')
outputDirectory = os.getcwd() + "/resizedVideosNew/"

if not os.path.exists("resizedVideosNew"):
    os.mkdir("resizedVideosNew")

bigFolder = filedialog.askdirectory()
if not bigFolder:
    sys.exit()
    
print(bigFolder)
os.chdir(bigFolder)
filesToResize = os.listdir()

# --- --- --- --- ---
# Loop for Each Video
# --- --- --- --- ---
count = 0
CROP_SIZE = (100, 100) 

for video in filesToResize:
    #if vid exists in output dir, skip
    output_path = os.path.join(outputDirectory, "[R] " + video)
    if os.path.exists(output_path):
        print(f"\nSkipping {video}, already processed.")
        continue
    if not video.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
        continue

    count += 1
    print(f"\n{count} | Processing: {video}")
    
    # --- PHASE 1: ANALYZE (Find the Best Face) ---
    print("  > Phase 1: Analyzing frames to find the main character...")
    tracker = sv.ByteTrack()
    id_counts = defaultdict(int)
    frame_detections = []  # Store detections here to avoid running YOLO twice

    cap = cv2.VideoCapture(video)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Run Model & Tracker
        results = model.predict(source=frame, conf=0.5, save=False, verbose=False)[0]
        detections = sv.Detections.from_ultralytics(results)
        detections = tracker.update_with_detections(detections)
        
        # Store detections for Phase 2
        frame_detections.append(detections)

        # Count occurrences of each ID
        if detections.tracker_id is not None:
            for track_id in detections.tracker_id:
                id_counts[track_id] += 1

    cap.release()

    # Determine the "Main Character" (ID with max frames)
    if not id_counts:
        print("  > No faces found. Skipping.")
        continue

    target_id = max(id_counts, key=id_counts.get)
    print(f"  > Target Locked: ID {target_id} (Appears in {id_counts[target_id]}/{total_frames} frames)")

    # --- PHASE 2: WRITE (Crop & Save) ---
    print("  > Phase 2: Writing video...")
    
    # Re-open video for reading
    cap = cv2.VideoCapture(video)
    
    temp_output_filename = "temp_" + video
    temp_output_path = os.path.join(outputDirectory, temp_output_filename)
    final_output_path = os.path.join(outputDirectory, "[R] " + video)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(temp_output_path, fourcc, fps, CROP_SIZE)

    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Retrieve the specific detection for this frame from memory
        # (We use the list index because frames correspond 1-to-1)
        if frame_idx < len(frame_detections):
            detections = frame_detections[frame_idx]
        else:
            detections = [] # Safety fallback

        face_img = None

        # Look for our pre-calculated Target ID
        if detections.tracker_id is not None and target_id in detections.tracker_id:
            # Find the index of the target_id in this frame's detections
            # (tracker_id is a numpy array, so we find where it equals target_id)
            match_idx = list(detections.tracker_id).index(target_id)
            
            # Extract coordinates using that index
            x1, y1, x2, y2 = detections.xyxy[match_idx].astype(int)
            
            # Clamp and Crop
            h, w, _ = frame.shape
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)

            if x2 > x1 and y2 > y1:
                face = frame[y1:y2, x1:x2]
                try:
                    face_img = cv2.resize(face, CROP_SIZE)
                except Exception:
                    face_img = None
        
        # Write Frame
        if face_img is not None:
            out.write(face_img)
        else:
            black_frame = np.zeros((CROP_SIZE[1], CROP_SIZE[0], 3), dtype=np.uint8)
            out.write(black_frame)
            
        frame_idx += 1

    cap.release()
    out.release()
    cv2.destroyAllWindows()

    # --- PHASE 3: AUDIO MUXING ---
    try:
        processed_clip = VideoFileClip(temp_output_path)
        original_clip = VideoFileClip(video)
        
        if original_clip.audio:
            # Using v2.0+ syntax 'with_audio'
            final_clip = processed_clip.with_audio(original_clip.audio)
            final_clip.write_videofile(final_output_path, codec="libx264", audio_codec="aac", logger=None)
        else:
            processed_clip.write_videofile(final_output_path, codec="libx264", logger=None)

        processed_clip.close()
        original_clip.close()
        
        if os.path.exists(temp_output_path):
            os.remove(temp_output_path)

    except Exception as e:
        print(f"  > Audio Error: {e}")