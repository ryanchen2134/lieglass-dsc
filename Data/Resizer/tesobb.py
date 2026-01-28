# --- --- --- --- ---
# Imports
# --- --- --- --- ---
from ultralytics import YOLO
import supervision as sv
import numpy as np
import os
import cv2
import sys
from collections import defaultdict
from tkinter import filedialog
from moviepy import VideoFileClip  # MoviePy v2.0+

# --- --- --- --- ---
# Initializing model / directories
# --- --- --- --- ---
os.chdir(os.path.dirname(os.path.abspath(__file__)))
model = YOLO('yolov8n-face.pt')
outputDirectory = os.getcwd() + "/resizedVideosNewRot/"

if not os.path.exists("resizedVideosNewRot"):
    os.mkdir("resizedVideosNewRot")

bigFolder = filedialog.askdirectory()
if not bigFolder:
    sys.exit()
    
print(bigFolder)
os.chdir(bigFolder)
filesToResize = os.listdir()

# --- --- --- --- ---
# Helper: Rotation Function
# --- --- --- --- ---
def crop_rotated_face(frame, box_xyxy, keypoints, crop_size=(100, 100)):
    """
    Rotates the frame to make the eyes horizontal, then crops the face.
    """
    # 1. Calculate Center of Face (using BBox)
    x1, y1, x2, y2 = box_xyxy
    cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
    
    # 2. Calculate Angle from Eyes (Idx 0: Left Eye, Idx 1: Right Eye)
    # Note: yolov8-face keypoints are usually [Left Eye, Right Eye, Nose, Left Mouth, Right Mouth]
    if keypoints is not None and len(keypoints) >= 2:
        left_eye = keypoints[0]
        right_eye = keypoints[1]
        
        dx = right_eye[0] - left_eye[0]
        dy = right_eye[1] - left_eye[1]
        
        # Calculate angle in degrees
        angle = np.degrees(np.arctan2(dy, dx))
    else:
        angle = 0  # Fallback if no keypoints found

    # 3. Get Rotation Matrix (Rotate around face center)
    M = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
    
    # 4. Rotate the entire frame (or a large ROI for speed)
    h, w = frame.shape[:2]
    rotated_frame = cv2.warpAffine(frame, M, (w, h))
    
    # 5. Crop Fixed Size from Rotated Frame
    # We want a square crop centered at (cx, cy)
    half_w = crop_size[0] // 2
    half_h = crop_size[1] // 2
    
    # Calculate crop coordinates
    start_x = cx - half_w
    start_y = cy - half_h
    end_x = start_x + crop_size[0]
    end_y = start_y + crop_size[1]
    
    # Handle Boundary Checks (Pad if crop goes outside image)
    if start_x < 0 or start_y < 0 or end_x > w or end_y > h:
        # Complex padding logic omitted for brevity; using resizing fallback
        # If rotation pushes crop out of bounds, fall back to simple resize of bbox
        face = frame[int(y1):int(y2), int(x1):int(x2)]
        return cv2.resize(face, crop_size)
    
    final_crop = rotated_frame[start_y:end_y, start_x:end_x]
    
    # Ensure exact size output
    if final_crop.shape[0] != crop_size[1] or final_crop.shape[1] != crop_size[0]:
        final_crop = cv2.resize(final_crop, crop_size)
        
    return final_crop

# --- --- --- --- ---
# Loop for Each Video
# --- --- --- --- ---
count = 0
CROP_SIZE = (100, 100) 

for video in filesToResize:
    if not video.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
        continue

    count += 1
    print(f"\n{count} | Processing: {video}")
    
    # --- PHASE 1: ANALYZE ---
    print("  > Phase 1: Analyzing frames to find the main character...")
    tracker = sv.ByteTrack()
    id_counts = defaultdict(int)
    frame_detections = [] 

    cap = cv2.VideoCapture(video)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Run Model
        results = model.predict(source=frame, conf=0.5, save=False, verbose=False)[0]
        
        # Convert to Supervision Detections
        # Crucial: Ensure we capture keypoints if they exist
        detections = sv.Detections.from_ultralytics(results)
        
        # Run Tracker
        detections = tracker.update_with_detections(detections)
        
        # Store detections (now containing IDs and hopefully Keypoints)
        frame_detections.append(detections)

        if detections.tracker_id is not None:
            for track_id in detections.tracker_id:
                id_counts[track_id] += 1

    cap.release()

    if not id_counts:
        print("  > No faces found. Skipping.")
        continue

    target_id = max(id_counts, key=id_counts.get)
    print(f"  > Target Locked: ID {target_id}")

    # --- PHASE 2: WRITE ---
    print("  > Phase 2: Writing video with OBB rotation...")
    
    cap = cv2.VideoCapture(video)
    temp_output_path = os.path.join(outputDirectory, "temp_" + video)
    final_output_path = os.path.join(outputDirectory, "[R] " + video)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(temp_output_path, fourcc, fps, CROP_SIZE)

    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_idx < len(frame_detections):
            detections = frame_detections[frame_idx]
        else:
            detections = [] 

        face_img = None

        if detections.tracker_id is not None and target_id in detections.tracker_id:
            match_idx = list(detections.tracker_id).index(target_id)
            
            # 1. Get Bounding Box
            x1, y1, x2, y2 = detections.xyxy[match_idx].astype(int)
            
            # 2. Get Keypoints (if available)
            kpts = None
            # Check if keypoints exist in the detections object
            if hasattr(detections, 'keypoints') and detections.keypoints is not None:
                # Keypoints format is usually (N, K, 2)
                # We need the keypoints for the specific matched face
                raw_kpts = detections.keypoints.xy[match_idx]
                if len(raw_kpts) > 0:
                    kpts = raw_kpts
            
            # 3. Perform OBB Crop (Rotate & Crop)
            try:
                face_img = crop_rotated_face(frame, (x1, y1, x2, y2), kpts, CROP_SIZE)
            except Exception as e:
                # Fallback to simple crop if rotation fails
                face = frame[max(0, y1):min(frame.shape[0], y2), max(0, x1):min(frame.shape[1], x2)]
                face_img = cv2.resize(face, CROP_SIZE)
        
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