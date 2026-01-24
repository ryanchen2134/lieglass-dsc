# --- --- --- --- ---
# Imports
# --- --- --- --- ---

from ultralytics import YOLO
import supervision as sv
import os
import sys

import cv2

from tkinter import filedialog

# --- --- --- --- ---
# Initializing model / videos
# --- --- --- --- ---

os.chdir(os.path.dirname(os.path.abspath(__file__)))
model = YOLO('yolov8n-face.pt')     # Pretrained YOLOv8 face model
outputDirectory = os.getcwd() + "/resizedVideos/"

if not os.path.exists("resizedVideos"):
    os.mkdir("resizedVideos")

bigFolder = filedialog.askdirectory()
print(bigFolder)
os.chdir(bigFolder)

filesToResize = os.listdir()

# --- --- --- --- ---
# Loop for Each Video
# --- --- --- --- ---
count = 0
for video in filesToResize:
    count += 1
    print(str(count) + " | " + video)
    # Open input video
    cap = cv2.VideoCapture(video)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Define output video writer (cropped face size)
    output_path = outputDirectory + "[R] " + video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (100, 100))  # Adjust size as needed

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Run YOLO inference
        results = model.predict(source=frame, conf=0.5, save=False, verbose=False)

        # Get bounding box
        for box in results[0].boxes:
            x1, y1, x2, y2 = box.xyxy[0].int().tolist()
            face = frame[(y1 - 0):(y2 + 0), (x1 - 0):(x2 + 0)]  # Crop face region
            face_resized = cv2.resize(face, (100, 100))  # Resize to fixed size
            out.write(face_resized)

        # Optional: Display frame
        #cv2.imshow("Face Detection", frame)
        #if cv2.waitKey(1) & 0xFF == ord('q'):
        #    break

    cap.release()
    out.release()
    cv2.destroyAllWindows()   