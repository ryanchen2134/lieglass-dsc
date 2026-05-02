#!/usr/bin/env python3
"""
xreal_capture.py - XREAL Eye Camera + Audio Capture for Model Training
-----------------------------------------------------------------------
Captures synchronized video frames and audio from the XREAL One Pro + Eye.

Video:  Grayscale frames from the SLAM camera over USB-C ethernet (port 52997)
Audio:  Beamforming mic array (both channels averaged to mono) via USB audio

Requirements:
    pip install numpy opencv-python sounddevice

Usage:
    python3 xreal_capture.py

Controls:
    Q / ESC  - quit and save all data to disk
    SPACE    - pause/resume capture
    C        - clear all captured data
    S        - save snapshot of current frame

Output structure:
    xreal_video_raw/
        session_001/
            frames.npy          - shape (N, 378, 512) uint8
            timestamps.npy      - shape (N,) float64
    xreal_audio_raw/
        session_001/
            audio.npy           - shape (total_samples,) float32
            audio_start.npy     - shape (1,) float64

    Session number is shared across both folders so session_001 video
    always pairs with session_001 audio.

    To sync frame i to audio:
        idx    = int((timestamps[i] - audio_start[0]) * 48000)
        window = audio[idx : idx + 48000]
"""

import socket
import threading
import queue
import time
import os
import numpy as np
import cv2
import sounddevice as sd
from ultralytics import YOLO

# -- Video connection ---------------------------------------------------------
HOST            = "169.254.2.1"
PORT            = 52997
CONNECT_TIMEOUT = 10
READ_TIMEOUT    = 5

# -- Video frame format -------------------------------------------------------
FRAME_SIZE  = 193862
HEADER_SIZE = 64
ZERO_PAD    = 199
DATA_OFFSET = HEADER_SIZE + ZERO_PAD
MAGIC       = bytes.fromhex("27480002")

CAM_W       = 512
CAM_H       = 378
CAM_PIXELS  = CAM_W * CAM_H

# -- Audio device -------------------------------------------------------------
AUDIO_DEVICE   = 2       # XREAL One input (index 2, 2ch, 48kHz)
AUDIO_CHANNELS = 2       # both beamforming mics
AUDIO_RATE     = 48000   # Hz
AUDIO_BLOCK    = 1024    # samples per callback (~21ms)

# -- Capture settings ---------------------------------------------------------
DISPLAY_SCALE     = 2.0
WINDOW_NAME       = "XREAL FEED"

# -- Model Settings -----------------------------------------------------------
FACE_MODEL_PATH = "yolov8n-face.pt"
TARGET_SIZE = 128  # The 128x128 you requested

# -- Frame alignment ----------------------------------------------------------
ROLL_PCT       = 0.12
CROP_LEFT_PCT  = 0.09
CROP_RIGHT_PCT = 0.13

# -----------------------------------------------------------------------------
# Video decode
# -----------------------------------------------------------------------------

def decode_frame(raw_bytes):
    arr = np.frombuffer(raw_bytes[:CAM_PIXELS], dtype=np.uint8)
    img = arr.reshape(CAM_H, CAM_W).copy()
    shift     = int(CAM_W * ROLL_PCT)
    img       = np.roll(img, -shift, axis=1)
    col_left  = int(CAM_W * CROP_LEFT_PCT)
    col_right = CAM_W - int(CAM_W * CROP_RIGHT_PCT)
    img       = img[:, col_left:col_right]
    img       = cv2.resize(img, (CAM_W, CAM_H), interpolation=cv2.INTER_LINEAR)
    return img

# -----------------------------------------------------------------------------
# Video stream reader (background thread)
# -----------------------------------------------------------------------------

def stream_reader(sock, frame_queue, stop_event):
    buffer      = bytearray()
    frame_count = 0
    dropped     = 0

    while not stop_event.is_set():
        try:
            chunk = sock.recv(65536)
            if not chunk:
                break
            buffer.extend(chunk)

            while len(buffer) >= FRAME_SIZE:
                if buffer[:4] == MAGIC:
                    frame_bytes  = bytes(buffer[:FRAME_SIZE])
                    buffer       = buffer[FRAME_SIZE:]
                    frame_count += 1

                    if frame_count % 2 == 0:
                        continue

                    pixel_data = frame_bytes[DATA_OFFSET:]
                    try:
                        img = decode_frame(pixel_data)
                    except Exception as e:
                        print("[video] Decode error:", e)
                        continue

                    ts = time.time()

                    if frame_queue.full():
                        try:
                            frame_queue.get_nowait()
                            dropped += 1
                        except queue.Empty:
                            pass

                    frame_queue.put((img, frame_count, ts))

                else:
                    next_magic = buffer.find(MAGIC, 1)
                    if next_magic == -1:
                        buffer = buffer[-(len(MAGIC) - 1):]
                    else:
                        buffer = buffer[next_magic:]

        except socket.timeout:
            continue
        except Exception as e:
            print("[video] Error:", e)
            break

    stop_event.set()
    print("[video] Done. Frames: " + str(frame_count) +
          " | Dropped: " + str(dropped))

# -----------------------------------------------------------------------------
# Audio capture (background thread)
# -----------------------------------------------------------------------------

def audio_reader(audio_queue, stop_event):
    def callback(indata, frames, time_info, status):
        if status:
            print("[audio] Status:", status)
        mono = indata.mean(axis=1).astype(np.float32)
        ts   = time.time() - (len(mono) / AUDIO_RATE)
        try:
            audio_queue.put_nowait((mono.copy(), ts))
        except queue.Full:
            pass

    try:
        with sd.InputStream(
            device=AUDIO_DEVICE,
            samplerate=AUDIO_RATE,
            channels=AUDIO_CHANNELS,
            dtype='float32',
            blocksize=AUDIO_BLOCK,
            callback=callback
        ):
            print("[audio] Stream open -- " + str(AUDIO_RATE) +
                  "Hz mono, block=" + str(AUDIO_BLOCK) + " samples")
            while not stop_event.is_set():
                time.sleep(0.05)
    except Exception as e:
        print("[audio] Error:", e)

# -----------------------------------------------------------------------------
# Connection
# -----------------------------------------------------------------------------

def connect_to_glasses():
    print("Connecting to " + HOST + ":" + str(PORT) + " ...")
    while True:
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(CONNECT_TIMEOUT)
            sock.connect((HOST, PORT))
            sock.settimeout(READ_TIMEOUT)
            print("[video] Connected!")
            return sock
        except socket.timeout:
            print("Timed out, retrying...")
            time.sleep(2)
        except ConnectionRefusedError:
            print("Refused -- is Spatial Anchor enabled? Retrying...")
            time.sleep(2)
        except OSError as e:
            print("Network error:", e)
            time.sleep(3)

# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main(inference_engine=None, gui_face_queue=None): # Added parameter
    # 1. Initialize YOLO Face Model
    print(f"Loading model: {FACE_MODEL_PATH}...")
    model = YOLO(FACE_MODEL_PATH)

    # 2. Setup connection and queues
    sock = connect_to_glasses()
    frame_queue = queue.Queue(maxsize=4)
    audio_queue = queue.Queue(maxsize=200)
    stop_event = threading.Event()

    # 3. Start background threads
    threading.Thread(target=stream_reader, args=(sock, frame_queue, stop_event), daemon=True).start()
    threading.Thread(target=audio_reader, args=(audio_queue, stop_event), daemon=True).start()

    last_face = np.zeros((TARGET_SIZE, TARGET_SIZE), dtype=np.uint8)
    
    try:
        while not stop_event.is_set():
            # Handle Audio for Inference
            while not audio_queue.empty():
                try: 
                    audio_chunk, _ = audio_queue.get_nowait()
                    ### ADDED: Feed audio to the model
                    if inference_engine:
                        inference_engine.audio_input_queue.put(audio_chunk)
                except queue.Empty: break

            try:
                gray_img, _, _ = frame_queue.get(timeout=1.0)
                img = cv2.cvtColor(gray_img, cv2.COLOR_GRAY2BGR)
                results = model(img, stream=True, conf=0.5, verbose=False)
                
                best_face = None
                max_area = 0

                for result in results:
                    boxes = result.boxes.xyxy.cpu().numpy()
                    for box in boxes:
                        x1, y1, x2, y2 = map(int, box)
                        area = (x2 - x1) * (y2 - y1)
                        if area > max_area:
                            max_area = area
                            y1, y2 = max(0, y1), min(CAM_H, y2)
                            x1, x2 = max(0, x1), min(CAM_W, x2)
                            best_face = gray_img[y1:y2, x1:x2]

                if best_face is not None and best_face.size > 0:
                    face_resized = cv2.resize(best_face, (TARGET_SIZE, TARGET_SIZE))
                    
                    # 1. Send to inference engine
                    if inference_engine:
                        norm_face = face_resized.astype(np.float32) / 255.0
                        inference_engine.frame_input_queue.put(norm_face)
                        
                    # 2. Send to GUI face window
                    if gui_face_queue:
                        try:
                            # Use a non-blocking put, clear if full so it stays real-time
                            if gui_face_queue.full():
                                gui_face_queue.get_nowait()
                            gui_face_queue.put_nowait(face_resized)
                        except:
                            pass
                else:
                    display_output = (last_face * 0.5).astype(np.uint8)
                    ### OPTIONAL: Feed the "ghosted" last face to keep the buffer moving
                    if inference_engine:
                         inference_engine.frame_input_queue.put(last_face.astype(np.float32) / 255.0)

                #cv2.imshow(WINDOW_NAME, display_output)

            except queue.Empty:
                continue

            if cv2.waitKey(1) & 0xFF in (ord('q'), 27):
                break
    finally:
        stop_event.set()
        sock.close()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
