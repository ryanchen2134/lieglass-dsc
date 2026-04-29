import torch
import numpy as np
from collections import deque
import threading
import queue
from Deploy.fusion_model import FusionModel
from Deploy.config import ModelConfig # Assuming your config class is here
import time
from Deploy.config import ModelConfig

class LocalInferenceProcessor(threading.Thread):
    def __init__(self, model_path, result_queue):
        super().__init__(daemon=True)
        self.result_queue = result_queue
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 1. Initialize the config you just sent
        self.config = ModelConfig() 
        
        # 2. Load Model
        self.model = FusionModel(self.config)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()

        # 3. Sliding Window (Tuned to your config)
        # Using 64 frames for a ~2-second "reaction" time
        self.frame_buffer = deque(maxlen=64) 
        self.audio_buffer = deque(maxlen=16000 * 2) 
        
        self.frame_input_queue = queue.Queue(maxsize=128)
        self.audio_input_queue = queue.Queue(maxsize=1000)
        self.stop_event = threading.Event()

    def run(self):
        with torch.no_grad():
            while not self.stop_event.is_set():
                # Drain input queues into sliding window
                while not self.frame_input_queue.empty():
                    self.frame_buffer.append(self.frame_input_queue.get_nowait())
                while not self.audio_input_queue.empty():
                    self.audio_buffer.append(self.audio_input_queue.get_nowait())

                # Requirement: We need at least 64 frames to inference
                if len(self.frame_buffer) >= 64: 
                    # Even if audio is empty, create a dummy silent array so the model doesn't crash
                    if len(self.audio_buffer) < 16000:
                        audio_array = np.zeros(16000, dtype=np.float32)
                    else:
                        audio_array = np.array(list(self.audio_buffer))
                        
                    waveform = torch.from_numpy(audio_array).unsqueeze(0).to(self.device)
                    
                    # 2. Process Frames (Grayscale 224x224)
                    # Config says in_channels=1, so we send (B, N, 1, H, W)
                    frames_list = [torch.from_numpy(f) for f in self.frame_buffer]
                    frames_tensor = torch.stack(frames_list).unsqueeze(0).unsqueeze(2).to(self.device)
                    # Shape is now [1, 64, 1, 224, 224]

                    batch = {
                        "waveform": waveform,
                        "frames": frames_tensor,
                        "waveform_mask": None,
                        "frame_mask": None
                    }

                    # 3. Run Fusion Model
                    logits = self.model(batch)
                    score = torch.sigmoid(logits).item()

                    self.result_queue.put({"score": score})
                    #print("Inference Sent!")
                    
                    # To keep the UI responsive, don't over-saturate the GPU
                    time.sleep(0.01) 
                else:
                    time.sleep(0.1)