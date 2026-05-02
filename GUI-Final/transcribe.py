import threading
import queue
import time
import json
import numpy as np
import whisper
import anthropic
import sounddevice as sd

# --- CONSTANTS ---
WHISPER_RATE = 16000
CHUNK_DURATION = 5         # Transcribe every 5 seconds
MIN_WORDS_FOR_LLM = 30      # Only call Claude if we have at least 20 new words
AUDIO_DEVICE_INDEX = 2      # XREAL Mic Index

class XRealAudioProcessor(threading.Thread):
    def __init__(self, api_key, results_queue):
        super().__init__(daemon=True)
        self.results_queue = results_queue
        self.api_key = api_key
        
        self.stop_event = threading.Event()
        self.master_transcript = ""
        self.last_analyzed_transcript = "" # Track what Claude has already seen
        
        print("Loading Whisper Model (medium.en)...")
        self.whisper_model = whisper.load_model("medium.en")
        self.client = anthropic.Anthropic(api_key=self.api_key)
        self.llm_model = "claude-opus-4-5" # Set to Opus as requested

    def run(self):
        audio_buffer = []
        
        def callback(indata, frames, time_info, status):
            if status:
                print(f"[Audio Status] {status}")
            audio_buffer.append(indata.copy())

        # Specified the device index here
        try:
            with sd.InputStream(samplerate=WHISPER_RATE, 
                                channels=1, 
                                device=AUDIO_DEVICE_INDEX,
                                callback=callback):
                
                print(f"Listening on device {AUDIO_DEVICE_INDEX}...")
                while not self.stop_event.is_set():
                    time.sleep(1) 
                    
                    if len(audio_buffer) > 0:
                        # Collect what we have so far
                        current_raw = np.concatenate(audio_buffer).flatten()
                        
                        # Only clear buffer and process if we have a full chunk
                        if len(current_raw) >= (WHISPER_RATE * CHUNK_DURATION):
                            audio_buffer.clear()
                            self.process_audio(current_raw)
        except Exception as e:
            print(f"Failed to open audio device: {e}")

    def process_audio(self, audio_data):
        # 1. Whisper Transcription
        result = self.whisper_model.transcribe(audio_data, fp16=False, language="en")
        text = result['text'].strip()
        
        if not text:
            return

        self.master_transcript = (self.master_transcript + " " + text).strip()
        print(f"[Transcript]: {text}")

        # 2. Token Saver Logic
        # Check how many words have been added since the last Claude call
        total_words = len(self.master_transcript.split())
        analyzed_words = len(self.last_analyzed_transcript.split())
        new_word_count = total_words - analyzed_words

        if new_word_count >= MIN_WORDS_FOR_LLM:
            print(f"Calling Claude Opus... ({new_word_count} new words)")
            self.run_llm_analysis(self.master_transcript)
            self.last_analyzed_transcript = self.master_transcript
        else:
            print(f"Buffer building: {new_word_count}/{MIN_WORDS_FOR_LLM} words.")

    def run_llm_analysis(self, transcript):
        system_prompt = """
        You are an expert interrogator analyzing a conversation transcript for lies and inconsistencies.
        The transcript has NO speaker labels. Infer who is speaking based on context.
        You will be assisting another user who is driving the conversation. Your role in this interrogation is the following:

        - Notice lies or inconsistencies in the subject's story, remember small details throughout the entirety of the conversation.
        - Provide suggestions on what the user should say, your suggestions should help the user covertly bait the subject into telling more lies / exposing more information.
        - Report on lies and their severity
                
        Here's what lies you should be looking for
        - Logical fallacies
        - Factually incorrect information
        - Half truths
        - Intentionally and extremely overcomplicated speech
        - Misleading speech
        
        You should follow these rules when writing your responses:
        - All answers should be short, less than 20 words, consise and to the point.
        - Only add something to the inconsistencies list if there is something PROVABLY incorrect
            - Speech that "feels like" true or false speech shouldn't be reported
            - Only respond when there is something you KNOW is not true based on your knowledge and the current conversation
        - Whenever you add an inconsistency, the suggestion is not shown. Give suggestions when nothing is happening, report on lies / truths if something shows up.
        - DO NOT REPORT THE SAME LIE TWICE!!!
        - Someone saying 
        Respond ONLY in valid JSON matching this exact schema:
        {
          "inconsistencies": [
            {
              "severity": "MAJOR | MINOR | TRUTH",
              "description": "what the inconsistency is"
            }
          ],
          "suggested_question": "question to ask the subject"
        }
        """
        
        try:
            response = self.client.messages.create(
                model=self.llm_model,
                max_tokens=1000,
                system=system_prompt.strip(),
                messages=[{"role": "user", "content": f"Analyze this transcript:\n\n{transcript}"}]
            )

            raw_text = response.content[0].text
            
            # Robust JSON extraction (finds the first '{' and last '}')
            start_idx = raw_text.find("{")
            end_idx = raw_text.rfind("}") + 1
            
            if start_idx != -1 and end_idx != -1:
                json_str = raw_text[start_idx:end_idx]
                data = json.loads(json_str)
                self.results_queue.put(data)
            else:
                print("Claude didn't return a valid JSON object.")

        except Exception as e:
            print(f"LLM API Error: {e}")