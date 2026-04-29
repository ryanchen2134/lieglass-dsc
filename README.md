# Lieglass

What if the truth was always visible, and we just never had the right lens? 

LieGlass is a pair of AR frames built for investigators and reporters who need more than instinct alone, routing live audio and video from XReal One glasses and XReal Eye through a multimodal vision-language model that derives a continuous lie score and surfaces smart prompts directly into the user's field of view. The model is trained and validated on real-world courtroom trial recordings and the DOLOS dataset, grounding its outputs in environments that reflect how reporters and investigators actually work.

Everyone has a tell. Now we have the tools to find it. 🕵️

## Built by: 
**Operation Lead**, Ryan Chen

**Intelligence Agent**, Brian Jin

**The Whisperer**, Ella Kim

**Systems Operative**, Max Pinderski

## Model Architecture
<p align="center">
  <img src="assets/lieglassmodel.jpg" alt="Lie Glass Model Architecture" width="700"/>
</p>

<p align="center">
  <em>
    Architecture overview of the Lie Glass model. AR glasses capture a synchronized video and audio stream, 
    which is processed through two parallel dataflows. Dataflow A extracts visual embeddings via a 
    ViT-B/16 encoder and audio embeddings via a frozen Wav2Vec2 transformer, fusing them through a 
    CrossFusionModule into a deception probability score. Dataflow B transcribes speech with OpenAI 
    Whisper and passes the transcript to an LLM to detect inconsistencies, generating real-time 
    HUD prompts. The model is trained end-to-end with BCE-with-logits loss and label smoothing.
  </em>
</p>

## Project Poster
<p align="center">
  <img src="assets/lieglassposter.png" alt="Lie Glass Poster" width="700"/>
</p>

# lieglass-dsc

All mp4 files are LFS pointers. Make sure to have Git LFS setup correctly before cloning or pulling.

Run: 
# From project root
python -m deception_detection.data.preprocessing.preprocess_resized \
    --manifest Data/manifest_mixed.csv \
    --resized_dir Data/Combined \
    --feature_dir features \
    --workers 4 \
    --force
    
python -m deception_detection.data.preprocessing.extract_frames \
    --manifest Data/manifest_mixed.csv \
    --resized_dir Data/Combined \
    --feature_dir features \
    --workers 4 \
    --force

conda activate LieGlass
cd Shared/
cd Shared\ Vault/
cd Projects/
cd Dammit/
cd lieglass-dsc/

python -m deception_detection.train \
    --manifest Data/manifest_mixed.csv \
    --feature_dir features | tee output.txt
    
    
BRI_WILTY_EP64_truth_1

python Inference.py \
    --video features/BRI_WILTY_EP64_truth_1/video.mp4 \
    --audio features/BRI_WILTY_EP64_truth_1/audio.wav \
    --checkpoint checkpoints/fold_0_best.pt
