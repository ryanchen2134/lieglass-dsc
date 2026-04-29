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
